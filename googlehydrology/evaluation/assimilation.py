"""Model-agnostic Data Assimilation (DA) engine for hydrological forecasting models."""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
import torch
import torch.nn as nn

from googlehydrology.evaluation.metrics import calculate_metrics, get_available_metrics
from googlehydrology.modelzoo.basemodel import BaseModel
from googlehydrology.training import get_loss_obj, get_regularization_obj
from googlehydrology.utils.assimilationconfig import AssimilationConfig
from googlehydrology.utils.cmal_deterministic import calc_cmal_mean, ensure_y_hat

logger = logging.getLogger(__name__)

# Backward-compatibility alias
_ensure_y_hat = ensure_y_hat


def _create_assimilation_optimizer(
    param_groups: List[Dict[str, Any]], cfg: Any
) -> torch.optim.Optimizer:
    """Create optimizer for data assimilation parameter groups without modifying model training contracts."""
    optimizer_name = getattr(cfg, 'optimizer', 'adam').lower()
    default_lr = getattr(cfg, 'initial_learning_rate', getattr(cfg, 'learning_rate', 0.01))
    if isinstance(default_lr, dict):
        default_lr = default_lr.get(0, 0.01)

    if optimizer_name == 'adam':
        return torch.optim.Adam(param_groups, lr=default_lr)
    elif optimizer_name == 'adamw':
        return torch.optim.AdamW(param_groups, lr=default_lr)
    elif optimizer_name == 'sgd':
        return torch.optim.SGD(param_groups, lr=default_lr)
    elif optimizer_name == 'asgd':
        return torch.optim.ASGD(param_groups, lr=default_lr)
    elif optimizer_name == 'rmsprop':
        return torch.optim.RMSprop(param_groups, lr=default_lr)
    elif optimizer_name == 'adagrad':
        return torch.optim.Adagrad(param_groups, lr=default_lr)
    elif optimizer_name == 'adadelta':
        return torch.optim.Adadelta(param_groups, lr=default_lr)
    elif optimizer_name == 'adamax':
        return torch.optim.Adamax(param_groups, lr=default_lr)
    else:
        raise NotImplementedError(
            f'{optimizer_name} not implemented or not supported for Data Assimilation'
        )


def _copy_data_dict(data_dict: dict) -> dict:
    result = {}
    for key, val in data_dict.items():
        if isinstance(val, dict):
            result[key] = _copy_data_dict(val)
        elif isinstance(val, torch.Tensor):
            result[key] = val.clone()
        elif isinstance(val, np.ndarray):
            result[key] = val.copy()
        else:
            result[key] = val
    return result


def _infer_sequence_length(batch_dict: dict) -> Optional[int]:
    """Helper to infer sequence length from target y or 3D feature tensors."""
    if 'y' in batch_dict and isinstance(batch_dict['y'], (torch.Tensor, np.ndarray)) and batch_dict['y'].ndim >= 2:
        return batch_dict['y'].shape[1]
    for val in batch_dict.values():
        if isinstance(val, (torch.Tensor, np.ndarray)) and val.ndim == 3:
            return val.shape[1]
        elif isinstance(val, dict):
            sub_sequence_length = _infer_sequence_length(val)
            if sub_sequence_length is not None:
                return sub_sequence_length
    return None


def _get_variable_learning_rate(learning_rate_config: Any, variable_name: str) -> float:
    """Retrieves target-specific learning rate from learning_rate_config."""
    if isinstance(learning_rate_config, dict):
        if variable_name in learning_rate_config:
            return float(learning_rate_config[variable_name])
        return float(list(learning_rate_config.values())[0])
    elif isinstance(learning_rate_config, (list, tuple)):
        return float(learning_rate_config[0])
    else:
        return float(learning_rate_config)


# Backward-compatibility alias
_get_var_lr = _get_variable_learning_rate


def _slice_hydrology_batch(batch_dict: dict, slice_start_step: int, slice_end_step: int) -> dict:
    """Fast, zero-copy slicing of 3D sequence tensors in a googlehydrology batch dict."""
    non_sequence_keys = {
        'x_s', 'x_one_hot', 'static_features',
        'last_prediction', 'assimilation_overrides',
    }
    hindcast_length, forecast_length = None, None
    if 'x_d_hindcast' in batch_dict and isinstance(batch_dict['x_d_hindcast'], dict):
        for val in batch_dict['x_d_hindcast'].values():
            if isinstance(val, torch.Tensor) and val.ndim == 3:
                hindcast_length = val.shape[1]
                break
    if 'x_d_forecast' in batch_dict and isinstance(batch_dict['x_d_forecast'], dict):
        for val in batch_dict['x_d_forecast'].values():
            if isinstance(val, torch.Tensor) and val.ndim == 3:
                forecast_length = val.shape[1]
                break
    lead_delta = (
        (forecast_length - hindcast_length)
        if (hindcast_length is not None and forecast_length is not None and forecast_length > hindcast_length)
        else 0
    )

    result = {}
    for key, val in batch_dict.items():
        if isinstance(val, dict) and key != 'assimilation_overrides':
            if key in ('x_d_forecast', 'forecast_features'):
                result[key] = _slice_hydrology_batch(val, slice_start_step, slice_end_step + lead_delta)
            else:
                result[key] = _slice_hydrology_batch(val, slice_start_step, slice_end_step)
        elif isinstance(val, torch.Tensor) and key not in non_sequence_keys and val.ndim == 3:
            sequence_length = val.shape[1]
            slice_end_index = (
                min(slice_end_step + lead_delta, sequence_length)
                if key in ('x_d_forecast', 'y')
                else min(slice_end_step, sequence_length)
            )
            result[key] = val[:, min(slice_start_step, sequence_length):slice_end_index, :]
        elif isinstance(val, np.ndarray) and key in ('date', 'y') and val.ndim == 2:
            sequence_length = val.shape[1]
            slice_end_index = min(slice_end_step + lead_delta, sequence_length)
            result[key] = val[:, min(slice_start_step, sequence_length):slice_end_index]
        else:
            result[key] = val
    return result


class _FrozenModelContext:
    """Context manager to evaluate model and freeze its weights during assimilation."""
    def __init__(self, model: nn.Module):
        self.model = model
        self.prev_training = model.training
        self.prev_grad_states = {p: p.requires_grad for p in model.parameters()}

    def __enter__(self):
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        return self.model

    def __exit__(self, exc_type, exc_val, exc_tb):
        for p, state in self.prev_grad_states.items():
            p.requires_grad = state
        self.model.train(self.prev_training)


class Assimilation(object):
    """Model-agnostic Data Assimilation (DA) engine for hydrological forecasting models."""

    def __init__(self, cfg: AssimilationConfig):
        self.cfg = cfg
        self.assimilation_window_length = getattr(
            cfg, 'assimilation_window_length', getattr(cfg, 'assimilation_window', 1)
        )
        self.window = self.assimilation_window_length
        self.history = getattr(cfg, 'history', 1)
        self.assimilation_lead_time = getattr(cfg, 'assimilation_lead_time', 0)
        self.lead_time = self.assimilation_lead_time
        self.epochs = getattr(cfg, 'epochs', 10)

        self.assimilation_components = getattr(cfg, 'assimilation_components', {})
        self.targets = getattr(cfg, 'assimilation_targets', list(self.assimilation_components.keys()))

        # Sequence boundary steps
        self.assimilation_end_step = cfg.seq_length - self.assimilation_lead_time
        self.assimilation_start_step = max(
            0, self.assimilation_end_step - (self.history * self.assimilation_window_length)
        )
        self._start_timestep = self.assimilation_start_step
        self._end_timestep = self.assimilation_end_step

        if self.assimilation_end_step > cfg.seq_length:
            raise ValueError("Warmup + assimilation period cannot exceed total sequence length.")

        self._loss_obj = get_loss_obj(cfg)
        self._loss_obj.set_regularization_terms(get_regularization_obj(cfg=cfg))

    def validate_data_structure(self, data: Dict[str, Any]):
        """Validates that required keys exist in the input data dictionary."""
        if 'y' not in data:
            raise KeyError("[DA Validation Error] Missing required key 'y'.")
        if 'x_d_hindcast' not in data and 'x_d' not in data:
            raise KeyError("[DA Validation Error] Batch must contain 'x_d_hindcast' or 'x_d'.")

    def check_discharge_timing(self, data: Dict[str, Any], verbose: bool = True) -> Dict[str, Any]:
        """Checks for timing mismatches between feature series and targets."""
        diagnostics = {'has_timing_mismatch': False, 'warnings': [], 'details': {}}
        if 'date' in data:
            d = data['date']
            start_date = str(d[0, 0]) if (isinstance(d, np.ndarray) and d.ndim >= 2) else (str(d[0]) if hasattr(d, '__getitem__') else str(d))
            diagnostics['details']['sequence_start_date'] = start_date

        x_d = data.get('x_d', data.get('x_d_hindcast', None))
        y = data.get('y', None)

        if x_d is not None and y is not None:
            x_d_dict = x_d if isinstance(x_d, dict) else {'x_d': x_d}
            for feat_name, feat_val in x_d_dict.items():
                match = re.search(r'^(.*)_shift(\d+)$', feat_name)
                if match or 'streamflow' in feat_name or 'discharge' in feat_name:
                    shift = int(match.group(2)) if match else 1
                    f_tensor = torch.as_tensor(feat_val) if not isinstance(feat_val, torch.Tensor) else feat_val
                    y_tensor = torch.as_tensor(y) if not isinstance(y, torch.Tensor) else y

                    check_t = min(5, y_tensor.shape[1] - 1)
                    if check_t >= shift:
                        val_at_t = f_tensor[0, check_t, 0] if f_tensor.ndim == 3 else f_tensor[0, check_t]
                        y_same = y_tensor[0, check_t, 0] if y_tensor.ndim == 3 else y_tensor[0, check_t]
                        y_prev = y_tensor[0, check_t - shift, 0] if y_tensor.ndim == 3 else y_tensor[0, check_t - shift]

                        if not (torch.isnan(val_at_t) or torch.isnan(y_same) or torch.isnan(y_prev)):
                            if torch.abs(val_at_t - y_same).item() < 1e-5 and torch.abs(val_at_t - y_prev).item() > 1e-4:
                                diagnostics['has_timing_mismatch'] = True
                                diagnostics['warnings'].append(f"TIMING MISMATCH DETECTED in '{feat_name}' at t={check_t}.")
        return diagnostics

    def _get_active_components(self, model: Optional[BaseModel] = None) -> Dict[str, Dict[str, Any]]:
        """Returns normalized component specifications: {comp_name: {'weight': float, 'lr': float}}."""
        raw_components = getattr(self.cfg, 'assimilation_components', {})
        if not raw_components and hasattr(self, 'assimilation_components'):
            raw_components = self.assimilation_components

        if not raw_components:
            if self.targets:
                comp_names = [str(t) for t in self.targets] if isinstance(self.targets, (list, tuple)) else [str(self.targets)]
            elif model is not None and hasattr(model, 'get_supported_assimilation_components'):
                comp_names = model.get_supported_assimilation_components()
            else:
                comp_names = []
            raw_components = {name: {} for name in comp_names}

        normalized_components = {}
        for comp_name, comp_cfg in raw_components.items():
            if isinstance(comp_cfg, (int, float)):
                weight = float(comp_cfg)
                lr = None
            elif isinstance(comp_cfg, dict):
                weight = comp_cfg.get('weight', None)
                lr = comp_cfg.get('lr', comp_cfg.get('learning_rate', None))
            else:
                weight = None
                lr = None

            if weight is None:
                weight = getattr(
                    self.cfg,
                    f'{comp_name}_regularization_weight',
                    getattr(self.cfg, 'regularization_weight', 0.01),
                )
            if lr is None:
                lr = _get_var_lr(self.cfg.learning_rate, comp_name)

            normalized_components[comp_name] = {
                'weight': float(weight),
                'lr': float(lr),
            }

        return normalized_components

    def assimilate(self, model: BaseModel, data: Dict[str, torch.Tensor], verbose: bool = False, **kwargs) -> Dict[str, Any]:
        self.validate_data_structure(data)
        if kwargs.get('check_timing', False) or verbose:
            self.check_discharge_timing(data, verbose=verbose)

        active_components = self._get_active_components(model)

        with _FrozenModelContext(model):
            observed_discharge = data['y'] if data['y'].ndim == 3 else data['y'].unsqueeze(-1)
            total_sequence_length = observed_discharge.shape[1]
            num_target_features = observed_discharge.shape[-1]

            assimilation_start_step = self.assimilation_start_step
            assimilation_end_step = self.assimilation_end_step

            assimilated_discharge_predictions = []
            distribution_parameter_chunks = {k: [] for k in ['mu', 'b', 'tau', 'pi']}
            prior_discharge_predictions = []

            # =========================================================================
            # PHASE 1: Warmup Phase (0 -> assimilation_start_step)
            # =========================================================================
            persistent_components: Dict[str, torch.Tensor] = {}
            last_prediction_curr = data.get('last_prediction', None)

            if assimilation_start_step > 0:
                warmup_data = _slice_hydrology_batch(data, 0, assimilation_start_step)
                with torch.no_grad():
                    warmup_out = ensure_y_hat(model(warmup_data), use_median=True)
                    base_y_warmup = warmup_out['y_hat']
                    if base_y_warmup.ndim == 2:
                        base_y_warmup = base_y_warmup.unsqueeze(-1)
                    if base_y_warmup.shape[-1] > num_target_features:
                        base_y_warmup = base_y_warmup[..., :num_target_features]
                    assimilated_discharge_predictions.append(base_y_warmup[:, :assimilation_start_step, :])

                    for key in ['mu', 'b', 'tau', 'pi']:
                        if key in warmup_out and isinstance(warmup_out[key], torch.Tensor):
                            distribution_parameter_chunks[key].append(warmup_out[key][:, :assimilation_start_step, ...])

                    if 'last_prediction' in warmup_out:
                        last_prediction_curr = warmup_out['last_prediction']

            last_optimized_components: Dict[str, torch.Tensor] = {}

            # =========================================================================
            # PHASE 2: Sequential Window Optimization (assimilation_start_step -> assimilation_end_step)
            # =========================================================================
            current_step_index = assimilation_start_step
            for _ in range(self.history):
                if current_step_index >= assimilation_end_step:
                    break
                window_end_step = min(current_step_index + self.assimilation_window_length, assimilation_end_step)
                window_length = window_end_step - current_step_index

                chunk_data = _slice_hydrology_batch(data, current_step_index, window_end_step)
                for comp_name, comp_val in persistent_components.items():
                    chunk_data[comp_name] = comp_val
                    chunk_data.setdefault('assimilation_overrides', {})[comp_name] = comp_val
                if last_prediction_curr is not None:
                    chunk_data['last_prediction'] = last_prediction_curr

                # Pre-optimization baseline pass
                with torch.no_grad():
                    prior_out = ensure_y_hat(model(chunk_data), use_median=True)
                    prior_predicted_discharge_window = prior_out['y_hat'][:, :window_length, :]
                    if prior_predicted_discharge_window.ndim == 2:
                        prior_predicted_discharge_window = prior_predicted_discharge_window.unsqueeze(-1)
                    if prior_predicted_discharge_window.shape[-1] > num_target_features:
                        prior_predicted_discharge_window = prior_predicted_discharge_window[..., :num_target_features]
                    prior_discharge_predictions.append(prior_predicted_discharge_window)

                    baseline_components = {}
                    for comp_name in active_components:
                        if comp_name in prior_out and prior_out[comp_name] is not None:
                            baseline_components[comp_name] = prior_out[comp_name]
                        elif comp_name in chunk_data and chunk_data[comp_name] is not None:
                            baseline_components[comp_name] = chunk_data[comp_name]

                # Setup learnable parameters and optimizer
                parameters_to_optimize = []
                param_groups = []
                optimized_components = {}

                for comp_name, comp_cfg in active_components.items():
                    if comp_name in baseline_components:
                        base_tensor = baseline_components[comp_name]
                        opt_tensor = base_tensor.clone().detach().requires_grad_(True)
                        optimized_components[comp_name] = opt_tensor
                        parameters_to_optimize.append(opt_tensor)
                        comp_lr = comp_cfg['lr']
                        param_groups.append({
                            'params': [opt_tensor],
                            'lr': comp_lr,
                            'base_lr': comp_lr,
                        })

                if parameters_to_optimize:
                    optimizer = _create_assimilation_optimizer(param_groups, self.cfg)

                    drop_factor = getattr(self.cfg, 'learning_rate_drop_factor', 0.9)
                    epoch_drop = getattr(self.cfg, 'learning_rate_epoch_drop', 5)

                    for epoch in range(self.epochs):
                        # Learning rate decay schedule
                        if epoch_drop > 0 and epoch > 0:
                            current_factor = drop_factor ** (epoch // epoch_drop)
                            for pg in optimizer.param_groups:
                                pg['lr'] = pg['base_lr'] * current_factor

                        optimizer.zero_grad()
                        for comp_name, opt_tensor in optimized_components.items():
                            chunk_data[comp_name] = opt_tensor
                            chunk_data.setdefault('assimilation_overrides', {})[comp_name] = opt_tensor
                        if last_prediction_curr is not None:
                            chunk_data['last_prediction'] = last_prediction_curr

                        pred_dict = ensure_y_hat(model(chunk_data), use_median=False)
                        predicted_discharge_window = pred_dict['y_hat'][:, :window_length, :]
                        if predicted_discharge_window.ndim == 2:
                            predicted_discharge_window = predicted_discharge_window.unsqueeze(-1)
                        if predicted_discharge_window.shape[-1] > num_target_features:
                            predicted_discharge_window = predicted_discharge_window[..., :num_target_features]

                        observed_discharge_window = chunk_data['y'][:, :window_length, :]
                        if observed_discharge_window.ndim == 2:
                            observed_discharge_window = observed_discharge_window.unsqueeze(-1)
                        if observed_discharge_window.shape[-1] > num_target_features:
                            observed_discharge_window = observed_discharge_window[..., :num_target_features]

                        valid_mask = ~torch.isnan(observed_discharge_window) & ~torch.isnan(predicted_discharge_window)
                        if valid_mask.any():
                            prediction_loss = torch.mean(
                                (predicted_discharge_window[valid_mask] - observed_discharge_window[valid_mask]) ** 2
                            )
                            regularization_loss = 0.0
                            for comp_name, opt_tensor in optimized_components.items():
                                base_tensor = baseline_components[comp_name]
                                reg_weight = active_components[comp_name]['weight']
                                # Regularization loss uses torch.mean to ensure scale-invariance across dimensions and window lengths
                                regularization_loss = regularization_loss + reg_weight * torch.mean((opt_tensor - base_tensor) ** 2)

                            total_loss = prediction_loss + regularization_loss
                            if torch.isfinite(total_loss) and total_loss.requires_grad:
                                total_loss.backward()
                                clip_norm = getattr(self.cfg, 'clip_gradient_norm', 0.0)
                                if clip_norm > 0:
                                    torch.nn.utils.clip_grad_norm_(parameters_to_optimize, clip_norm)
                                optimizer.step()
                        else:
                            logger.debug(
                                'Window [%d:%d] contains 0 valid target observations. Bypassing gradient update.',
                                current_step_index, window_end_step
                            )

                # Post-optimization rollout for current window
                with torch.no_grad():
                    for comp_name, opt_tensor in optimized_components.items():
                        opt_detached = opt_tensor.detach()
                        chunk_data[comp_name] = opt_detached
                        chunk_data.setdefault('assimilation_overrides', {})[comp_name] = opt_detached
                        last_optimized_components[comp_name] = opt_detached

                        # If component is time-invariant (e.g. 2D [batch, dim] with no sequence length dimension),
                        # persist it across subsequent windows and post-DA forecast horizons
                        if opt_detached.ndim <= 2:
                            persistent_components[comp_name] = opt_detached

                    rollout = ensure_y_hat(model(chunk_data), use_median=True)
                    if 'last_prediction' in rollout:
                        last_prediction_curr = rollout['last_prediction']

                    predicted_rollout = rollout['y_hat'][:, :window_length, :]
                    if predicted_rollout.ndim == 2:
                        predicted_rollout = predicted_rollout.unsqueeze(-1)
                    if predicted_rollout.shape[-1] > num_target_features:
                        predicted_rollout = predicted_rollout[..., :num_target_features]
                    assimilated_discharge_predictions.append(predicted_rollout)

                    for key in ['mu', 'b', 'tau', 'pi']:
                        if key in rollout and isinstance(rollout[key], torch.Tensor):
                            distribution_parameter_chunks[key].append(rollout[key][:, :window_length, ...])

                current_step_index = window_end_step

            # =========================================================================
            # PHASE 3: Post-DA Forecast Horizon (assimilation_end_step -> total_sequence_length)
            # =========================================================================
            if current_step_index < total_sequence_length:
                with torch.no_grad():
                    forecast_data = _slice_hydrology_batch(data, current_step_index, total_sequence_length)
                    for comp_name, comp_val in persistent_components.items():
                        forecast_data[comp_name] = comp_val
                        forecast_data.setdefault('assimilation_overrides', {})[comp_name] = comp_val
                    if last_prediction_curr is not None:
                        forecast_data['last_prediction'] = last_prediction_curr

                    forecast_out = ensure_y_hat(model(forecast_data), use_median=True)
                    forecast_y = forecast_out['y_hat'][:, :(total_sequence_length - current_step_index), :]
                    if forecast_y.ndim == 2:
                        forecast_y = forecast_y.unsqueeze(-1)
                    if forecast_y.shape[-1] > num_target_features:
                        forecast_y = forecast_y[..., :num_target_features]
                    assimilated_discharge_predictions.append(forecast_y)

                    for key in ['mu', 'b', 'tau', 'pi']:
                        if key in forecast_out and isinstance(forecast_out[key], torch.Tensor):
                            distribution_parameter_chunks[key].append(
                                forecast_out[key][:, :(total_sequence_length - current_step_index), ...]
                            )

            final_predicted_discharge = torch.cat(assimilated_discharge_predictions, dim=1)[:, :total_sequence_length, :]

            # Pre/Post Hindcast Evaluation Metrics
            with torch.no_grad():
                evaluation_window_length = min(assimilation_end_step - assimilation_start_step, observed_discharge.shape[1])
                if evaluation_window_length > 0 and prior_discharge_predictions:
                    target_hindcast = observed_discharge[:, assimilation_start_step:assimilation_start_step + evaluation_window_length, :]
                    prior_hindcast = torch.cat(prior_discharge_predictions, dim=1)[:, :evaluation_window_length, :]
                    post_hindcast = final_predicted_discharge[:, assimilation_start_step:assimilation_start_step + evaluation_window_length, :]

                    if prior_hindcast.shape[-1] > num_target_features:
                        prior_hindcast = prior_hindcast[..., :num_target_features]
                    if post_hindcast.shape[-1] > num_target_features:
                        post_hindcast = post_hindcast[..., :num_target_features]

                    valid_len = min(target_hindcast.shape[1], prior_hindcast.shape[1], post_hindcast.shape[1])
                    if valid_len > 0:
                        target_hindcast = target_hindcast[:, :valid_len, :]
                        prior_hindcast = prior_hindcast[:, :valid_len, :]
                        post_hindcast = post_hindcast[:, :valid_len, :]
                        mask_pre = ~torch.isnan(target_hindcast) & ~torch.isnan(prior_hindcast)
                        mask_post = ~torch.isnan(target_hindcast) & ~torch.isnan(post_hindcast)
                    else:
                        mask_pre = torch.zeros(1, dtype=torch.bool)
                        mask_post = torch.zeros(1, dtype=torch.bool)
                else:
                    valid_len = 0
                    prior_hindcast, post_hindcast, target_hindcast = None, None, None
                    mask_pre = torch.zeros(1, dtype=torch.bool)
                    mask_post = torch.zeros(1, dtype=torch.bool)

                def _compute_metrics(simulation, observation, mask):
                    if not mask.any() or valid_len == 0 or simulation is None or observation is None:
                        return {'MSE': float('nan'), 'NSE': float('nan')}
                    sim_valid, obs_valid = simulation[mask], observation[mask]
                    mean_squared_error = torch.mean((sim_valid - obs_valid) ** 2).item()
                    variance_observed = torch.var(obs_valid, unbiased=False).item()
                    nash_sutcliffe_efficiency = (
                        1.0 - (mean_squared_error / (variance_observed + 1e-6))
                        if variance_observed > 1e-8
                        else float('nan')
                    )
                    return {'MSE': mean_squared_error, 'NSE': nash_sutcliffe_efficiency}

                metrics_pre = _compute_metrics(prior_hindcast, target_hindcast, mask_pre)
                metrics_post = _compute_metrics(post_hindcast, target_hindcast, mask_post)

            results = {
                'y_hat': final_predicted_discharge.detach(),
                'hindcast_metrics_pre': metrics_pre,
                'hindcast_metrics_post': metrics_post,
            }
            for comp_name, opt_tensor in last_optimized_components.items():
                results[comp_name] = opt_tensor

            for key, chunks in distribution_parameter_chunks.items():
                if chunks:
                    results[key] = torch.cat(chunks, dim=1)[:, :total_sequence_length, ...]

            return results
