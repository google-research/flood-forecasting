"""Unified Data Assimilation (DA) engine for hydrological forecasting models."""

import logging
import re
from typing import Any, Dict, Optional, Tuple
import warnings

import numpy as np
import torch
import torch.nn as nn

from googlehydrology.evaluation.metrics import calculate_metrics, get_available_metrics
from googlehydrology.modelzoo.basemodel import BaseModel
from googlehydrology.modelzoo.head import calc_cmal_mean, ensure_y_hat
from googlehydrology.training import get_loss_obj, get_optimizer, get_regularization_obj
from googlehydrology.utils.assimilationconfig import AssimilationConfig

logger = logging.getLogger(__name__)

# Backward-compatibility alias
_ensure_y_hat = ensure_y_hat


def _copy_data_dict(d: dict) -> dict:
    res = {}
    for k, v in d.items():
        if isinstance(v, dict):
            res[k] = _copy_data_dict(v)
        elif isinstance(v, torch.Tensor):
            res[k] = v.clone()
        elif isinstance(v, np.ndarray):
            res[k] = v.copy()
        else:
            res[k] = v
    return res


def _infer_seq_len(d: dict) -> int | None:
    """Helper to infer sequence length from target y or 3D feature tensors."""
    if 'y' in d and isinstance(d['y'], (torch.Tensor, np.ndarray)) and d['y'].ndim >= 2:
        return d['y'].shape[1]
    for v in d.values():
        if isinstance(v, (torch.Tensor, np.ndarray)) and v.ndim == 3:
            return v.shape[1]
        elif isinstance(v, dict):
            sub_len = _infer_seq_len(v)
            if sub_len is not None:
                return sub_len
    return None


def _get_var_lr(lr_cfg: Any, var_name: str) -> float:
    """Retrieves target-specific learning rate from lr_cfg."""
    if isinstance(lr_cfg, dict):
        if var_name in lr_cfg:
            return float(lr_cfg[var_name])
        norm_name = 'c_n' if 'c' in var_name else ('h_n' if 'h' in var_name else var_name)
        if norm_name in lr_cfg:
            return float(lr_cfg[norm_name])
        return float(list(lr_cfg.values())[0])
    elif isinstance(lr_cfg, (list, tuple)):
        return float(lr_cfg[0])
    else:
        return float(lr_cfg)


def _slice_hydrology_batch(d: dict, slice_start: int, slice_end: int) -> dict:
    """Fast, zero-copy slicing of 3D sequence tensors in a googlehydrology batch dict."""
    non_seq_keys = {
        'x_s', 'x_one_hot', 'static_features',
        'c_n', 'h_n', 'c_0', 'h_0',
        'c_0_hindcast', 'h_0_hindcast', 'c_0_forecast', 'h_0_forecast',
        'last_prediction', 'static_embedding',
    }
    h_len, f_len = None, None
    if 'x_d_hindcast' in d and isinstance(d['x_d_hindcast'], dict):
        for v in d['x_d_hindcast'].values():
            if isinstance(v, torch.Tensor) and v.ndim == 3:
                h_len = v.shape[1]
                break
    if 'x_d_forecast' in d and isinstance(d['x_d_forecast'], dict):
        for v in d['x_d_forecast'].values():
            if isinstance(v, torch.Tensor) and v.ndim == 3:
                f_len = v.shape[1]
                break
    lead_delta = (f_len - h_len) if (h_len is not None and f_len is not None and f_len > h_len) else 0

    res = {}
    for k, v in d.items():
        if isinstance(v, dict):
            if k in ('x_d_forecast', 'forecast_features'):
                res[k] = _slice_hydrology_batch(v, slice_start, slice_end + lead_delta)
            else:
                res[k] = _slice_hydrology_batch(v, slice_start, slice_end)
        elif isinstance(v, torch.Tensor) and k not in non_seq_keys and v.ndim == 3:
            t_len = v.shape[1]
            end_idx = min(slice_end + lead_delta, t_len) if k in ('x_d_forecast', 'y') else min(slice_end, t_len)
            res[k] = v[:, min(slice_start, t_len):end_idx, :]
        elif isinstance(v, np.ndarray) and k in ('date', 'y') and v.ndim == 2:
            t_len = v.shape[1]
            end_idx = min(slice_end + lead_delta, t_len)
            res[k] = v[:, min(slice_start, t_len):end_idx]
        else:
            res[k] = v
    return res


class _FrozenModelContext:
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
    """Unified Data Assimilation (DA) state and parameter updating for hydrological forecasting models."""

    def __init__(self, cfg: AssimilationConfig):
        self.cfg = cfg
        self.window = getattr(cfg, 'assimilation_window', 1)
        self.history = getattr(cfg, 'history', 1)
        self.lead_time = getattr(cfg, 'assimilation_lead_time', 0)
        self.epochs = getattr(cfg, 'epochs', 10)
        self.targets = getattr(cfg, 'assimilation_targets', ['c_n'])

        # Boundary timesteps
        self._end_timestep = cfg.seq_length - self.lead_time
        self._start_timestep = max(0, self._end_timestep - (self.history * self.window))

        if self._end_timestep > cfg.seq_length:
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

    def _parse_target_flags(self) -> Tuple[bool, bool, bool, bool]:
        """Parses self.targets into boolean flags for (c_hc, h_hc, c_fc, h_fc)."""
        opt_c_hc = any(k in self.targets for k in ['c_n_hindcast', 'c_0_hindcast', 'c_hc', 'c_n', 'c_0'])
        opt_h_hc = any(k in self.targets for k in ['h_n_hindcast', 'h_0_hindcast', 'h_hc', 'h_n', 'h_0'])
        opt_c_fc = any(k in self.targets for k in ['c_n_forecast', 'c_0_forecast', 'c_fc', 'c_n', 'c_0'])
        opt_h_fc = any(k in self.targets for k in ['h_n_forecast', 'h_0_forecast', 'h_fc', 'h_n', 'h_0'])
        return opt_c_hc, opt_h_hc, opt_c_fc, opt_h_fc

    def _detect_da_type(self) -> str:
        """Determines the DA mode from self.targets: 'embedding', 'precip', or 'state'."""
        targets = [str(t).lower() for t in self.targets] if isinstance(self.targets, (list, tuple)) else [str(self.targets).lower()]
        target_str = ' '.join(targets)
        if any(k in target_str for k in ['embedded', 'embedding', 'static_embedding', 'hindcast_embedding', 'forecast_embedding']):
            return 'embedding'
        elif any(k in target_str for k in ['precip', 'precipitation', 'forcing', 'rain', 'tp']):
            return 'precip'
        return 'state'

    def _parse_embedding_masks(self) -> Tuple[bool, bool, bool]:
        """Returns (mask_e_stat, mask_e_dyn, mask_e_fc) based on configured targets."""
        targets = [str(t).lower() for t in self.targets] if isinstance(self.targets, (list, tuple)) else [str(self.targets).lower()]
        target_str = ' '.join(targets)
        mask_e_stat = any(k in target_str for k in ['stat', 'both', 'all', 'embedding'])
        mask_e_dyn = any(k in target_str for k in ['dyn', 'both', 'all', 'temporal', 'hc', 'hindcast'])
        mask_e_fc = any(k in target_str for k in ['all', 'three', 'fc', 'forecast'])
        return mask_e_stat, mask_e_dyn, mask_e_fc

    def _extract_state_tensor(self, state_val: Optional[torch.Tensor], t_idx: int) -> Optional[torch.Tensor]:
        if state_val is None:
            return None
        if state_val.ndim == 4:  # [num_layers, batch, seq_len, hidden_size]
            t_clamp = min(max(0, t_idx), state_val.shape[2] - 1)
            return state_val[:, :, t_clamp, :].detach().clone()
        elif state_val.ndim == 3:  # [num_layers, batch, hidden_size] or [batch, 1, hidden_size]
            return state_val.detach().clone()
        return None

    def assimilate(self, model: BaseModel, data: Dict[str, torch.Tensor], verbose: bool = False, **kwargs) -> Dict[str, Any]:
        self.validate_data_structure(data)
        if kwargs.get('check_timing', False) or verbose:
            self.check_discharge_timing(data, verbose=verbose)

        da_type = self._detect_da_type()
        mask_e_stat, mask_e_dyn, mask_e_fc = (False, False, False)
        opt_c_hc, opt_h_hc, opt_c_fc, opt_h_fc = self._parse_target_flags()

        with _FrozenModelContext(model):
            y_tensor = data['y'] if data['y'].ndim == 3 else data['y'].unsqueeze(-1)
            batch_size = y_tensor.shape[0]
            total_len = y_tensor.shape[1]

            hidden_size = (
                getattr(model, 'hidden_size', None)
                or getattr(getattr(model, 'config_data', None), 'hidden_size', None)
                or getattr(getattr(model, 'cfg', None), 'hidden_size', None)
                or getattr(getattr(model, 'hindcast_lstm', None), 'hidden_size', None)
                or getattr(self.cfg, 'hidden_size', 64)
            )
            if isinstance(hidden_size, dict):
                hidden_size = list(hidden_size.values())[0]
            hidden_size = int(hidden_size)
            device = y_tensor.device

            a_start = self._start_timestep
            a_end = self._end_timestep

            y_chunks = []
            dist_chunks = {k: [] for k in ['mu', 'b', 'tau', 'pi']}
            p_pre_chunks = []

            target_key = self.targets[0] if self.targets else 'c_n'
            lr = _get_var_lr(self.cfg.learning_rate, target_key)

            c_user = data.get('c_0', data.get('c_n', None))
            h_user = data.get('h_0', data.get('h_n', None))

            # =========================================================================
            # PHASE 1: Warmup Phase (0 -> a_start)
            # =========================================================================
            static_embedding_opt = None
            if a_start > 0:
                warmup_data = _slice_hydrology_batch(data, 0, a_start)
                if c_user is not None: warmup_data['c_0'] = c_user
                if h_user is not None: warmup_data['h_0'] = h_user

                with torch.no_grad():
                    warmup_out = ensure_y_hat(model(warmup_data), use_median=True)
                    base_y_warmup = warmup_out['y_hat']
                    if base_y_warmup.ndim == 2:
                        base_y_warmup = base_y_warmup.unsqueeze(-1)
                    y_chunks.append(base_y_warmup[:, :a_start, :])

                    for key in ['mu', 'b', 'tau', 'pi']:
                        if key in warmup_out and isinstance(warmup_out[key], torch.Tensor):
                            dist_chunks[key].append(warmup_out[key][:, :a_start, ...])

                    c_hc_curr = warmup_out.get('c_n_hindcast', warmup_out.get('c_n'))
                    h_hc_curr = warmup_out.get('h_n_hindcast', warmup_out.get('h_n'))
                    c_fc_curr = warmup_out.get('c_n_forecast', warmup_out.get('c_n', c_hc_curr))
                    h_fc_curr = warmup_out.get('h_n_forecast', warmup_out.get('h_n', h_hc_curr))
            else:
                c_hc_curr = c_user.detach().clone() if c_user is not None else torch.zeros(1, batch_size, hidden_size, device=device)
                h_hc_curr = h_user.detach().clone() if h_user is not None else torch.zeros(1, batch_size, hidden_size, device=device)
                c_fc_curr = c_hc_curr.clone()
                h_fc_curr = h_hc_curr.clone()

            if c_hc_curr is None: c_hc_curr = torch.zeros(1, batch_size, hidden_size, device=device)
            if h_hc_curr is None: h_hc_curr = torch.zeros(1, batch_size, hidden_size, device=device)
            if c_fc_curr is None: c_fc_curr = c_hc_curr.clone()
            if h_fc_curr is None: h_fc_curr = h_hc_curr.clone()

            last_prediction_curr = data.get('last_prediction', None)

            last_e_stat = None
            last_e_dyn = None
            last_e_fc = None
            last_p_opt = None
            last_p_opts = None

            # =========================================================================
            # PHASE 2: Sequential Window Optimization (a_start -> a_end)
            # =========================================================================
            curr_idx = a_start
            for w_idx in range(self.history):
                if curr_idx >= a_end: break
                w_end = min(curr_idx + self.window, a_end)
                win_len = w_end - curr_idx

                chunk_data = _slice_hydrology_batch(data, curr_idx, w_end)
                chunk_data['c_0_hindcast'] = c_hc_curr
                chunk_data['h_0_hindcast'] = h_hc_curr
                chunk_data['c_0_forecast'] = c_fc_curr
                chunk_data['h_0_forecast'] = h_fc_curr
                chunk_data['c_0'] = c_hc_curr
                chunk_data['h_0'] = h_hc_curr
                chunk_data['c_n'] = c_hc_curr
                chunk_data['h_n'] = h_hc_curr
                if static_embedding_opt is not None:
                    chunk_data['static_embedding'] = static_embedding_opt
                if last_prediction_curr is not None:
                    chunk_data['last_prediction'] = last_prediction_curr

                # -------------------------------------------------------------
                # 2C. RECURRENT STATE DATA ASSIMILATION (DEFAULT)
                # -------------------------------------------------------------
                if True:
                    c_hc_opt = c_hc_curr.clone().detach().requires_grad_(True) if (opt_c_hc and c_hc_curr is not None) else (c_hc_curr.clone().detach() if c_hc_curr is not None else None)
                    h_hc_opt = h_hc_curr.clone().detach().requires_grad_(True) if (opt_h_hc and h_hc_curr is not None) else (h_hc_curr.clone().detach() if h_hc_curr is not None else None)
                    c_fc_opt = c_fc_curr.clone().detach().requires_grad_(True) if (opt_c_fc and c_fc_curr is not None) else (c_fc_curr.clone().detach() if c_fc_curr is not None else None)
                    h_fc_opt = h_fc_curr.clone().detach().requires_grad_(True) if (opt_h_fc and h_fc_curr is not None) else (h_fc_curr.clone().detach() if h_fc_curr is not None else None)

                    opt_vars = [p for p in (c_hc_opt, h_hc_opt, c_fc_opt, h_fc_opt) if p is not None and p.requires_grad]

                    with torch.no_grad():
                        pre_pred = ensure_y_hat(model(chunk_data), use_median=True)
                        p_pre_sub = pre_pred['y_hat'][:, :win_len, :]
                        if p_pre_sub.ndim == 2: p_pre_sub = p_pre_sub.unsqueeze(-1)
                        p_pre_chunks.append(p_pre_sub)

                    if opt_vars:
                        optimizer = get_optimizer(opt_vars, self.cfg)
                        for pg in optimizer.param_groups: pg["lr"] = lr

                        for epoch in range(self.epochs):
                            optimizer.zero_grad()

                            chunk_data['c_0_hindcast'] = c_hc_opt
                            chunk_data['h_0_hindcast'] = h_hc_opt
                            chunk_data['c_0_forecast'] = c_fc_opt
                            chunk_data['h_0_forecast'] = h_fc_opt
                            chunk_data['c_0'] = c_hc_opt if opt_c_hc else (c_fc_opt if opt_c_fc else c_hc_opt)
                            chunk_data['h_0'] = h_hc_opt if opt_h_hc else (h_fc_opt if opt_h_fc else h_hc_opt)
                            chunk_data['c_n'] = chunk_data['c_0']
                            chunk_data['h_n'] = chunk_data['h_0']
                            if last_prediction_curr is not None:
                                chunk_data['last_prediction'] = last_prediction_curr

                            pred_dict = ensure_y_hat(model(chunk_data), use_median=False)
                            p_sub = pred_dict['y_hat'][:, :win_len, :]
                            if p_sub.ndim == 2: p_sub = p_sub.unsqueeze(-1)
                            t_sub = chunk_data['y'][:, :win_len, :]

                            mask = ~torch.isnan(t_sub) & ~torch.isnan(p_sub)
                            if mask.any():
                                loss = torch.mean((p_sub[mask] - t_sub[mask]) ** 2)
                                reg_weight = getattr(self.cfg, 'bg_regularization_weight', getattr(self.cfg, 'regularization_weight', 0.0)) or 0.0
                                if reg_weight > 0:
                                    reg_loss = 0.0
                                    if opt_c_hc and c_hc_curr is not None: reg_loss = reg_loss + torch.sum((c_hc_opt - c_hc_curr) ** 2)
                                    if opt_h_hc and h_hc_curr is not None: reg_loss = reg_loss + torch.sum((h_hc_opt - h_hc_curr) ** 2)
                                    if opt_c_fc and c_fc_curr is not None: reg_loss = reg_loss + torch.sum((c_fc_opt - c_fc_curr) ** 2)
                                    if opt_h_fc and h_fc_curr is not None: reg_loss = reg_loss + torch.sum((h_fc_opt - h_fc_curr) ** 2)
                                    loss = loss + reg_weight * reg_loss

                                if torch.isfinite(loss) and loss.requires_grad:
                                    loss.backward()
                                    if getattr(self.cfg, 'clip_gradient_norm', 0) > 0:
                                        torch.nn.utils.clip_grad_norm_(opt_vars, self.cfg.clip_gradient_norm)
                                    optimizer.step()
                            else:
                                logger.debug('Window [%d:%d] contains 0 valid target observations. Bypassing gradient update.', curr_idx, w_end)

                    with torch.no_grad():
                        chunk_data['c_0_hindcast'] = c_hc_opt.detach() if c_hc_opt is not None else None
                        chunk_data['h_0_hindcast'] = h_hc_opt.detach() if h_hc_opt is not None else None
                        chunk_data['c_0_forecast'] = c_fc_opt.detach() if c_fc_opt is not None else None
                        chunk_data['h_0_forecast'] = h_fc_opt.detach() if h_fc_opt is not None else None
                        chunk_data['c_0'] = chunk_data['c_0_hindcast']
                        chunk_data['h_0'] = chunk_data['h_0_hindcast']
                        chunk_data['c_n'] = chunk_data['c_0']
                        chunk_data['h_n'] = chunk_data['h_0']
                        if last_prediction_curr is not None:
                            chunk_data['last_prediction'] = last_prediction_curr

                        rollout = ensure_y_hat(model(chunk_data), use_median=True)
                        if 'last_prediction' in rollout:
                            last_prediction_curr = rollout['last_prediction']

                        p_roll = rollout['y_hat'][:, :win_len, :]
                        if p_roll.ndim == 2: p_roll = p_roll.unsqueeze(-1)
                        y_chunks.append(p_roll)

                        for key in ['mu', 'b', 'tau', 'pi']:
                            if key in rollout and isinstance(rollout[key], torch.Tensor):
                                dist_chunks[key].append(rollout[key][:, :win_len, ...])

                        c_hc_curr = rollout.get('c_n_hindcast', rollout.get('c_n'))
                        h_hc_curr = rollout.get('h_n_hindcast', rollout.get('h_n'))
                        c_fc_curr = rollout.get('c_n_forecast', rollout.get('c_n', c_hc_curr))
                        h_fc_curr = rollout.get('h_n_forecast', rollout.get('h_n', h_hc_curr))

                curr_idx = w_end

            # =========================================================================
            # PHASE 3: Post-DA Forecast Horizon (a_end -> total_len)
            # =========================================================================
            if curr_idx < total_len:
                with torch.no_grad():
                    fc_data = _slice_hydrology_batch(data, curr_idx, total_len)
                    fc_data['c_0_hindcast'] = c_hc_curr
                    fc_data['h_0_hindcast'] = h_hc_curr
                    fc_data['c_0_forecast'] = c_fc_curr
                    fc_data['h_0_forecast'] = h_fc_curr
                    fc_data['c_0'] = c_fc_curr
                    fc_data['h_0'] = h_fc_curr
                    fc_data['c_n'] = c_fc_curr
                    fc_data['h_n'] = h_fc_curr
                    if static_embedding_opt is not None:
                        fc_data['static_embedding'] = static_embedding_opt
                    if last_prediction_curr is not None:
                        fc_data['last_prediction'] = last_prediction_curr

                    fc_pred = ensure_y_hat(model(fc_data), use_median=True)
                    fc_y = fc_pred['y_hat'][:, :(total_len - curr_idx), :]
                    if fc_y.ndim == 2: fc_y = fc_y.unsqueeze(-1)
                    y_chunks.append(fc_y)

                    for key in ['mu', 'b', 'tau', 'pi']:
                        if key in fc_pred and isinstance(fc_pred[key], torch.Tensor):
                            dist_chunks[key].append(fc_pred[key][:, :(total_len - curr_idx), ...])

            out_y = torch.cat(y_chunks, dim=1)[:, :total_len, :]

            # Pre/Post Hindcast Validation Metrics
            with torch.no_grad():
                min_len = min(a_end - a_start, y_tensor.shape[1])
                if min_len > 0 and p_pre_chunks:
                    t_hc = y_tensor[:, a_start:a_start + min_len, :]
                    p_pre = torch.cat(p_pre_chunks, dim=1)[:, :min_len, :]
                    p_post = out_y[:, a_start:a_start + min_len, :]

                    valid_len = min(t_hc.shape[1], p_pre.shape[1], p_post.shape[1])
                    if valid_len > 0:
                        t_hc = t_hc[:, :valid_len, :]
                        p_pre = p_pre[:, :valid_len, :]
                        p_post = p_post[:, :valid_len, :]
                        mask_pre = ~torch.isnan(t_hc) & ~torch.isnan(p_pre)
                        mask_post = ~torch.isnan(t_hc) & ~torch.isnan(p_post)
                    else:
                        mask_pre = torch.zeros(1, dtype=torch.bool)
                        mask_post = torch.zeros(1, dtype=torch.bool)
                else:
                    valid_len = 0
                    p_pre, p_post, t_hc = None, None, None
                    mask_pre = torch.zeros(1, dtype=torch.bool)
                    mask_post = torch.zeros(1, dtype=torch.bool)

                def calc_metrics(sim, obs, mask):
                    if not mask.any() or valid_len == 0 or sim is None or obs is None:
                        return {'MSE': float('nan'), 'NSE': float('nan')}
                    s, o = sim[mask], obs[mask]
                    mse = torch.mean((s - o) ** 2).item()
                    var_o = torch.var(o, unbiased=False).item()
                    nse = 1.0 - (mse / (var_o + 1e-6)) if var_o > 1e-8 else float('nan')
                    return {'MSE': mse, 'NSE': nse}

                metrics_pre = calc_metrics(p_pre, t_hc, mask_pre)
                metrics_post = calc_metrics(p_post, t_hc, mask_post)

            res = {
                'y_hat': out_y.detach(),
                'c_n_hindcast': c_hc_curr,
                'h_n_hindcast': h_hc_curr,
                'c_n_forecast': c_fc_curr,
                'h_n_forecast': h_fc_curr,
                'c_n': c_fc_curr if c_fc_curr is not None else c_hc_curr,
                'h_n': h_fc_curr if h_fc_curr is not None else h_hc_curr,
                'hindcast_metrics_pre': metrics_pre,
                'hindcast_metrics_post': metrics_post,
            }
            for key, chunks in dist_chunks.items():
                if chunks:
                    res[key] = torch.cat(chunks, dim=1)[:, :total_len, ...]
            return res
