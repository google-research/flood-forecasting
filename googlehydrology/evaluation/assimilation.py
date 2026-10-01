# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Gradient-based (variational) data assimilation of latent model components.

For every batch, the components listed in the assimilation config (for
example the static and dynamic embeddings of ``MeanEmbeddingForecastLSTM``)
are treated as free variables over an assimilation window that ends at the
forecast issue time, i.e. ``lead_time`` steps before the end of the targets.
They are optimized so that the model's prediction matches the observations
inside the window, while a background term keeps them close to the model's
unassimilated (prior) values. The model weights are never changed.

Objective
---------
Every sequence of the batch is an independent problem. The observation term
is the configured loss with ``per_sequence=True`` (averaged within each
sequence over its valid observations, then over sequences) and the background
term (``BackgroundEmbeddingRegularization``) is likewise a mean over
sequences of a scale-normalised squared departure (the departure of every
sequence is divided by the mean square of its own prior, so
``regularization_weight`` is dimensionless). Sequences without a valid
observation in the window are excluded from both reductions and kept at the
prior, and the batch mean is rescaled to a sum over the remaining sequences,
so every sequence receives the gradient it would receive on its own.

Model interface
---------------
The engine only relies on the model

* validating component names in ``validate_assimilation_components(names)``,
* returning those components from ``forward(data, return_embeddings=True)``,
* accepting optimized values through the keyword arguments
  ``forward(data, assimilation_overrides=..., assimilation_slice=...)``,
  where time-varying components (``[B, T, ...]``) are passed for the window
  only and static components (``[B, ...]``) as a whole,
* reducing its outputs to a ``[B, T, n_targets]`` tensor in
  ``point_prediction(outputs)``.

Missing data
------------
Missing observations are handled by the masked losses. Non-finite values are
handled per sequence: a sequence whose gradient (or updated value) is not
finite is frozen at its last finite value for the rest of the run and
reported once; the run is aborted (returning the last finite values) only if
the total loss itself is not finite. The whole procedure runs in float32 with
autocast disabled, independently of the caller's autocast state.

Optionally, sequences whose prediction at the end of the window is already
close to the observation stop being optimized (``early_stopping_tolerance``).
"""

import contextlib
import dataclasses
import enum
import logging
from collections.abc import Iterable, Iterator

import torch
from torch import nn

from googlehydrology.modelzoo.basemodel import BaseModel
from googlehydrology.training import (
    get_loss_obj,
    get_optimizer,
    get_regularization_obj,
)
from googlehydrology.utils.assimilationconfig import AssimilationConfig

LOGGER = logging.getLogger(__name__)

# Lower bound of the epsilon added to per-sequence gradient norms before
# dividing; raised to the dtype's machine epsilon for low precision dtypes.
_CLIP_EPS = 1e-6
# Components with a time axis are [B, T, ...]; static ones are [B, ...].
_TIME_VARYING_NDIM = 3


class _Outcome(enum.Enum):
    """Result of one optimization step."""

    STEPPED = enum.auto()
    RETRY = enum.auto()  # Sequences with a non-finite loss were frozen.
    NONFINITE_LOSS = enum.auto()  # Total loss not finite; nothing changed.
    STOP = enum.auto()  # Every sequence converged or frozen.


@dataclasses.dataclass(frozen=True)
class _Problem:
    """Per-batch quantities that do not change during the optimization."""

    model: BaseModel
    data: dict[str, torch.Tensor]
    window: tuple[int, int]  # Window on the model's time axis.
    y: torch.Tensor  # Full observed targets, [B, T_y, n_targets].
    observed: torch.Tensor  # Observations inside the window.
    loss_data: dict[str, torch.Tensor]
    baselines: dict[str, torch.Tensor]  # Prior components inside the window.
    weights: dict[str, float]  # Per-component regularization weights.

    @property
    def n_window(self) -> int:
        return self.window[1] - self.window[0]


@dataclasses.dataclass
class _State:
    """Mutable state of the optimization of one batch."""

    optimized: dict[str, torch.Tensor]
    last_finite: dict[str, torch.Tensor]
    # [B] sequences that take part in the objective: those with a valid
    # observation whose loss and gradients are finite.
    active: torch.Tensor
    # [B] sequences frozen because of non-finite values; a subset of
    # `~active` (sequences without observations are never updated either).
    nonfinite: torch.Tensor

    @property
    def frozen(self) -> torch.Tensor:
        """[B] sequences that are no longer updated."""
        return ~self.active

    def freeze(self, rows: torch.Tensor) -> None:
        """Stop updating ``rows`` because of non-finite values."""
        self.nonfinite |= rows & self.active
        self.active &= ~rows


class Assimilation:
    """Optimizes latent model components over a window of observations.

    Parameters
    ----------
    cfg : AssimilationConfig
        The assimilation configuration.
    """

    def __init__(self, cfg: AssimilationConfig):
        """Build the per-sequence loss with its background term."""
        self.cfg = cfg
        self._loss_obj = get_loss_obj(cfg, per_sequence=True)
        self._loss_obj.set_regularization_terms(get_regularization_obj(cfg))
        # The CMAL negative log-likelihood needs the full head outputs; the
        # other losses fit the point prediction.
        self._fits_head_outputs = cfg.loss.upper() == 'CMAL'
        self._warned_nonfinite = False

    def assimilate(
        self, model: BaseModel, data: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Run data assimilation for one batch.

        Parameters
        ----------
        model : BaseModel
            The trained model. Its weights and training state are restored on
            exit.
        data : dict[str, torch.Tensor]
            The input batch, including the observations ``data['y']``.

        Returns
        -------
        dict[str, torch.Tensor]
            The model output dictionary computed with the assimilated
            components (float32), detached from the graph.

        Raises
        ------
        ValueError
            If a component is not supported by the model, if the assimilation
            window does not fit into the observed period, or if the data
            lacks what the configured loss needs.
        RuntimeError
            If every sequence had non-finite gradients at the first step.
        """
        components = self.cfg.assimilation_components
        model.validate_assimilation_components(components)
        device_type = data['y'].device.type
        if not torch.amp.is_autocast_available(device_type):
            device_type = 'cpu'
        # Autocast is disabled explicitly: half precision underflows the
        # optimizer epsilon and overflows the loss, see the module docstring.
        with (
            torch.autocast(device_type=device_type, enabled=False),
            _frozen(model),
        ):
            return self._assimilate(model, data, components)

    def _assimilate(
        self,
        model: BaseModel,
        data: dict[str, torch.Tensor],
        components: dict[str, dict[str, float | None]],
    ) -> dict[str, torch.Tensor]:
        y = data['y']
        with torch.no_grad():
            prior = model(data, return_embeddings=True)
        model_len = model.point_prediction(prior).shape[1]
        window, (start_y, end_y) = self._window(model_len, y.shape[1])
        _check_component_lengths(prior, components, model_len)
        observed = y[:, start_y:end_y]
        active = torch.isfinite(observed).flatten(1).any(dim=1)
        if self.cfg.epochs == 0 or not active.any():
            LOGGER.info(
                'Returning the prior unchanged: %s.',
                'epochs is 0'
                if self.cfg.epochs == 0
                else 'no finite observation in the assimilation window',
            )
            return _detach(prior)

        problem = _Problem(
            model=model,
            data=data,
            window=window,
            y=y,
            observed=observed,
            loss_data=self._loss_data(data, observed),
            # Window-sliced so that they match the optimized tensors, and
            # float32 regardless of the dtype the model produced them in.
            baselines={
                name: _window_of(prior[name].detach().float(), window)
                for name in components
            },
            weights={
                name: spec['regularization_weight']
                for name, spec in components.items()
            },
        )
        optimized = self._optimize(problem, active)
        with torch.no_grad():
            final = model(
                data,
                assimilation_overrides=optimized,
                assimilation_slice=window,
            )
        return _detach(final)

    def _window(
        self, model_len: int, y_len: int
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        """Return the window on the model axis and on the target axis.

        The window ends ``lead_time`` steps before the end of the targets,
        i.e. at the forecast issue time. Targets and model outputs are
        aligned at their last time step; the output may be shorter than the
        targets (e.g. a forecast model only predicts forecast_overlap +
        lead_time steps). The config validates the window against
        ``seq_length``; the actual tensor lengths are checked here.
        """
        n_window = self.cfg.assimilation_window
        lead_time = self.cfg.lead_time
        end_y = y_len - lead_time
        start_y = end_y - n_window
        if start_y < 0:
            raise ValueError(
                f'Assimilation window ({n_window} steps) plus lead time '
                f'({lead_time}) exceeds the target sequence length ({y_len}); '
                'reduce assimilation_window or increase seq_length.'
            )
        offset = model_len - y_len
        window = (start_y + offset, end_y + offset)
        if window[0] < 0:
            raise ValueError(
                f'The assimilation window ({n_window} steps, ending '
                f'{lead_time} steps before the end of the targets) must fit '
                f'within the model output of {model_len} steps '
                '(forecast_overlap + lead_time for forecast models).'
            )
        return window, (start_y, end_y)

    def _optimize(
        self, problem: _Problem, active: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Optimize the components; return the last finite values."""
        state = _State(
            optimized={
                name: base.clone().requires_grad_(requires_grad=True)
                for name, base in problem.baselines.items()
            },
            last_finite={
                name: base.clone() for name, base in problem.baselines.items()
            },
            active=active.clone(),
            nonfinite=torch.zeros_like(active),
        )
        optimizer = get_optimizer(self._param_groups(state.optimized), self.cfg)
        for epoch in range(self.cfg.epochs):
            outcome = self._step(problem, state, optimizer)
            while outcome is _Outcome.RETRY:
                outcome = self._step(problem, state, optimizer)
            if epoch == 0 and bool((state.nonfinite | ~active).all()):
                raise RuntimeError(
                    'Every sequence had a non-finite loss or gradient at the '
                    'first assimilation step.'
                )
            if outcome is _Outcome.STOP:
                break
            if outcome is _Outcome.NONFINITE_LOSS:
                self._warn_once(
                    'Non-finite assimilation loss at epoch %d that no single '
                    'sequence explains; returning the last finite components '
                    '(reported once).',
                    epoch,
                )
                break
            for name, tensor in state.optimized.items():
                state.last_finite[name].copy_(tensor.detach())
        n_nonfinite = int(state.nonfinite.sum())
        if n_nonfinite:
            self._warn_once(
                '%d of %d sequences were frozen at their last finite values '
                'because of a non-finite loss or gradients (reported once).',
                n_nonfinite,
                problem.y.shape[0],
            )
        return state.last_finite

    def _warn_once(self, msg: str, *args: object) -> None:
        if not self._warned_nonfinite:
            self._warned_nonfinite = True
            LOGGER.warning(msg, *args)

    def _param_groups(self, optimized: dict[str, torch.Tensor]) -> list[dict]:
        groups = []
        for name, tensor in optimized.items():
            group = {'params': [tensor]}
            lr = self.cfg.assimilation_components[name]['initial_learning_rate']
            if lr is not None:
                group['lr'] = lr
            groups.append(group)
        return groups

    def _step(
        self,
        problem: _Problem,
        state: _State,
        optimizer: torch.optim.Optimizer,
    ) -> _Outcome:
        """Run one optimization step on all sequences of the batch."""
        optimizer.zero_grad()
        output = problem.model(
            problem.data,
            assimilation_overrides=state.optimized,
            assimilation_slice=problem.window,
        )
        start, end = problem.window
        point = problem.model.point_prediction(output)[:, start:end]
        hold = state.frozen
        converged = self._converged_mask(point.detach(), problem)
        if converged is not None:
            hold = hold | converged
        if hold.all():
            return _Outcome.STOP
        prediction = self._prediction(problem, output, point)
        total_loss = self._loss(problem, state, prediction)
        if not torch.isfinite(total_loss):
            culprits = self._nonfinite_loss_rows(problem, state, prediction)
            if not culprits.any():
                return _Outcome.NONFINITE_LOSS
            state.freeze(culprits)
            return _Outcome.RETRY
        total_loss.backward()
        bad = _nonfinite_rows(t.grad for t in state.optimized.values())
        state.freeze(bad)
        _zero_grad_rows(state.optimized.values(), bad)
        if self.cfg.clip_gradient_norm is not None:
            _clip_per_sequence(
                state.optimized.values(), self.cfg.clip_gradient_norm
            )
        optimizer.step()
        bad = _nonfinite_rows(t.detach() for t in state.optimized.values())
        state.freeze(bad)
        # Undo the step for frozen and converged sequences. `last_finite`
        # holds the values before this step.
        _restore_rows(state.optimized, state.last_finite, hold | bad)
        return _Outcome.STEPPED

    def _converged_mask(
        self, point: torch.Tensor, problem: _Problem
    ) -> torch.Tensor | None:
        tolerance = self.cfg.early_stopping_tolerance
        if tolerance is None:
            return None
        return _converged(point, problem.observed, problem.y, tolerance)

    def _prediction(
        self,
        problem: _Problem,
        output: dict[str, torch.Tensor],
        point: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return the prediction dictionary of the loss, window-sliced."""
        if not self._fits_head_outputs:
            return {'y_hat': point}
        start, end = problem.window
        return {
            k: v[:, start:end]
            for k, v in output.items()
            if _is_time_varying(v) and v.shape[1] >= end
        }

    def _loss(
        self,
        problem: _Problem,
        state: _State,
        prediction: dict[str, torch.Tensor],
        rows: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sum over the active sequences of the per-sequence objectives.

        Inactive sequences are excluded from the observation term (their
        observations are masked) and from the background term (their
        components are masked), so both terms are means over the same
        ``n_active`` sequences; rescaling by ``n_active`` turns them into a
        sum, i.e. every sequence gets the gradient it would get on its own,
        whatever the batch composition. With ``rows``, only those rows of the
        batch are evaluated.
        """
        active = state.active if rows is None else state.active & rows
        inactive = ~active
        loss_data = dict(problem.loss_data)
        loss_data['y'] = _mask_rows(loss_data['y'], inactive)
        total_loss, _ = self._loss_obj(
            prediction,
            loss_data,
            # The window is the evaluation period during DA.
            predict_last_n=problem.n_window,
            other_model_data={
                'optimized_components': {
                    k: _mask_rows(v, inactive)
                    for k, v in state.optimized.items()
                },
                'baseline_components': {
                    k: _mask_rows(v, inactive)
                    for k, v in problem.baselines.items()
                },
                'component_weights': problem.weights,
            },
        )
        return total_loss * int(active.sum())

    def _nonfinite_loss_rows(
        self,
        problem: _Problem,
        state: _State,
        prediction: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """[B] mask of the active sequences whose own objective is not finite.

        Only evaluated when the total loss is not finite, one sequence at a
        time, so that the culprits can be frozen without aborting the others.
        """
        culprits = torch.zeros_like(state.active)
        with torch.no_grad():
            for i in torch.nonzero(state.active).flatten().tolist():
                rows = torch.zeros_like(state.active)
                rows[i] = True
                if not torch.isfinite(
                    self._loss(problem, state, prediction, rows)
                ):
                    culprits[i] = True
        return culprits

    def _loss_data(
        self,
        data: dict[str, torch.Tensor],
        observed: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Build the ground-truth dictionary for the loss object."""
        loss_data = {'y': observed}
        if self.cfg.loss.upper() == 'NSE':
            if 'per_basin_target_stds' not in data:
                raise ValueError(
                    "The NSE assimilation loss requires 'per_basin_target_stds'"
                    ' in the batch (the datasets provide it when the run '
                    'config or the assimilation config uses the NSE loss).'
                )
            loss_data['per_basin_target_stds'] = data['per_basin_target_stds']
        return loss_data


def _check_component_lengths(
    output: dict[str, torch.Tensor], components: Iterable[str], model_len: int
) -> None:
    """Time-varying components must share the time axis of the output.

    The model receives a single ``assimilation_slice`` for all components,
    so they must all be aligned with the model output.
    """
    for name in components:
        tensor = output[name]
        if _is_time_varying(tensor) and tensor.shape[1] != model_len:
            raise ValueError(
                f'{name} has {tensor.shape[1]} time steps but the model '
                f'output has {model_len}; time-varying components must '
                'be aligned with the model output.'
            )


@contextlib.contextmanager
def _frozen(model: nn.Module) -> Iterator[None]:
    """Freeze weights and disable dropout; restore everything on exit.

    Only the recurrent modules are put in training mode, because cuDNN only
    supports the backward pass of RNNs in training mode; the mode of every
    other module is left untouched. Dropout (standalone and inside RNNs) is
    disabled so that the optimization is deterministic. Modules whose
    training-mode behaviour cannot be neutralised this way (BatchNorm) are
    rejected.
    """
    batch_norms = [
        type(m).__name__
        for m in model.modules()
        if isinstance(m, nn.modules.batchnorm._BatchNorm)  # noqa: SLF001
    ]
    if batch_norms:
        raise ValueError(
            f'Data assimilation does not support models with {batch_norms} '
            'modules.'
        )
    requires_grad = {p: p.requires_grad for p in model.parameters()}
    dropout_p = {m: m.p for m in model.modules() if isinstance(m, nn.Dropout)}
    rnns = {
        m: (m.training, m.dropout)
        for m in model.modules()
        if isinstance(m, nn.RNNBase)
    }
    try:
        for p in requires_grad:
            p.requires_grad_(requires_grad=False)
        for m in dropout_p:
            m.p = 0.0
        for m in rnns:
            m.dropout = 0.0
            m.train(mode=True)
        yield
    finally:
        for p, flag in requires_grad.items():
            p.requires_grad_(flag)
        for m, p in dropout_p.items():
            m.p = p
        for m, (training, dropout) in rnns.items():
            m.dropout = dropout
            m.train(training)


def _converged(
    pred: torch.Tensor,
    obs: torch.Tensor,
    y_full: torch.Tensor,
    tolerance: float,
) -> torch.Tensor:
    """Return which sequences meet the early stopping criterion.

    The criterion is the relative error at the last step of the window,
    ``|pred - obs| / max(obs - y_zero, 0.05)``, where ``y_zero`` is the minimum
    of the full observed sequence (0 if it is all missing), so that the error
    is relative to the flow above the sequence's base level. If the last step
    has no valid observation or prediction, the maximum relative error over
    the valid steps of the window is used instead; a sequence without any
    valid step never converges. A sequence converged if the error of every
    target is <= ``tolerance``.

    Parameters
    ----------
    pred : torch.Tensor
        Point prediction inside the window, [B, W, n_targets].
    obs : torch.Tensor
        Observations inside the window, [B, W, n_targets], NaN if missing.
    y_full : torch.Tensor
        The full observed target sequences, [B, T, n_targets].
    tolerance : float
        Relative error at or below which a sequence converged.

    Returns
    -------
    torch.Tensor
        Boolean tensor of shape [B].
    """
    y_zero = y_full.nan_to_num(nan=float('inf')).amin(dim=1, keepdim=True)
    y_zero = torch.where(torch.isinf(y_zero), torch.zeros_like(y_zero), y_zero)
    valid = ~torch.isnan(obs) & ~torch.isnan(pred)
    rel_err = (pred - obs).abs() / (obs - y_zero).clamp(min=0.05)
    window_err = torch.where(valid, rel_err, torch.zeros_like(rel_err))
    err = torch.where(valid[:, -1], rel_err[:, -1], window_err.amax(dim=1))
    err = torch.where(valid.any(dim=1), err, torch.full_like(err, torch.inf))
    return (err <= tolerance).all(dim=-1)


def _is_time_varying(tensor: torch.Tensor) -> bool:
    """Whether a component is [B, T, ...] rather than [B, ...]."""
    return tensor.ndim >= _TIME_VARYING_NDIM


def _window_of(tensor: torch.Tensor, window: tuple[int, int]) -> torch.Tensor:
    """Return the window of a time-varying [B, T, ...] component."""
    if _is_time_varying(tensor):
        return tensor[:, window[0] : window[1]]
    return tensor


def _mask_rows(tensor: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    """Return ``tensor`` with the batch rows in ``rows`` set to NaN."""
    if not rows.any():
        return tensor
    shape = (-1, *([1] * (tensor.ndim - 1)))
    return tensor.masked_fill(rows.view(shape), float('nan'))


def _nonfinite_rows(tensors: Iterable[torch.Tensor | None]) -> torch.Tensor:
    """[B] mask of the batch rows with a non-finite entry in any tensor.

    ``None`` entries (e.g. gradients that were never populated) are skipped.
    """
    tensors = [t for t in tensors if t is not None]
    bad = torch.zeros(
        tensors[0].shape[0], dtype=torch.bool, device=tensors[0].device
    )
    for t in tensors:
        bad |= ~torch.isfinite(t).flatten(1).all(dim=1)
    return bad


def _zero_grad_rows(
    tensors: Iterable[torch.Tensor], rows: torch.Tensor
) -> None:
    """In place, zero the gradients of the batch rows in ``rows``."""
    if not rows.any():
        return
    for t in tensors:
        if t.grad is not None:
            t.grad[rows] = 0.0


def _restore_rows(
    tensors: dict[str, torch.Tensor],
    values: dict[str, torch.Tensor],
    rows: torch.Tensor,
) -> None:
    """In place, set ``tensors[name][rows] = values[name][rows]``."""
    if not rows.any():
        return
    with torch.no_grad():
        for name, tensor in tensors.items():
            tensor[rows] = values[name][rows]


def _clip_per_sequence(
    tensors: Iterable[torch.Tensor], max_norm: float
) -> None:
    """Clip the gradient norm of every sequence (batch row) independently.

    Each sequence is an independent optimization problem, so a global norm
    would couple them.
    """
    tensors = [t for t in tensors if t.grad is not None]
    if not tensors:
        return
    sq_norm = sum(t.grad.flatten(1).pow(2).sum(dim=1) for t in tensors)
    eps = max(_CLIP_EPS, float(torch.finfo(sq_norm.dtype).eps))
    scale = (max_norm / (sq_norm.sqrt() + eps)).clamp(max=1.0)
    for t in tensors:
        t.grad.mul_(scale.view(-1, *([1] * (t.grad.ndim - 1))))


def _detach(output: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        k: v.detach() if isinstance(v, torch.Tensor) else v
        for k, v in output.items()
    }
