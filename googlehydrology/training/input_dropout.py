# Copyright 2026 Google LLC
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

"""Model-agnostic input feature-group dropout for training batches."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from googlehydrology.utils.config import validate_dropout_probability
from googlehydrology.utils.configutils import group_features_list

if TYPE_CHECKING:
    from googlehydrology.utils.config import Config

LOGGER = logging.getLogger(__name__)
_EXPECTED_FEATURE_NDIM = 3
_MIN_DROPPABLE_GROUPS = 2


@dataclass(frozen=True)
class _WindowDims:
    """Temporal dimensions shared across hindcast and forecast passes."""

    batch_size: int
    hind_len: int
    fcst_len: int
    overlap_len: int
    horizon_len: int
    overlap_start: int
    device: torch.device


class InputDropout:
    """Randomly masks input feature groups with NaNs during training.

    Inspired by the training-time input dropout used in Google's operational
    FloodHub models:

    - **Whole-sequence dropout** (``nan_sequence_probability``): sets an entire
      hindcast or forecast sequence of a feature group to ``NaN``.
    - **Contiguous streak dropout** (``nan_streak_probability``): draws
      ``start ~ Uniform{0, ..., L - 1}`` and
      ``duration ~ Uniform{1, ..., L}``, clamps ``end = min(start + duration,
      L)``, and sets the half-open slice ``[start, end)`` to ``NaN``. Because
      streaks clamp at ``L``, later timesteps in the window have higher
      marginal dropout probability (mirroring outages near forecast issue time
      or at longer lead times).

    When both probabilities are non-zero, whole-sequence and streak drops are
    sampled independently per window and combined via logical OR. Per-group
    mappings use whitelist semantics (omitted groups default to ``0.0``), and
    shared groups use the same configured probability for hindcast and forecast
    windows.

    For forecasting runs, the hindcast window and forecast horizon are sampled
    independently for each feature group, while keeping the overlap prefix of
    ``x_d_forecast`` synchronized with the trailing slice of ``x_d_hindcast``
    for shared groups.

    Groups are processed in an unbiased random permutation per sample while
    enforcing that every timestep with at least one valid (non-NaN) input
    group before dropout retains at least one valid group afterward in every
    active input pool (hindcast pool over the hindcast window, forecast pool
    over the overlap and horizon, and the shared-group subset over the horizon
    when forecast-only groups also exist). Because survivor protection acts
    per timestep, it may split a streak or reduce effective drop rates in small
    pools. Feature sets across distinct groups within ``hindcast_groups`` (or
    within ``forecast_groups``) must be pairwise disjoint when dropout is
    enabled.

    Parameters
    ----------
    hindcast_groups : dict[str, list[str]]
        Mapping from hindcast group name to its feature names.
    forecast_groups : dict[str, list[str]]
        Mapping from forecast group name to its feature names (empty for
        hindcast-only runs).
    sequence_probabilities : float | dict[str, float], optional
        Whole-sequence drop probability, either uniform or per group name.
    streak_probabilities : float | dict[str, float], optional
        Contiguous-streak drop probability, either uniform or per group name.
    forecast_overlap : int | None, optional
        Number of hindcast overlap timesteps prepended to ``x_d_forecast``.
    """

    def __init__(
        self,
        *,
        hindcast_groups: dict[str, list[str]],
        forecast_groups: dict[str, list[str]],
        sequence_probabilities: float | dict[str, float] = 0.0,
        streak_probabilities: float | dict[str, float] = 0.0,
        forecast_overlap: int | None = None,
    ) -> None:
        """Initialize feature-group drop probabilities and window settings."""
        self._hindcast_groups = {
            name: list(feats) for name, feats in hindcast_groups.items()
        }
        self._forecast_groups = {
            name: list(feats) for name, feats in forecast_groups.items()
        }
        self._forecast_overlap = int(forecast_overlap or 0)
        if self._forecast_overlap < 0:
            raise ValueError(
                f'forecast_overlap must be >= 0, got {self._forecast_overlap}.'
            )

        all_groups = list(
            dict.fromkeys([*self._hindcast_groups, *self._forecast_groups])
        )
        self._all_groups = all_groups
        self._seq_probs = _resolve_group_probabilities(
            sequence_probabilities,
            all_groups,
            name='nan_sequence_probability',
        )
        self._streak_probs = _resolve_group_probabilities(
            streak_probabilities,
            all_groups,
            name='nan_streak_probability',
        )
        self._enabled = any(self._is_group_active(g) for g in all_groups)
        self._has_droppable_pool = False
        self._device_cache: dict[tuple[Any, ...], Any] = {}
        self._warned_missing_groups: set[tuple[str, str]] = set()
        self._warned_unrecognized_batch = False

        if self._enabled:
            _validate_disjoint_groups(
                self._hindcast_groups, pool_name='hindcast_inputs'
            )
            _validate_disjoint_groups(
                self._forecast_groups, pool_name='forecast_inputs'
            )
            self._has_droppable_pool = self._check_droppable_pools()

    @classmethod
    def from_config(cls, cfg: Config) -> InputDropout:
        """Build an ``InputDropout`` instance from a run ``Config``."""
        # Read raw dict keys directly so missing optional keys do not trigger
        # Config._get_value_verbose errors on hindcast-only configs.
        raw_cfg = cfg.as_dict()
        raw_hindcast = raw_cfg.get('hindcast_inputs')
        raw_forecast = raw_cfg.get('forecast_inputs')
        hindcast_groups = (
            group_features_list(raw_hindcast) if raw_hindcast else {}
        )
        forecast_groups = (
            group_features_list(raw_forecast) if raw_forecast else {}
        )
        return cls(
            hindcast_groups=hindcast_groups,
            forecast_groups=forecast_groups,
            sequence_probabilities=cfg.nan_sequence_probability,
            streak_probabilities=cfg.nan_streak_probability,
            forecast_overlap=cfg.forecast_overlap,
        )

    @property
    def enabled(self) -> bool:
        """Whether any feature group has a non-zero drop probability."""
        return self._enabled

    @property
    def sequence_probabilities(self) -> dict[str, float]:
        """Resolved per-group whole-sequence drop probabilities."""
        return dict(self._seq_probs)

    @property
    def streak_probabilities(self) -> dict[str, float]:
        """Resolved per-group contiguous-streak drop probabilities."""
        return dict(self._streak_probs)

    def _is_group_active(self, name: str) -> bool:
        return self._seq_probs[name] > 0.0 or self._streak_probs[name] > 0.0

    def _check_droppable_pools(self) -> bool:
        hind_active = any(
            self._is_group_active(g) for g in self._hindcast_groups
        )
        fcst_active = any(
            self._is_group_active(g) for g in self._forecast_groups
        )
        hind_droppable = (
            len(self._hindcast_groups) >= _MIN_DROPPABLE_GROUPS and hind_active
        )
        fcst_droppable = (
            len(self._forecast_groups) >= _MIN_DROPPABLE_GROUPS and fcst_active
        )
        if not hind_droppable and not fcst_droppable:
            LOGGER.warning(
                'Input dropout is enabled, but every configured input pool '
                'has fewer than 2 groups (or no active drop probability); '
                'the survivor invariant prevents dropping any inputs.'
            )
            return False
        if len(self._hindcast_groups) == 1 and hind_active:
            LOGGER.warning(
                'hindcast_inputs has only 1 feature group; the survivor '
                'invariant prevents dropping it in the hindcast window.'
            )
        if len(self._forecast_groups) == 1 and fcst_active:
            LOGGER.warning(
                'forecast_inputs has only 1 feature group; the survivor '
                'invariant prevents dropping it in the forecast window.'
            )
        return True

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        """Return a batch dict with synthetic NaN dropout applied.

        Never mutates the caller's batch dictionary or its input tensors in
        place.
        """
        if not self._enabled or not self._has_droppable_pool:
            return data

        if 'x_d_hindcast' in data and 'x_d_forecast' in data:
            return self._apply_forecast_batch(data)
        if 'x_d_hindcast' in data:
            return self._apply_hindcast_only_batch(data, key='x_d_hindcast')
        if 'x_d' in data:
            return self._apply_hindcast_only_batch(data, key='x_d')

        if not self._warned_unrecognized_batch:
            self._warned_unrecognized_batch = True
            LOGGER.warning(
                'InputDropout is enabled but batch has none of '
                "'x_d_hindcast', 'x_d_forecast', or 'x_d'; skipping dropout."
            )
        return data

    def _present_groups_with_warning(
        self,
        groups: dict[str, list[str]],
        features_dict: dict[str, torch.Tensor],
        pool_label: str,
    ) -> dict[str, list[str]]:
        present = _present_groups(groups, features_dict)
        if len(present) < len(groups):
            for name in groups:
                if name not in present and (
                    (pool_label, name) not in self._warned_missing_groups
                ):
                    self._warned_missing_groups.add((pool_label, name))
                    LOGGER.warning(
                        'Configured %s feature group %r is missing one or '
                        'more features from the batch and will be skipped '
                        'by InputDropout.',
                        pool_label,
                        name,
                    )
        return present

    def _get_prob_tensors(
        self, names: list[str], device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, bool, bool]:
        key = ('probs', device, tuple(names))
        cached = self._device_cache.get(key)
        if cached is not None:
            return cached
        seq_vals = [self._seq_probs[n] for n in names]
        streak_vals = [self._streak_probs[n] for n in names]
        seq_t = torch.tensor(seq_vals, dtype=torch.float32, device=device).view(
            -1, 1, 1
        )
        streak_t = torch.tensor(
            streak_vals, dtype=torch.float32, device=device
        ).view(-1, 1, 1)
        result = (
            seq_t,
            streak_t,
            any(p > 0.0 for p in seq_vals),
            any(p > 0.0 for p in streak_vals),
        )
        self._device_cache[key] = result
        return result

    def _get_pool_tensor(
        self, flags: tuple[bool, ...], device: torch.device
    ) -> torch.Tensor:
        key = ('pool', device, flags)
        cached = self._device_cache.get(key)
        if cached is not None:
            return cached
        tensor = torch.tensor(flags, dtype=torch.bool, device=device)
        self._device_cache[key] = tensor
        return tensor

    def _sample_window_drop_masks(
        self,
        *,
        names: list[str],
        batch_size: int,
        window_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Sample candidate drop masks ``[G, B, L]`` vectorized over groups."""
        num_groups = len(names)
        mask = torch.zeros(
            (num_groups, batch_size, window_len),
            dtype=torch.bool,
            device=device,
        )
        if window_len <= 0 or num_groups == 0:
            return mask

        seq_probs, streak_probs, has_seq, has_streak = self._get_prob_tensors(
            names, device
        )
        if has_streak:
            use_streak = (
                torch.rand((num_groups, batch_size, 1), device=device)
                < streak_probs
            )
            start = torch.randint(
                0, window_len, (num_groups, batch_size, 1), device=device
            )
            duration = torch.randint(
                1, window_len + 1, (num_groups, batch_size, 1), device=device
            )
            end = torch.clamp(start + duration, max=window_len)
            steps = torch.arange(window_len, device=device).view(
                1, 1, window_len
            )
            mask = mask | (use_streak & (steps >= start) & (steps < end))

        if has_seq:
            use_seq = (
                torch.rand((num_groups, batch_size, 1), device=device)
                < seq_probs
            )
            mask = mask | use_seq.expand(num_groups, batch_size, window_len)

        return mask

    def _apply_hindcast_only_batch(
        self, data: dict[str, Any], *, key: str
    ) -> dict[str, Any]:
        x_d: dict[str, torch.Tensor] = data[key]
        groups = self._present_groups_with_warning(
            self._hindcast_groups, x_d, key
        )
        names = list(groups)
        if len(names) <= 1 or not any(self._is_group_active(n) for n in names):
            return data

        first_feat = groups[names[0]][0]
        sample_tensor = x_d[first_feat]
        _validate_feature_tensor_ndim(sample_tensor, first_feat)
        batch_size, seq_len = sample_tensor.shape[:2]
        if batch_size == 0 or seq_len == 0:
            return data
        device = sample_tensor.device

        valid = torch.stack(
            [_group_valid_mask(x_d, groups[name]) for name in names], dim=0
        )
        candidate = self._sample_window_drop_masks(
            names=names,
            batch_size=batch_size,
            window_len=seq_len,
            device=device,
        )
        in_pool = self._get_pool_tensor((True,) * len(names), device)
        applied = _enforce_survivor_invariants(
            valid=valid,
            candidate=candidate,
            pool_memberships=[in_pool],
            pool_active_slices=[(0, seq_len)],
        )
        masks = {
            name: applied[i]
            for i, name in enumerate(names)
            if self._is_group_active(name)
        }

        new_data = dict(data)
        new_data[key] = _mask_feature_dict(x_d, groups, masks)
        return new_data

    def _apply_forecast_batch(self, data: dict[str, Any]) -> dict[str, Any]:
        x_h: dict[str, torch.Tensor] = data['x_d_hindcast']
        x_f: dict[str, torch.Tensor] = data['x_d_forecast']

        hind_groups = self._present_groups_with_warning(
            self._hindcast_groups, x_h, 'x_d_hindcast'
        )
        fcst_groups = self._present_groups_with_warning(
            self._forecast_groups, x_f, 'x_d_forecast'
        )
        dims = _resolve_window_dims(
            x_h, x_f, hind_groups, fcst_groups, self._forecast_overlap
        )
        if dims is None:
            return data

        hind_masks, fcst_overlap_masks = self._sample_hindcast_window(
            x_h, x_f, hind_groups, fcst_groups, dims
        )
        fcst_horizon_masks = self._sample_forecast_horizon(
            x_f, hind_groups, fcst_groups, dims
        )
        fcst_masks = _combine_forecast_masks(
            fcst_groups,
            fcst_overlap_masks,
            fcst_horizon_masks,
            dims,
            active_names={n for n in fcst_groups if self._is_group_active(n)},
        )

        new_data = dict(data)
        new_data['x_d_hindcast'] = _mask_feature_dict(
            x_h, hind_groups, hind_masks
        )
        new_data['x_d_forecast'] = _mask_feature_dict(
            x_f, fcst_groups, fcst_masks
        )
        return new_data

    def _sample_hindcast_window(
        self,
        x_h: dict[str, torch.Tensor],
        x_f: dict[str, torch.Tensor],
        hind_groups: dict[str, list[str]],
        fcst_groups: dict[str, list[str]],
        dims: _WindowDims,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Sample hindcast masks synced with ``x_d_forecast[:, :overlap]``."""
        if dims.hind_len <= 0:
            return {}, {}

        pass1_names = list(
            dict.fromkeys(
                [
                    *hind_groups,
                    *(fcst_groups if dims.overlap_len > 0 else ()),
                ]
            )
        )
        if len(pass1_names) <= 1 or not any(
            self._is_group_active(n) for n in pass1_names
        ):
            return {}, {}

        in_hind_flags = tuple(name in hind_groups for name in pass1_names)
        in_fcst_flags = tuple(
            dims.overlap_len > 0 and name in fcst_groups for name in pass1_names
        )
        valid_list = [
            _pass1_group_valid_mask(
                x_h=x_h,
                x_f=x_f,
                hind_feats=hind_groups.get(name),
                fcst_feats=(
                    fcst_groups.get(name) if in_fcst_flags[idx] else None
                ),
                dims=dims,
            )
            for idx, name in enumerate(pass1_names)
        ]
        candidate = self._sample_pass1_candidates(
            pass1_names=pass1_names,
            in_hind_flags=in_hind_flags,
            dims=dims,
        )

        pool_memberships = [self._get_pool_tensor(in_hind_flags, dims.device)]
        pool_active_slices = [(0, dims.hind_len)]
        if dims.overlap_len > 0 and any(in_fcst_flags):
            pool_memberships.append(
                self._get_pool_tensor(in_fcst_flags, dims.device)
            )
            pool_active_slices.append((dims.overlap_start, dims.hind_len))

        applied = _enforce_survivor_invariants(
            valid=torch.stack(valid_list, dim=0),
            candidate=candidate,
            pool_memberships=pool_memberships,
            pool_active_slices=pool_active_slices,
        )

        hind_masks: dict[str, torch.Tensor] = {}
        fcst_overlap_masks: dict[str, torch.Tensor] = {}
        for idx, name in enumerate(pass1_names):
            if not self._is_group_active(name):
                continue
            mask = applied[idx]
            if in_hind_flags[idx]:
                hind_masks[name] = mask
            if in_fcst_flags[idx]:
                fcst_overlap_masks[name] = mask[:, dims.overlap_start :]
        return hind_masks, fcst_overlap_masks

    def _sample_pass1_candidates(
        self,
        *,
        pass1_names: list[str],
        in_hind_flags: tuple[bool, ...],
        dims: _WindowDims,
    ) -> torch.Tensor:
        if all(in_hind_flags):
            return self._sample_window_drop_masks(
                names=pass1_names,
                batch_size=dims.batch_size,
                window_len=dims.hind_len,
                device=dims.device,
            )

        # In pass1_names, hindcast groups precede forecast-only groups.
        hind_count = sum(in_hind_flags)
        hind_cand = self._sample_window_drop_masks(
            names=pass1_names[:hind_count],
            batch_size=dims.batch_size,
            window_len=dims.hind_len,
            device=dims.device,
        )
        fcst_only_names = pass1_names[hind_count:]
        fcst_only_cand = torch.zeros(
            (len(fcst_only_names), dims.batch_size, dims.hind_len),
            dtype=torch.bool,
            device=dims.device,
        )
        fcst_only_cand[:, :, dims.overlap_start :] = (
            self._sample_window_drop_masks(
                names=fcst_only_names,
                batch_size=dims.batch_size,
                window_len=dims.overlap_len,
                device=dims.device,
            )
        )
        return torch.cat([hind_cand, fcst_only_cand], dim=0)

    def _sample_forecast_horizon(
        self,
        x_f: dict[str, torch.Tensor],
        hind_groups: dict[str, list[str]],
        fcst_groups: dict[str, list[str]],
        dims: _WindowDims,
    ) -> dict[str, torch.Tensor]:
        """Sample masks on the forecast horizon ``[overlap_len : fcst_len]``."""
        if dims.horizon_len <= 0 or len(fcst_groups) <= 1:
            return {}

        fcst_names = list(fcst_groups)
        if not any(self._is_group_active(n) for n in fcst_names):
            return {}

        valid_list = [
            _group_valid_mask(x_f, fcst_groups[name], start=dims.overlap_len)
            for name in fcst_names
        ]
        candidate = self._sample_window_drop_masks(
            names=fcst_names,
            batch_size=dims.batch_size,
            window_len=dims.horizon_len,
            device=dims.device,
        )

        pool_memberships = [
            self._get_pool_tensor((True,) * len(fcst_names), dims.device)
        ]
        pool_active_slices = [(0, dims.horizon_len)]

        shared_flags = tuple(name in hind_groups for name in fcst_names)
        if any(shared_flags) and not all(shared_flags):
            pool_memberships.append(
                self._get_pool_tensor(shared_flags, dims.device)
            )
            pool_active_slices.append((0, dims.horizon_len))

        applied = _enforce_survivor_invariants(
            valid=torch.stack(valid_list, dim=0),
            candidate=candidate,
            pool_memberships=pool_memberships,
            pool_active_slices=pool_active_slices,
        )
        return {
            name: applied[i]
            for i, name in enumerate(fcst_names)
            if self._is_group_active(name)
        }


def _validate_disjoint_groups(
    groups: dict[str, list[str]], *, pool_name: str
) -> None:
    """Ensure no feature name appears in more than one group in a pool."""
    feature_to_group: dict[str, str] = {}
    for group_name, feats in groups.items():
        for feat in feats:
            prev_group = feature_to_group.get(feat)
            if prev_group is not None and prev_group != group_name:
                raise ValueError(
                    f'Feature {feat!r} appears in multiple groups '
                    f'({prev_group!r} and {group_name!r}) in {pool_name}. '
                    'InputDropout requires pairwise disjoint feature sets '
                    'across groups within the same input pool.'
                )
            feature_to_group[feat] = group_name


def _validate_feature_tensor_ndim(tensor: torch.Tensor, feat: str) -> None:
    if tensor.ndim != _EXPECTED_FEATURE_NDIM:
        raise ValueError(
            f'InputDropout expects 3D [batch, seq_len, features] tensors, '
            f'got shape {tuple(tensor.shape)} for feature {feat!r}.'
        )


def _present_groups(
    groups: dict[str, list[str]], features_dict: dict[str, torch.Tensor]
) -> dict[str, list[str]]:
    return {
        name: feats
        for name, feats in groups.items()
        if feats and all(f in features_dict for f in feats)
    }


def _resolve_window_dims(
    x_h: dict[str, torch.Tensor],
    x_f: dict[str, torch.Tensor],
    hind_groups: dict[str, list[str]],
    fcst_groups: dict[str, list[str]],
    forecast_overlap: int,
) -> _WindowDims | None:
    if not hind_groups and not fcst_groups:
        return None
    first_h_feat = next(iter(hind_groups.values()))[0] if hind_groups else None
    first_f_feat = next(iter(fcst_groups.values()))[0] if fcst_groups else None
    first_h = x_h[first_h_feat] if first_h_feat is not None else None
    first_f = x_f[first_f_feat] if first_f_feat is not None else None
    if first_h is not None and first_h_feat is not None:
        _validate_feature_tensor_ndim(first_h, first_h_feat)
    if first_f is not None and first_f_feat is not None:
        _validate_feature_tensor_ndim(first_f, first_f_feat)

    ref = first_h if first_h is not None else first_f
    assert ref is not None
    batch_size = ref.shape[0]
    if batch_size == 0:
        return None
    hind_len = first_h.shape[1] if first_h is not None else 0
    fcst_len = first_f.shape[1] if first_f is not None else 0

    if forecast_overlap > 0:
        if first_h is not None and forecast_overlap > hind_len:
            raise ValueError(
                f'forecast_overlap ({forecast_overlap}) exceeds hindcast '
                f'sequence length ({hind_len}).'
            )
        if first_f is not None and forecast_overlap > fcst_len:
            raise ValueError(
                f'forecast_overlap ({forecast_overlap}) exceeds forecast '
                f'sequence length ({fcst_len}).'
            )

    overlap_len = (
        forecast_overlap if (first_h is not None and first_f is not None) else 0
    )
    return _WindowDims(
        batch_size=batch_size,
        hind_len=hind_len,
        fcst_len=fcst_len,
        overlap_len=overlap_len,
        horizon_len=fcst_len - overlap_len,
        overlap_start=hind_len - overlap_len,
        device=ref.device,
    )


def _combine_forecast_masks(
    fcst_groups: dict[str, list[str]],
    overlap_masks: dict[str, torch.Tensor],
    horizon_masks: dict[str, torch.Tensor],
    dims: _WindowDims,
    *,
    active_names: set[str],
) -> dict[str, torch.Tensor]:
    combined: dict[str, torch.Tensor] = {}
    for name in fcst_groups:
        if name not in active_names:
            continue
        if name not in overlap_masks and name not in horizon_masks:
            continue
        parts: list[torch.Tensor] = []
        if dims.overlap_len > 0:
            overlap_mask = overlap_masks.get(name)
            if overlap_mask is None:
                overlap_mask = torch.zeros(
                    (dims.batch_size, dims.overlap_len),
                    dtype=torch.bool,
                    device=dims.device,
                )
            parts.append(overlap_mask)
        if dims.horizon_len > 0:
            horizon_mask = horizon_masks.get(name)
            if horizon_mask is None:
                horizon_mask = torch.zeros(
                    (dims.batch_size, dims.horizon_len),
                    dtype=torch.bool,
                    device=dims.device,
                )
            parts.append(horizon_mask)
        if parts:
            combined[name] = (
                parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)
            )
    return combined


def _mask_feature_dict(
    features_dict: dict[str, torch.Tensor],
    groups: dict[str, list[str]],
    masks: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    new_dict = dict(features_dict)
    for name, feats in groups.items():
        mask = masks.get(name)
        if mask is not None:
            mask_3d = mask.unsqueeze(-1)
            for feat in feats:
                tensor = new_dict[feat]
                _validate_feature_tensor_ndim(tensor, feat)
                new_dict[feat] = tensor.masked_fill(mask_3d, float('nan'))
    return new_dict


def _resolve_group_probabilities(
    probs: float | dict[str, float],
    all_groups: list[str],
    *,
    name: str,
) -> dict[str, float]:
    if isinstance(probs, dict):
        validated: dict[str, float] = {}
        for group, prob in probs.items():
            if not isinstance(group, str) or not group:
                raise ValueError(
                    f'{name} keys must be non-empty feature group names, '
                    f'got {group!r}.'
                )
            validated[group] = validate_dropout_probability(
                prob, f'{name}[{group!r}]'
            )
        unknown = sorted(set(validated) - set(all_groups))
        if unknown:
            raise ValueError(
                f'Unknown feature group(s) in {name}: {unknown}. '
                f'Configured groups: {sorted(all_groups)}.'
            )
        return {group: validated.get(group, 0.0) for group in all_groups}

    scalar = validate_dropout_probability(probs, name)
    if scalar > 0.0 and not all_groups:
        raise ValueError(
            f'{name} is non-zero ({scalar}), but no input feature groups '
            'are configured.'
        )
    return dict.fromkeys(all_groups, scalar)


def _group_valid_mask(
    features_dict: dict[str, torch.Tensor],
    feature_names: list[str],
    *,
    start: int = 0,
    end: int | None = None,
) -> torch.Tensor:
    """Return bool tensor [B, L] where all features of a group are non-NaN."""
    first = features_dict[feature_names[0]]
    _validate_feature_tensor_ndim(first, feature_names[0])
    has_nan = torch.isnan(first[:, start:end]).any(dim=-1)
    for feat in feature_names[1:]:
        tensor = features_dict[feat]
        _validate_feature_tensor_ndim(tensor, feat)
        has_nan = has_nan | torch.isnan(tensor[:, start:end]).any(dim=-1)
    return ~has_nan


def _pass1_group_valid_mask(
    *,
    x_h: dict[str, torch.Tensor],
    x_f: dict[str, torch.Tensor],
    hind_feats: list[str] | None,
    fcst_feats: list[str] | None,
    dims: _WindowDims,
) -> torch.Tensor:
    if hind_feats is not None:
        valid = _group_valid_mask(x_h, hind_feats)
        if fcst_feats is not None and dims.overlap_len > 0:
            fcst_valid = _group_valid_mask(
                x_f, fcst_feats, end=dims.overlap_len
            )
            valid = valid.clone()
            valid[:, dims.overlap_start :] &= fcst_valid
        return valid

    valid = torch.zeros(
        (dims.batch_size, dims.hind_len), dtype=torch.bool, device=dims.device
    )
    if fcst_feats is not None and dims.overlap_len > 0:
        valid[:, dims.overlap_start :] = _group_valid_mask(
            x_f, fcst_feats, end=dims.overlap_len
        )
    return valid


def _enforce_survivor_invariants(
    *,
    valid: torch.Tensor,
    candidate: torch.Tensor,
    pool_memberships: list[torch.Tensor],
    pool_active_slices: list[tuple[int, int]],
) -> torch.Tensor:
    """Accept candidate drops in random group order without emptying any pool.

    Parameters
    ----------
    valid : torch.Tensor
        Boolean tensor ``[G, B, L]`` indicating pre-existing non-NaN validity.
    candidate : torch.Tensor
        Boolean tensor ``[G, B, L]`` of proposed drops.
    pool_memberships : list[torch.Tensor]
        For each pool, a boolean tensor ``[G]`` of groups belonging to it.
    pool_active_slices : list[tuple[int, int]]
        For each pool, the ``(start, end)`` time slice where the pool must
        keep at least one valid group whenever it had one before dropout.

    Returns
    -------
    torch.Tensor
        Boolean tensor ``[G, B, L]`` of accepted drops.
    """
    num_groups, batch_size, seq_len = valid.shape
    device = valid.device

    if num_groups <= 1:
        return torch.zeros_like(candidate)

    # Independent uniform random permutation of groups for every batch element.
    rand_scores = torch.rand((num_groups, batch_size), device=device)
    order = torch.argsort(rand_scores, dim=0)
    idx = order.unsqueeze(-1).expand(num_groups, batch_size, seq_len)

    valid_perm = torch.gather(valid, 0, idx)
    cand_perm = torch.gather(candidate, 0, idx)
    applied_perm = torch.zeros_like(cand_perm)

    # Precompute per-pool remaining valid counts [B, L] (int32) and permuted
    # group membership masks [G, B, L]. Outside a pool's active time slice, the
    # membership flag is False so that pool imposes no constraint there.
    pool_remaining: list[torch.Tensor] = []
    pool_member_perm: list[torch.Tensor] = []
    for membership, (slice_start, slice_end) in zip(
        pool_memberships, pool_active_slices, strict=True
    ):
        member_3d = membership.view(num_groups, 1, 1)
        remaining = (valid & member_3d).sum(dim=0, dtype=torch.int32)
        pool_remaining.append(remaining)

        member_perm = membership[order].unsqueeze(-1)
        if slice_start > 0 or slice_end < seq_len:
            active_time = torch.zeros(
                (1, 1, seq_len), dtype=torch.bool, device=device
            )
            active_time[:, :, slice_start:slice_end] = True
            member_perm = member_perm & active_time
        pool_member_perm.append(member_perm)

    for rank in range(num_groups):
        can_drop = cand_perm[rank] & valid_perm[rank]
        for remaining, member_perm in zip(
            pool_remaining, pool_member_perm, strict=True
        ):
            in_pool = member_perm[rank]
            can_drop = can_drop & (~in_pool | (remaining > 1))

        applied_perm[rank] = can_drop
        for remaining, member_perm in zip(
            pool_remaining, pool_member_perm, strict=True
        ):
            decrement = can_drop & member_perm[rank]
            remaining.sub_(decrement.to(torch.int32))

    applied = torch.zeros_like(candidate)
    applied.scatter_(0, idx, applied_perm)
    return applied
