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

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from googlehydrology.utils.configutils import group_features_list

if TYPE_CHECKING:
    from googlehydrology.utils.config import Config


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

    Replicates the training-time input dropout used in Google's operational
    FloodHub v2 model:

    - **Whole-sequence dropout** (``nan_sequence_probability``): sets an entire
      hindcast or forecast sequence of a feature group to ``NaN``.
    - **Contiguous streak dropout** (``nan_streak_probability``): draws a
      random start timestep and duration and sets that contiguous sub-period
      of a feature group to ``NaN``.

    For forecasting runs, the hindcast window and forecast horizon are sampled
    independently for each feature group, while keeping the overlap slice of
    ``x_d_forecast`` synchronized with ``x_d_hindcast`` for shared groups.

    Groups are processed in an unbiased random permutation per sample while
    enforcing that every timestep with at least one valid (non-NaN) input
    group before dropout retains at least one valid group afterward in every
    input pool.

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
        self._enabled = any(p > 0.0 for p in self._seq_probs.values()) or any(
            p > 0.0 for p in self._streak_probs.values()
        )

    @classmethod
    def from_config(cls, cfg: Config) -> InputDropout:
        """Build an ``InputDropout`` instance from a run ``Config``."""
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

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        """Return a batch dict with synthetic NaN dropout applied.

        Never mutates the caller's batch dictionary or its input tensors in
        place.
        """
        if not self._enabled:
            return data

        if 'x_d_hindcast' in data and 'x_d_forecast' in data:
            return self._apply_forecast_batch(data)
        if 'x_d' in data:
            return self._apply_hindcast_only_batch(data)
        return data

    def _apply_hindcast_only_batch(
        self, data: dict[str, Any]
    ) -> dict[str, Any]:
        x_d: dict[str, torch.Tensor] = data['x_d']
        groups = {
            name: feats
            for name, feats in self._hindcast_groups.items()
            if feats and all(f in x_d for f in feats)
        }
        if len(groups) <= 1:
            return data

        sample_tensor = x_d[next(iter(groups.values()))[0]]
        batch_size, seq_len = sample_tensor.shape[:2]
        if batch_size == 0 or seq_len == 0:
            return data
        device = sample_tensor.device

        names = list(groups)
        valid = torch.stack(
            [_group_valid_mask(x_d, groups[name]) for name in names], dim=0
        )
        candidate = torch.stack(
            [
                _sample_window_drop_mask(
                    batch_size=batch_size,
                    window_len=seq_len,
                    seq_prob=self._seq_probs[name],
                    streak_prob=self._streak_probs[name],
                    device=device,
                )
                for name in names
            ],
            dim=0,
        )
        in_pool = torch.ones((len(names),), dtype=torch.bool, device=device)
        applied = _enforce_survivor_invariants(
            valid=valid,
            candidate=candidate,
            pool_memberships=[in_pool],
            pool_active_slices=[(0, seq_len)],
        )
        masks = {name: applied[i] for i, name in enumerate(names)}

        new_data = dict(data)
        new_data['x_d'] = _mask_feature_dict(x_d, groups, masks)
        return new_data

    def _apply_forecast_batch(self, data: dict[str, Any]) -> dict[str, Any]:
        x_h: dict[str, torch.Tensor] = data['x_d_hindcast']
        x_f: dict[str, torch.Tensor] = data['x_d_forecast']

        hind_groups = _present_groups(self._hindcast_groups, x_h)
        fcst_groups = _present_groups(self._forecast_groups, x_f)
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
            fcst_groups, fcst_overlap_masks, fcst_horizon_masks, dims
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
        if not pass1_names:
            return {}, {}

        valid_list: list[torch.Tensor] = []
        cand_list: list[torch.Tensor] = []
        in_hind_flags: list[bool] = []
        in_fcst_flags: list[bool] = []

        for name in pass1_names:
            in_h = name in hind_groups
            in_f = dims.overlap_len > 0 and name in fcst_groups
            in_hind_flags.append(in_h)
            in_fcst_flags.append(in_f)

            valid_list.append(
                _pass1_group_valid_mask(
                    x_h=x_h,
                    x_f=x_f,
                    hind_feats=hind_groups.get(name),
                    fcst_feats=fcst_groups.get(name) if in_f else None,
                    dims=dims,
                )
            )
            cand_list.append(
                self._sample_pass1_candidate(name=name, in_h=in_h, dims=dims)
            )

        pool_memberships = [
            torch.tensor(in_hind_flags, dtype=torch.bool, device=dims.device)
        ]
        pool_active_slices = [(0, dims.hind_len)]
        if dims.overlap_len > 0 and any(in_fcst_flags):
            pool_memberships.append(
                torch.tensor(
                    in_fcst_flags, dtype=torch.bool, device=dims.device
                )
            )
            pool_active_slices.append((dims.overlap_start, dims.hind_len))

        applied = _enforce_survivor_invariants(
            valid=torch.stack(valid_list, dim=0),
            candidate=torch.stack(cand_list, dim=0),
            pool_memberships=pool_memberships,
            pool_active_slices=pool_active_slices,
        )

        hind_masks: dict[str, torch.Tensor] = {}
        fcst_overlap_masks: dict[str, torch.Tensor] = {}
        for idx, name in enumerate(pass1_names):
            mask = applied[idx]
            if name in hind_groups:
                hind_masks[name] = mask
            if dims.overlap_len > 0 and name in fcst_groups:
                fcst_overlap_masks[name] = mask[:, dims.overlap_start :]
        return hind_masks, fcst_overlap_masks

    def _sample_pass1_candidate(
        self, *, name: str, in_h: bool, dims: _WindowDims
    ) -> torch.Tensor:
        if in_h:
            return _sample_window_drop_mask(
                batch_size=dims.batch_size,
                window_len=dims.hind_len,
                seq_prob=self._seq_probs[name],
                streak_prob=self._streak_probs[name],
                device=dims.device,
            )
        cand = torch.zeros(
            (dims.batch_size, dims.hind_len),
            dtype=torch.bool,
            device=dims.device,
        )
        cand[:, dims.overlap_start :] = _sample_window_drop_mask(
            batch_size=dims.batch_size,
            window_len=dims.overlap_len,
            seq_prob=self._seq_probs[name],
            streak_prob=self._streak_probs[name],
            device=dims.device,
        )
        return cand

    def _sample_forecast_horizon(
        self,
        x_f: dict[str, torch.Tensor],
        hind_groups: dict[str, list[str]],
        fcst_groups: dict[str, list[str]],
        dims: _WindowDims,
    ) -> dict[str, torch.Tensor]:
        """Sample masks on the forecast horizon ``[overlap_len : fcst_len]``."""
        if dims.horizon_len <= 0 or not fcst_groups:
            return {}

        fcst_names = list(fcst_groups)
        valid_list = [
            _group_valid_mask(x_f, fcst_groups[name], start=dims.overlap_len)
            for name in fcst_names
        ]
        cand_list = [
            _sample_window_drop_mask(
                batch_size=dims.batch_size,
                window_len=dims.horizon_len,
                seq_prob=self._seq_probs[name],
                streak_prob=self._streak_probs[name],
                device=dims.device,
            )
            for name in fcst_names
        ]

        in_fcst_horizon = torch.ones(
            (len(fcst_names),), dtype=torch.bool, device=dims.device
        )
        pool_memberships = [in_fcst_horizon]
        pool_active_slices = [(0, dims.horizon_len)]

        shared_flags = [name in hind_groups for name in fcst_names]
        if any(shared_flags) and not all(shared_flags):
            pool_memberships.append(
                torch.tensor(shared_flags, dtype=torch.bool, device=dims.device)
            )
            pool_active_slices.append((0, dims.horizon_len))

        applied = _enforce_survivor_invariants(
            valid=torch.stack(valid_list, dim=0),
            candidate=torch.stack(cand_list, dim=0),
            pool_memberships=pool_memberships,
            pool_active_slices=pool_active_slices,
        )
        return {name: applied[i] for i, name in enumerate(fcst_names)}


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
    first_h = x_h[next(iter(hind_groups.values()))[0]] if hind_groups else None
    first_f = x_f[next(iter(fcst_groups.values()))[0]] if fcst_groups else None
    ref = first_h if first_h is not None else first_f
    assert ref is not None
    batch_size = ref.shape[0]
    if batch_size == 0:
        return None
    hind_len = first_h.shape[1] if first_h is not None else 0
    fcst_len = first_f.shape[1] if first_f is not None else 0
    overlap_len = min(forecast_overlap, hind_len, fcst_len)
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
) -> dict[str, torch.Tensor]:
    combined: dict[str, torch.Tensor] = {}
    for name in fcst_groups:
        parts: list[torch.Tensor] = []
        if dims.overlap_len > 0:
            parts.append(
                overlap_masks.get(
                    name,
                    torch.zeros(
                        (dims.batch_size, dims.overlap_len),
                        dtype=torch.bool,
                        device=dims.device,
                    ),
                )
            )
        if dims.horizon_len > 0:
            parts.append(
                horizon_masks.get(
                    name,
                    torch.zeros(
                        (dims.batch_size, dims.horizon_len),
                        dtype=torch.bool,
                        device=dims.device,
                    ),
                )
            )
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
        if mask is not None and mask.any():
            mask_3d = mask.unsqueeze(-1)
            for feat in feats:
                new_dict[feat] = features_dict[feat].masked_fill(
                    mask_3d, float('nan')
                )
    return new_dict


def _resolve_group_probabilities(
    probs: float | dict[str, float],
    all_groups: list[str],
    *,
    name: str,
) -> dict[str, float]:
    if isinstance(probs, dict):
        unknown = sorted(set(probs) - set(all_groups))
        if unknown:
            raise ValueError(
                f'Unknown feature group(s) in {name}: {unknown}. '
                f'Configured groups: {sorted(all_groups)}.'
            )
        return {group: float(probs.get(group, 0.0)) for group in all_groups}
    return dict.fromkeys(all_groups, float(probs))


def _group_valid_mask(
    features_dict: dict[str, torch.Tensor],
    feature_names: list[str],
    *,
    start: int = 0,
    end: int | None = None,
) -> torch.Tensor:
    """Return bool tensor [B, L] where all features of a group are non-NaN."""
    has_nan = torch.isnan(features_dict[feature_names[0]][:, start:end]).any(
        dim=-1
    )
    for feat in feature_names[1:]:
        has_nan = has_nan | torch.isnan(
            features_dict[feat][:, start:end]
        ).any(dim=-1)
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


def _sample_window_drop_mask(
    *,
    batch_size: int,
    window_len: int,
    seq_prob: float,
    streak_prob: float,
    device: torch.device,
) -> torch.Tensor:
    """Sample candidate drop mask [B, L] combining streak and sequence drops."""
    mask = torch.zeros(
        (batch_size, window_len), dtype=torch.bool, device=device
    )
    if streak_prob > 0.0 and window_len > 0:
        use_streak = torch.rand((batch_size, 1), device=device) < streak_prob
        start = torch.randint(0, window_len, (batch_size, 1), device=device)
        duration = torch.randint(
            1, window_len + 1, (batch_size, 1), device=device
        )
        end = torch.clamp(start + duration, max=window_len)
        steps = torch.arange(window_len, device=device).unsqueeze(0)
        mask = mask | (use_streak & (steps >= start) & (steps < end))

    if seq_prob > 0.0 and window_len > 0:
        use_seq = torch.rand((batch_size, 1), device=device) < seq_prob
        mask = mask | use_seq.expand(batch_size, window_len)

    return mask


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

    if num_groups <= 1 or not candidate.any():
        return torch.zeros_like(candidate)

    # Independent uniform random permutation of groups for every batch element.
    rand_scores = torch.rand((num_groups, batch_size), device=device)
    order = torch.argsort(rand_scores, dim=0)
    idx = order.unsqueeze(-1).expand(num_groups, batch_size, seq_len)

    valid_perm = torch.gather(valid, 0, idx)
    cand_perm = torch.gather(candidate, 0, idx)
    applied_perm = torch.zeros_like(cand_perm)

    # Precompute per-pool remaining valid counts [B, L] and permuted group
    # membership masks [G, B, L]. Outside a pool's active time slice, the
    # membership flag is False so that pool imposes no constraint there.
    pool_remaining: list[torch.Tensor] = []
    pool_member_perm: list[torch.Tensor] = []
    for membership, (slice_start, slice_end) in zip(
        pool_memberships, pool_active_slices, strict=True
    ):
        member_3d = membership.view(num_groups, 1, 1)
        remaining = (valid & member_3d).sum(dim=0)  # [B, L]
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
            remaining.sub_(decrement.to(remaining.dtype))

    applied = torch.zeros_like(candidate)
    applied.scatter_(0, idx, applied_perm)
    return applied
