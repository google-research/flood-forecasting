# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dataclasses
import logging
from pathlib import Path
from typing import ClassVar, Iterable

import numpy as np
import torch
import torch.nn as nn

from googlehydrology.modelzoo.basemodel import BaseModel
from googlehydrology.modelzoo.fc import FC
from googlehydrology.modelzoo.head import get_head
from googlehydrology.utils.config import Config, EmbeddingSpec, WeightInitOpt
from googlehydrology.utils.configutils import group_features_list
from googlehydrology.utils.lstm_utils import lstm_init

FC_XAVIER = WeightInitOpt.FC_XAVIER

LOGGER = logging.getLogger(__name__)


class MeanEmbeddingForecastLSTM(BaseModel):
    r"""
    A forecasting model using mean embedding and LSTMs for hindcast and forecast.

    This model implements a specific architecture designed to handle missing input data in hydrological 
    forecasting. It employs separate embedding networks for hindcast and forecast inputs, aggregating 
    them via a masked mean operation. This allows the model to robustly handle situations where some 
    input features might be missing (NaN) by effectively ignoring them in the aggregation step.

    The model consists of two main LSTM components:
    
    1.  **Hindcast LSTM:** Processes historical data (hindcast features) to build up a hidden state 
        representing the system's history up to the forecast issue time.
    2.  **Forecast LSTM:** Takes the final state of the Hindcast LSTM as initialization and unrolls 
        over the forecast horizon using forecast features (e.g., weather forecasts).

    Key features include:
    
    -   **Static Embeddings:** Static catchment attributes are embedded and provided to all dynamic 
        embedding networks.
    -   **Dynamic Embeddings:** Hindcast and forecast features are grouped (e.g., by source or type) 
        and processed by separate, specific fully-connected embedding networks.
    -   **Masked Mean Aggregation:** The outputs of the dynamic embedding networks are aggregated 
        using a masked mean, which ensures that missing data (represented as NaNs) do not propagate 
        errors or bias the embedding.
    -   **Shared Embeddings:** Features present in both hindcast and forecast periods can share 
        embedding networks to enforce consistent representation.

    This model is based on the approach described in [#]_.

    **Data assimilation hooks:** ``forward`` accepts optional keyword arguments
    (``assimilation_overrides``, ``assimilation_slice``, ``return_embeddings``)
    that expose and override the embeddings listed in
    ``supported_assimilation_components``. Dynamic embeddings are overridden
    after the masked mean, i.e. at the point where each LSTM consumes them.
    ``hindcast_embedding`` and ``forecast_embedding`` both include the
    shared-group contribution, so select both to assimilate all dynamic
    information. Overrides cannot cover the forecast horizon (the last
    ``lead_time`` steps). When no keyword arguments are passed the forward
    pass is unchanged.

    Parameters
    ----------
    cfg : Config
        The run configuration, containing all hyperparameters and settings for the model structure, 
        embedding specifications, and input features.

    References
    ----------
    .. [#] Gauch, M., et al. "How to deal w\_ missing input data." Hydrology and Earth System Sciences 29.21 (2025): 6221-6235.
        https://hess.copernicus.org/articles/29/6221/2025/
    """

    # Specify submodules of the model that can later be used for finetuning. Names must match class attributes.
    module_parts = [
        'static_embedding_fc',
        'hindcast_embeddings_fc',
        'forecast_embeddings_fc',
        'shared_embeddings_fc',
        'hindcast_lstm',
        'forecast_lstm',
        'head',
    ]

    # Embeddings returned by `forward(..., return_embeddings=True)` that may be
    # overridden for data assimilation via the keyword argument
    # `assimilation_overrides`.
    supported_assimilation_components: ClassVar[tuple[str, ...]] = (
        'static_embedding',
        'hindcast_embedding',
        'forecast_embedding',
    )

    def __init__(self, cfg: Config):
        super(MeanEmbeddingForecastLSTM, self).__init__(cfg=cfg)

        self.seq_length = cfg.seq_length
        self.lead_time = cfg.lead_time
        self._warned_single_dynamic_assimilation = False

        self.config_data = ConfigData.from_config(cfg)

        # Static embedding
        self.static_embedding_fc = self._create_fc(
            embedding_spec=self.config_data.statics_embedding,
            input_size=len(self.config_data.static_attributes),
        )

        # Hindcast embedding networks
        self.hindcast_embeddings_fc = nn.ModuleDict(
            {
                name: self._create_fc(
                    embedding_spec=self.config_data.hindcast_embedding,
                    input_size=(
                        len(self.config_data.hindcast_inputs_grouped[name])
                        + self.static_embedding_fc.output_size
                    ),
                )
                for name in set(
                    self.config_data.hindcast_inputs_grouped.keys()
                ).difference(self.config_data.shared_groups)
            }
        )
        # Forecast embedding networks
        self.forecast_embeddings_fc = nn.ModuleDict(
            {
                name: self._create_fc(
                    embedding_spec=self.config_data.forecast_embedding,
                    input_size=(
                        len(self.config_data.forecast_inputs_grouped[name])
                        + self.static_embedding_fc.output_size
                    ),
                )
                for name in set(
                    self.config_data.forecast_inputs_grouped.keys()
                ).difference(self.config_data.shared_groups)
            }
        )
        # Shared embedding networks (between hindcast and forecast LSTMs)
        self.shared_embeddings_fc = nn.ModuleDict(
            {
                name: self._create_fc(
                    embedding_spec=self.config_data.forecast_embedding,
                    input_size=(
                        len(self.config_data.forecast_inputs_grouped[name])
                        + self.static_embedding_fc.output_size
                    ),
                )
                for name in self.config_data.shared_groups
            }
        )

        # Hindcast LSTM
        self.hindcast_lstm = nn.LSTM(
            input_size=self.static_embedding_fc.output_size
            + self.config_data.hindcast_embedding.hiddens[-1],
            hidden_size=self.config_data.hidden_size,
            batch_first=True,
        )

        # Forecast LSTM
        self.forecast_lstm = nn.LSTM(
            input_size=self.static_embedding_fc.output_size
            + self.config_data.forecast_embedding.hiddens[-1]
            + self.config_data.hidden_size,
            hidden_size=self.config_data.hidden_size,
            batch_first=True,
        )

        # Head
        self.dropout = nn.Dropout(p=cfg.output_dropout)
        self.head = get_head(
            self.cfg,
            n_in=self.config_data.hidden_size,
            n_out=self.output_size,
            n_hidden=100,
        )

        lstm_init(
            lstms=[self.hindcast_lstm, self.forecast_lstm],
            forget_bias=cfg.initial_forget_bias,
            weight_opts=cfg.weight_init_opts,
        )

    def _create_fc(self, embedding_spec: EmbeddingSpec, input_size: int) -> FC:
        assert input_size > 0, 'Cannot create embedding layer with input size 0'

        emb_type = embedding_spec.type.lower()
        assert emb_type == 'fc', f'{emb_type=} not supported'

        hiddens = embedding_spec.hiddens
        assert len(hiddens) > 0, 'hiddens must have at least one entry'

        activation = embedding_spec.activation
        assert len(activation) == len(hiddens), (
            'hiddens and activation layers must match'
        )

        dropout = float(embedding_spec.dropout)

        return FC(
            input_size=input_size,
            hidden_sizes=hiddens,
            activation=activation,
            dropout=dropout,
            xavier_init=FC_XAVIER in self.cfg.weight_init_opts,
        )

    def forward(
        self,
        data: dict[str, torch.Tensor | dict[str, torch.Tensor]],
        *,
        assimilation_overrides: dict[str, torch.Tensor] | None = None,
        assimilation_slice: tuple[int, int] | None = None,
        return_embeddings: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Perform a forward pass on the MeanEmbeddingForecastLSTM model.

        Parameters
        ----------
        data : dict[str, torch.Tensor | dict[str, torch.Tensor]]
            Dictionary, containing input features as key-value pairs.
        assimilation_overrides : dict[str, torch.Tensor] | None, optional
            Mapping from names in ``supported_assimilation_components`` to
            tensors replacing the corresponding embeddings. A dynamic override
            either covers the full sequence or, together with
            ``assimilation_slice``, a slice of the observed period.
        assimilation_slice : tuple[int, int] | None, optional
            ``(start, end)`` time indices on the model time axis where a
            partial-length dynamic embedding override is spliced in. Must
            satisfy ``0 <= start < end <= T - lead_time`` where ``T`` is the
            length of the dynamic embeddings, i.e. it cannot reach into the
            forecast horizon (the last ``lead_time`` steps).
        return_embeddings : bool, optional
            If True, the output additionally contains ``'static_embedding'``
            [B, E_s], ``'hindcast_embedding'`` [B, T, E_h] and
            ``'forecast_embedding'`` [B, T, E_f]. The embeddings are also
            returned whenever ``assimilation_overrides`` is non-empty.

        Returns
        -------
        dict[str, torch.Tensor]
            Model outputs from the head, plus the (possibly overridden)
            embeddings if requested (see ``return_embeddings``).
        """
        forward_data = ForwardData.from_forward_data(data, self.config_data)
        overrides = assimilation_overrides or {}
        if overrides:
            self.validate_assimilation_components(overrides)

        if 'static_embedding' in overrides:
            static_embedding = self._validated_static_override(
                forward_data, overrides['static_embedding']
            )
        else:
            static_embedding = self._calc_static_embedding(forward_data)

        mean_hindcast_embedding, mean_forecast_embedding = (
            self._calc_mean_embeddings(forward_data, static_embedding)
        )
        mean_hindcast_embedding = self._apply_embedding_override(
            'hindcast_embedding',
            mean_hindcast_embedding,
            overrides,
            assimilation_slice,
        )
        mean_forecast_embedding = self._apply_embedding_override(
            'forecast_embedding',
            mean_forecast_embedding,
            overrides,
            assimilation_slice,
        )

        state = getattr(self, '_preloaded_state', None)
        h_hind_init = None
        h_fore_init = None
        if state is not None:
            device = static_embedding.device
            dtype = static_embedding.dtype
            h_hind_arr = state.get('h_hindcast', state.get('h_hind'))
            c_hind_arr = state.get('c_hindcast', state.get('c_hind'))
            h_fore_arr = state.get('h_forecast', state.get('h_fore'))
            c_fore_arr = state.get('c_forecast', state.get('c_fore'))

            def _to_3d_tensor(arr):
                t = torch.from_numpy(arr).to(device=device, dtype=dtype)
                while t.ndim < 3:
                    t = t.unsqueeze(0)
                return t

            if h_hind_arr is not None and c_hind_arr is not None:
                h_hind_init = (
                    _to_3d_tensor(h_hind_arr),
                    _to_3d_tensor(c_hind_arr),
                )
            if h_fore_arr is not None and c_fore_arr is not None:
                h_fore_init = (
                    _to_3d_tensor(h_fore_arr),
                    _to_3d_tensor(c_fore_arr),
                )

        # Time steps where every group feeding a masked mean is NaN (e.g. the
        # NaN-padded forecast horizon of the hindcast groups) have no valid
        # LSTM input. The LSTMs run on zero-filled inputs and the head outputs
        # from the first such step onwards are set to NaN afterwards; see
        # `_missing_steps`.
        hindcast_missing = self._missing_steps(mean_hindcast_embedding)
        forecast_missing = hindcast_missing | self._missing_steps(
            mean_forecast_embedding
        )
        hindcast_state = self._calc_lstm(
            lstm=self.hindcast_lstm,
            masked_mean_embeddings=mean_hindcast_embedding,
            static_embedding=static_embedding,
            initial_state=h_hind_init,
        )
        forecast_state = self._calc_lstm(
            lstm=self.forecast_lstm,
            masked_mean_embeddings=mean_forecast_embedding,
            static_embedding=static_embedding,
            other_inputs=hindcast_state,
            initial_state=h_fore_init,
        )

        head = {
            key: value.masked_fill(forecast_missing, float('nan'))
            for key, value in self._calc_head(forecast_state).items()
        }
        if return_embeddings or overrides:
            head['static_embedding'] = static_embedding
            head['hindcast_embedding'] = mean_hindcast_embedding
            head['forecast_embedding'] = mean_forecast_embedding
        return head

    def validate_assimilation_components(self, names: Iterable[str]) -> None:
        """Raise ValueError for names not in supported_assimilation_components.

        Additionally warns (once per model instance) when only one of the two
        dynamic embeddings is selected: both ``hindcast_embedding`` and
        ``forecast_embedding`` include the shared-input-group contribution, so
        assimilating only one of them leaves the shared contribution of the
        other un-assimilated.
        """
        names = set(names)
        super().validate_assimilation_components(names)
        dynamic = {'hindcast_embedding', 'forecast_embedding'}
        selected = sorted(names & dynamic)
        if len(selected) == 1 and not self._warned_single_dynamic_assimilation:
            self._warned_single_dynamic_assimilation = True
            (other,) = dynamic - set(selected)
            LOGGER.warning(
                'Assimilating only %s: both hindcast_embedding and '
                'forecast_embedding include the shared-input-group '
                'contribution, so the shared contribution of %s is left '
                'un-assimilated. Select both to assimilate all dynamic '
                'information.',
                selected[0],
                other,
            )

    def point_prediction(
        self, outputs: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Deterministic prediction ``[B, T, n_targets]`` via the head."""
        return self.head.point_prediction(outputs)

    def _calc_mean_embeddings(
        self, forward_data: 'ForwardData', static_embedding: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Masked-mean hindcast and forecast embeddings, each [B, T, E].

        Every dynamic input group is embedded by its own network; the hindcast
        mean averages the hindcast-only and shared groups, the forecast mean
        the forecast-only and shared groups (shared groups use the forecast
        data). Missing groups (NaN) are skipped by the masked mean.
        """

        def embed(
            networks: nn.ModuleDict,
            features: dict[str, torch.Tensor],
            *,
            append_nan: bool,
        ) -> list[torch.Tensor]:
            return [
                self._calc_dynamic_embedding(
                    embedding_network=fc,
                    dynamic_data=features[name],
                    static_embedding=static_embedding,
                    append_nan=append_nan,
                )
                for name, fc in networks.items()
            ]

        hindcast = embed(
            self.hindcast_embeddings_fc,
            forward_data.hindcast_features,
            append_nan=True,
        )
        forecast = embed(
            self.forecast_embeddings_fc,
            forward_data.forecast_features,
            append_nan=False,
        )
        # Shared embeddings are using the forecast data
        shared = embed(
            self.shared_embeddings_fc,
            forward_data.forecast_features,
            append_nan=False,
        )
        return (
            self._masked_mean(hindcast + shared),
            self._masked_mean(forecast + shared),
        )

    def _validated_static_override(
        self, forward_data: 'ForwardData', override: torch.Tensor
    ) -> torch.Tensor:
        """Check that `override` [B, E_s] can replace the static embedding."""
        if forward_data.static_features is None:
            msg = (
                'Cannot override static_embedding when the model has no '
                'static inputs.'
            )
            raise ValueError(msg)
        expected = (
            forward_data.static_features.shape[0],
            self.static_embedding_fc.output_size,
        )
        if tuple(override.shape) != expected:
            msg = (
                'static_embedding override has shape '
                f'{tuple(override.shape)}, expected {expected}'
            )
            raise ValueError(msg)
        expected_dtype = next(self.static_embedding_fc.parameters()).dtype
        if override.dtype != expected_dtype:
            msg = (
                f'static_embedding override has dtype {override.dtype}, '
                f'expected {expected_dtype}'
            )
            raise ValueError(msg)
        return override

    def _apply_embedding_override(
        self,
        name: str,
        embedding: torch.Tensor,
        overrides: dict[str, torch.Tensor],
        assimilation_slice: tuple[int, int] | None,
    ) -> torch.Tensor:
        """Replace `embedding[:, start:end]` with the override, if given.

        `(start, end) = assimilation_slice` and
        `end - start == override.shape[1]`. A full-length override may omit
        the slice, which then defaults to `(0, T)`. In every case the slice
        must lie within the observed period: it cannot reach into the
        forecast horizon (the last `lead_time` steps), so a full-length
        override is only valid when `lead_time == 0`.
        """
        override = overrides.get(name)
        if override is None:
            return embedding
        if override.dtype != embedding.dtype:
            msg = (
                f'{name} override has dtype {override.dtype}, '
                f'expected {embedding.dtype}'
            )
            raise ValueError(msg)
        if override.ndim != embedding.ndim or (
            override.shape[0] != embedding.shape[0]
            or override.shape[2] != embedding.shape[2]
        ):
            msg = (
                f'{name} override has shape {tuple(override.shape)}, '
                f'incompatible with {tuple(embedding.shape)}'
            )
            raise ValueError(msg)
        length = embedding.shape[1]
        if assimilation_slice is None:
            if override.shape[1] != length:
                msg = (
                    f'{name} override covers {override.shape[1]} of {length} '
                    'time steps; assimilation_slice=(start, end) is required'
                )
                raise ValueError(msg)
            assimilation_slice = (0, length)
        start, end = (int(e) for e in assimilation_slice)
        last_observed = length - self.cfg.lead_time
        if not 0 <= start < end <= last_observed:
            msg = (
                f'assimilation_slice {assimilation_slice} out of range: it '
                f'must lie within the observed period [0, {last_observed}]; '
                f'the last {self.cfg.lead_time} steps are the forecast horizon'
            )
            raise ValueError(msg)
        if end - start != override.shape[1]:
            msg = (
                f'{name} override has {override.shape[1]} time steps but '
                f'assimilation_slice {assimilation_slice} spans {end - start}'
            )
            raise ValueError(msg)
        return torch.cat(
            [embedding[:, :start], override, embedding[:, end:]], dim=1
        )

    @torch.no_grad()
    def save_state(
        self,
        data: dict[str, torch.Tensor | dict[str, torch.Tensor]],
        path: str | Path,
    ) -> None:
        """Perform a partial forward pass and save state for a hot start at path.

        Parameters
        ----------
        data : dict[str, torch.Tensor | dict[str, torch.Tensor]]
            Dictionary containing input features as key-value pairs.
        path : str | Path
            The file path where the state should be saved (.npz format).
        """
        forward_data = ForwardData.from_forward_data(data, self.config_data)

        static_embedding = self._calc_static_embedding(forward_data)

        # Only the observed period feeds the saved state.
        mean_hindcast_embedding, mean_forecast_embedding = (
            e[:, : self.seq_length, :]
            for e in self._calc_mean_embeddings(forward_data, static_embedding)
        )

        hindcast_state, (h_hind, c_hind) = self._calc_lstm(
            lstm=self.hindcast_lstm,
            masked_mean_embeddings=mean_hindcast_embedding,
            static_embedding=static_embedding,
            return_state=True,
        )
        forecast_state, (h_fore, c_fore) = self._calc_lstm(
            lstm=self.forecast_lstm,
            masked_mean_embeddings=mean_forecast_embedding,
            static_embedding=static_embedding,
            other_inputs=hindcast_state,
            return_state=True,
        )

        np.savez_compressed(
            path,
            h_hindcast=h_hind.detach().cpu().numpy(),
            c_hindcast=c_hind.detach().cpu().numpy(),
            h_forecast=h_fore.detach().cpu().numpy(),
            c_forecast=c_fore.detach().cpu().numpy(),
        )

    def load_state_from_disk(self, path: str | Path) -> None:
        """Pre-load a hot start state archive from disk into memory.

        Parameters
        ----------
        path : str | Path
            Path to the .npz state file to load.
        """
        self._preloaded_state = dict(np.load(path, allow_pickle=False))

    def _make_static_embedding_repeated(
        self, time_length: int, static_embedding: torch.Tensor
    ) -> torch.Tensor:
        """Returns the attributes repeated w.r.t the time length."""
        return static_embedding.unsqueeze(1).repeat(1, time_length, 1)

    def _make_nan_padding(
        self,
        batch_size: int,
        nan_padding_length: int,
        embedding_size: int,
        device: str,
    ) -> torch.Tensor:
        """Returns a nan-padding tensor."""
        return torch.full(
            (batch_size, nan_padding_length, embedding_size),
            np.nan,
            device=device,
        )

    def _append_static_embedding(
        self, embedding: torch.Tensor, static_embedding: torch.Tensor
    ) -> torch.Tensor:
        """Append static embedding to another embedding tensor."""
        # Dimension 1 is the time dimension. Duplicate static embedding in all time series.
        time_length = embedding.shape[1]
        static_embedding_repeated = self._make_static_embedding_repeated(
            time_length, static_embedding
        )
        return torch.cat([embedding, static_embedding_repeated], dim=-1)

    def _add_nan_padding(self, embedding: torch.Tensor) -> torch.Tensor:
        """Pad the embedding tensor with nan value to timespan of hindcast and forecast."""
        # Dimension 0 is the batch size. Note the batch size may change during training.
        batch_size = embedding.shape[0]
        # Dimension 1 is the time dimension. Pad nan to the full sequence length plus lead time.
        nan_padding_length = (
            self.seq_length + self.lead_time - embedding.shape[1]
        )
        # Dimension 2 is the length of embedding vector.
        embedding_size = embedding.shape[2]
        nan_padding = self._make_nan_padding(
            batch_size, nan_padding_length, embedding_size, embedding.device
        )
        return torch.cat([embedding, nan_padding], dim=1)

    def _masked_mean(self, tensors: Iterable[torch.Tensor]) -> torch.Tensor:
        """Calculate mean between list of tensors, skipping nan values. Calculates mean of the last dimension.
        All tensors have same dimensions."""
        merged = torch.cat([e.unsqueeze(-1) for e in tensors], dim=-1)
        return torch.nanmean(merged, dim=-1)

    def _calc_static_embedding(
        self, forward_data: 'ForwardData'
    ) -> torch.Tensor:
        return self.static_embedding_fc(forward_data.static_features)

    def _calc_dynamic_embedding(
        self,
        embedding_network: nn.Module,
        dynamic_data: torch.Tensor,
        static_embedding: torch.Tensor,
        append_nan: bool,
    ) -> torch.Tensor:
        # Zero out time steps with missing inputs before the network and set
        # them back to NaN afterwards. Forward values are unchanged, but this
        # keeps gradients w.r.t. parameters and static embedding finite.
        nan_mask = torch.isnan(dynamic_data).any(dim=-1, keepdim=True)
        dynamic_data = dynamic_data.masked_fill(nan_mask, 0.0)
        dynamic_data_concat = self._append_static_embedding(
            dynamic_data, static_embedding
        )
        output = embedding_network(dynamic_data_concat)
        output = output.masked_fill(nan_mask, float('nan'))
        if append_nan:
            output = self._add_nan_padding(output)
        return output

    @staticmethod
    def _missing_steps(masked_mean_embedding: torch.Tensor) -> torch.Tensor:
        """Mask [B, T, 1] of steps without valid input, and all steps after.

        A step is missing when every group feeding the masked mean is NaN.
        The LSTM state is undefined from the first missing step onwards, so
        the mask is cumulative along time. Callers run the LSTMs on
        zero-filled inputs and set the head outputs to NaN where this mask is
        True, which reproduces the NaN pattern of feeding NaN to the LSTM
        while keeping all gradients finite.
        """
        missing = torch.isnan(masked_mean_embedding).any(dim=-1, keepdim=True)
        return torch.cummax(missing.to(torch.int8), dim=1).values.bool()

    def _calc_lstm(
        self,
        lstm: nn.LSTM,
        masked_mean_embeddings: torch.Tensor,
        static_embedding: torch.Tensor,
        other_inputs: torch.Tensor | None = None,
        initial_state: tuple[torch.Tensor, torch.Tensor] | None = None,
        return_state: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if other_inputs is not None:
            masked_mean_embeddings = torch.cat(
                [masked_mean_embeddings, other_inputs], dim=-1
            )
        lstm_inputs = self._append_static_embedding(
            masked_mean_embeddings, static_embedding
        )
        # Zero-fill missing time steps (see `_missing_steps`). Fed directly,
        # a NaN input poisons the recurrent state from that step on and, in
        # the backward pass, every earlier step as well (0 * NaN = NaN).
        lstm_inputs = lstm_inputs.nan_to_num(nan=0.0)
        if initial_state is not None:
            output, hx = lstm(input=lstm_inputs, hx=initial_state)
        else:
            output, hx = lstm(input=lstm_inputs)
        if return_state:
            return output, hx
        return output

    def _calc_head(
        self, forecast_state: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        return self.head(self.dropout(forecast_state))


@dataclasses.dataclass(frozen=True, kw_only=True)
class ConfigData:
    @classmethod
    def from_config(cls, cfg: Config) -> 'ConfigData':
        statics_embedding = cfg.statics_embedding
        hindcast_embedding = cfg.hindcast_embedding or cfg.dynamics_embedding
        forecast_embedding = cfg.forecast_embedding or cfg.dynamics_embedding
        assert statics_embedding is not None
        assert hindcast_embedding is not None
        assert forecast_embedding is not None

        hindcast_inputs_grouped = group_features_list(cfg.hindcast_inputs)
        forecast_inputs_grouped = group_features_list(cfg.forecast_inputs)
        shared_groups = [
            e for e in hindcast_inputs_grouped if e in forecast_inputs_grouped
        ]
        for group in shared_groups:
            assert (
                hindcast_inputs_grouped[group] == forecast_inputs_grouped[group]
            ), (
                f'Same features must be defined in forecast and hindcast for {group=}'
            )

        return ConfigData(
            hidden_size=cfg.hidden_size,
            statics_embedding=statics_embedding,
            hindcast_embedding=hindcast_embedding,
            forecast_embedding=forecast_embedding,
            static_attributes=tuple(cfg.static_attributes),
            hindcast_inputs_grouped=hindcast_inputs_grouped,
            forecast_inputs_grouped=forecast_inputs_grouped,
            shared_groups=shared_groups,
        )

    hidden_size: int
    statics_embedding: EmbeddingSpec
    hindcast_embedding: EmbeddingSpec
    forecast_embedding: EmbeddingSpec
    static_attributes: tuple[str, ...]
    hindcast_inputs_grouped: dict[str, list[str]]
    forecast_inputs_grouped: dict[str, list[str]]
    shared_groups: list[str]


@dataclasses.dataclass(frozen=True, kw_only=True)
class ForwardData:
    @classmethod
    def from_forward_data(
        cls,
        data: dict[str, torch.Tensor | dict[str, torch.Tensor]],
        config_data: ConfigData,
    ) -> 'ForwardData':
        return ForwardData(
            static_features=data['x_s'],
            hindcast_features={
                name: _concat_tensors_from_dict(
                    data['x_d_hindcast'], keys=features
                )
                for name, features in config_data.hindcast_inputs_grouped.items()
            },
            forecast_features={
                name: _concat_tensors_from_dict(
                    data['x_d_forecast'], keys=features
                )
                for name, features in config_data.forecast_inputs_grouped.items()
            },
        )

    static_features: torch.Tensor
    hindcast_features: dict[str, torch.Tensor]
    forecast_features: dict[str, torch.Tensor]


def _concat_tensors_from_dict(
    data: dict[str, torch.Tensor], *, keys: Iterable[str]
) -> torch.Tensor:
    return torch.cat([data[e] for e in keys], dim=-1)
