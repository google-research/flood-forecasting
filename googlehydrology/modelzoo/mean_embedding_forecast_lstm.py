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

from typing import Iterable

import dataclasses
import numpy as np
import torch
import torch.nn as nn

from googlehydrology.modelzoo.basemodel import BaseModel
from googlehydrology.modelzoo.fc import FC
from googlehydrology.modelzoo.head import get_head
from googlehydrology.utils.config import Config, WeightInitOpt, EmbeddingSpec
from googlehydrology.utils.lstm_utils import lstm_init
from googlehydrology.utils.configutils import group_features_list

FC_XAVIER = WeightInitOpt.FC_XAVIER

class MeanEmbeddingForecastLSTM(BaseModel):
    """A forecasting model using mean embedding and LSTMs for hindcast and forecast.

    Parameters
    ----------
    cfg : Config
        The run configuration.
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

    def __init__(self, cfg: Config):
        super(MeanEmbeddingForecastLSTM, self).__init__(cfg=cfg)

        self.seq_length = cfg.seq_length
        self.lead_time = cfg.lead_time

        self.config_data = ConfigData.from_config(cfg)

        # Static embedding
        self.static_embedding_fc = self._create_fc(
            embedding_spec = self.config_data.statics_embedding,
            input_size=len(self.config_data.static_attributes),
        )

        # Hindcast embedding networks
        self.hindcast_embeddings_fc = nn.ModuleDict({
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
        })
        # Forecast embedding networks
        self.forecast_embeddings_fc = nn.ModuleDict({
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
        })
        # Shared embedding networks (between hindcast and forecast LSTMs)
        self.shared_embeddings_fc = nn.ModuleDict({
            name: self._create_fc(
                embedding_spec=self.config_data.forecast_embedding,
                input_size=(
                    len(self.config_data.forecast_inputs_grouped[name])
                    + self.static_embedding_fc.output_size
                ),
            )
            for name in self.config_data.shared_groups
        })

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
            self.cfg, n_in=self.config_data.hidden_size, n_out=3 * 4, n_hidden=100
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
        self, data: dict[str, torch.Tensor | dict[str, torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
        """Perform a forward pass on the MeanEmbeddingForecastLSTM model.

        Parameters
        ----------
        data : dict[str, torch.Tensor | dict[str, torch.Tensor]]
            Dictionary, containing input features as key-value pairs.

        Returns
        -------
        dict[str, torch.Tensor]
            Model outputs and intermediate states as a dictionary from CMAL head.
        """
        forward_data = ForwardData.from_forward_data(data, self.config_data)

        static_embedding = self._calc_static_embedding(forward_data)

        hindcast_embeddings = [
            self._calc_dynamic_embedding(
                embedding_network=fc,
                dynamic_data=forward_data.hindcast_features[name],
                static_embedding=static_embedding,
                append_nan=True,
            )
            for name, fc in self.hindcast_embeddings_fc.items()
        ]
        forecast_embeddings = [
            self._calc_dynamic_embedding(
                embedding_network=fc,
                dynamic_data=forward_data.forecast_features[name],
                static_embedding=static_embedding,
                append_nan=False,
            )
            for name, fc in self.forecast_embeddings_fc.items()
        ]
        # Shared embeddings are using the forecast data
        shared_embeddings = [
            self._calc_dynamic_embedding(
                embedding_network=fc,
                dynamic_data=forward_data.forecast_features[name],
                static_embedding=static_embedding,
                append_nan=False,
            )
            for name, fc in self.shared_embeddings_fc.items()
        ]

        hindcast_state = self._calc_lstm(
            lstm=self.hindcast_lstm,
            embeddings=hindcast_embeddings + shared_embeddings,
            static_embedding=static_embedding,
        )
        forecast_state = self._calc_lstm(
            lstm=self.forecast_lstm,
            embeddings=forecast_embeddings + shared_embeddings,
            static_embedding=static_embedding,
            other_inputs=hindcast_state,
        )

        head = self._calc_head(forecast_state)

        return head

    def _make_static_embedding_repeated(
        self, time_length: int, static_embedding: torch.Tensor
    ) -> torch.Tensor:
        """Returns the attributes repeated w.r.t the time length."""
        return static_embedding.unsqueeze(1).repeat(1, time_length, 1)

    def _make_nan_padding(
        self, batch_size: int, nan_padding_length: int, embedding_size: int, device: str
    ) -> torch.Tensor:
        """Returns a nan-padding tensor."""
        return torch.full(
            (batch_size, nan_padding_length, embedding_size), np.nan, device=device
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
        nan_padding_length = self.seq_length + self.lead_time - embedding.shape[1]
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

    def _calc_static_embedding(self, forward_data: "ForwardData") -> torch.Tensor:
        return self.static_embedding_fc(forward_data.static_features)

    def _calc_dynamic_embedding(
        self,
        embedding_network: nn.Module,
        dynamic_data: torch.Tensor,
        static_embedding: torch.Tensor,
        append_nan: bool,
    ) -> torch.Tensor:
        dynamic_data_concat = self._append_static_embedding(
            dynamic_data, static_embedding
        )
        output = embedding_network(dynamic_data_concat)
        if append_nan:
            output = self._add_nan_padding(output)
        return output

    def _calc_lstm(
        self,
        lstm: nn.LSTM,
        embeddings: Iterable[torch.Tensor],
        static_embedding: torch.Tensor,
        other_inputs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        masked_mean_embeddings = self._masked_mean(embeddings)
        if other_inputs is not None:
            masked_mean_embeddings = torch.cat(
                [masked_mean_embeddings, other_inputs], dim=-1
            )
        lstm_inputs = self._append_static_embedding(
            masked_mean_embeddings, static_embedding
        )
        output, _ = lstm(input=lstm_inputs)
        return output

    def _calc_head(self, forecast_state: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.head(self.dropout(forecast_state))


@dataclasses.dataclass(frozen=True, kw_only=True)
class ConfigData:
    @classmethod
    def from_config(cls, cfg: Config) -> 'ConfigData':
        assert (statics_embedding := cfg.statics_embedding) is not None
        assert (hindcast_embedding := cfg.hindcast_embedding) is not None
        assert (forecast_embedding := cfg.forecast_embedding) is not None

        hindcast_inputs_grouped = group_features_list(cfg.hindcast_inputs)
        forecast_inputs_grouped = group_features_list(cfg.forecast_inputs)
        shared_groups = set(hindcast_inputs_grouped) & set(forecast_inputs_grouped)
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
    hindcast_inputs_grouped: dict[str, set[str]]
    forecast_inputs_grouped: dict[str, set[str]]
    shared_groups: set[str]


@dataclasses.dataclass(frozen=True, kw_only=True)
class ForwardData:
    @classmethod
    def from_forward_data(
        cls,
        data: dict[str, torch.Tensor | dict[str, torch.Tensor]],
        config_data: ConfigData,
    ) -> "ForwardData":
        return ForwardData(
            static_features=data["x_s"],
            hindcast_features = {
                name: _concat_tensors_from_dict(data["x_d_hindcast"], keys=features)
                for name, features in config_data.hindcast_inputs_grouped.items()
            },
            forecast_features = {
                name: _concat_tensors_from_dict(data["x_d_forecast"], keys=features)
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
