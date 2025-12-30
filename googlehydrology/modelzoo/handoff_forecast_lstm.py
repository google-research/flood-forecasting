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

import dataclasses

import torch
import torch.nn as nn

from googlehydrology.modelzoo.basemodel import BaseModel
from googlehydrology.modelzoo.fc import FC
from googlehydrology.modelzoo.head import get_head
from googlehydrology.utils.config import Config, EmbeddingSpec, WeightInitOpt
from googlehydrology.utils.lstm_utils import lstm_init

FC_XAVIER = WeightInitOpt.FC_XAVIER


class HandoffForecastLSTM(BaseModel):
    """
    An encoder/decoder LSTM model class used for forecasting.

    This is a forecasting model that uses a state-handoff to transition from a hindcast sequence (LSTM)
    model to a forecast sequence (LSTM) model. The hindcast model is run from the past up to present
    (the issue time of the forecast) and then passes the cell state and hidden state of the LSTM into
    a (nonlinear) handoff network, which is then used to initialize the cell state and hidden state of a
    new LSTM that rolls out over the forecast period. The handoff network is implemented as a custom FC
    network, which can have multiple layers. The handoff network is implemented using the
    ``state_handoff_network`` config parameter.

    The hindcast and forecast LSTMs have different weights and biases, different heads, and can have
    different embedding networks, as defined by ``hindcast_embedding`` and ``forecast_embedding`` in the
    config. The hidden size of the hindcast LSTM is set using the ``hindcast_hidden_size`` config parameter
    and the hidden size of the forecast LSTM is set using the ``forecast_hidden_size`` config parameter,
    which both default to ``hidden_size`` if not set explicitly.

    The handoff forecast LSTM model can implement a delayed handoff, such that the handoff between the
    hindcast and forecast LSTM occurs prior to the forecast issue time. This is controlled by the
    ``forecast_overlap`` parameter in the config file. The forecast and hindcast LSTMs run concurrently
    for the number of timesteps indicated by ``forecast_overlap``. We recommend using the
    ``ForecastOverlapMSERegularization`` regularization option to regularize the loss function by
    (dis)agreement between the overlapping portion of the hindcast and forecast LSTMs. This regularization
    term can be requested by setting  the ``regularization`` parameter list in the config file to include
    ``forecast_overlap``. The model architecture is based on [#]_.

    Parameters
    ----------
    cfg : Config
        The run configuration.

    References
    ----------
    .. [#] Nearing, G., Cohen, D., Dube, V., Gauch, M., Gilon, O., Harrigan, S., ... & Matias, Y. (2024).
       Global prediction of extreme floods in ungauged watersheds. Nature, 627(8004), 559-563.
       https://www.nature.com/articles/s41586-024-07145-1
    """
    # Specify submodules of the model that can later be used for finetuning. Names must match class attributes.
    module_parts = [
        'hindcast_embedding_net',
        'forecast_embedding_net',
        'statics_embedding_net',
        'hindcast_lstm',
        'forecast_lstm',
        'handoff_net',
        'hindcast_head',
        'forecast_head',
    ]

    def __init__(self, cfg: Config):
        super(HandoffForecastLSTM, self).__init__(cfg=cfg)

        self.overlap_output = False
        if 'forecast_overlap' in cfg.regularization:
            self.overlap_output = True
            if cfg.head not in ['regression']:
                raise ValueError('Forecast overlap regularization only works with a regression head.')
           
        self.hindcast_inputs = cfg.hindcast_inputs
        self.forecast_inputs = cfg.forecast_inputs
        
        # Determines whether there is an overlap between forecast and hindcast, which,
        # if present, is used for regularization.
        self.overlap = 0
        if cfg.forecast_overlap is not None:
            self.overlap = cfg.forecast_overlap

        # Data sizes for expanding features in the forward pass.
        self.seq_length = cfg.seq_length
        # TODO (future) :: Models assume that all lead times are present up to the longest `lead_time`.
        # Multimet does not require or enforce this assumption.
        self.lead_time = cfg.lead_time

        # Hidden sizes are necessary for setting initial forget gate biases.
        self.hindcast_hidden_size = cfg.hindcast_hidden_size
        self.forecast_hidden_size = cfg.forecast_hidden_size

        # Input embedding layers.
        if cfg.hindcast_embedding is not None:
            hindcast_embedding = cfg.hindcast_embedding
        elif cfg.dynamics_embedding is not None:
            hindcast_embedding = cfg.hindcast_embedding
        else:
            hindcast_embedding = None
            
        if cfg.hindcast_embedding is not None:
            self.hindcast_embedding_net = self._create_fc(
                embedding_spec=cfg.hindcast_embedding,
                input_size=len(cfg.hindcast_inputs)
            )
            hindcast_embedding_output_size = self.hindcast_embedding_net.output_size
        else:
            hindcast_embedding_output_size = len(cfg.hindcast_inputs)
            self.hindcast_embedding_net = nn.Identity(
                hindcast_embedding_output_size,
                hindcast_embedding_output_size    
            )
            
        if cfg.forecast_embedding is not None:
            forecast_embedding = cfg.forecast_embedding
        elif cfg.forecast_embedding is not None:
            forecast_embedding = cfg.forecast_embedding
        else:
            forecast_embedding = None
            
        if cfg.forecast_embedding is not None:
            self.forecast_embedding_net = self._create_fc(
                embedding_spec=cfg.forecast_embedding,
                input_size=len(cfg.forecast_inputs)
            )
            forecast_embedding_output_size = self.forecast_embedding_net.output_size
        else:
            forecast_embedding_output_size = len(cfg.forecast_inputs)
            self.forecast_embedding_net = nn.Identity(
                forecast_embedding_output_size,
                forecast_embedding_output_size
            )

        if cfg.statics_embedding is not None:
            self.statics_embedding_net = self._create_fc(
                embedding_spec=cfg.statics_embedding,
                input_size=len(cfg.static_attributes)
            )
            statics_embedding_output_size = self.statics_embedding_net.output_size
        else:
            statics_embedding_output_size = len(cfg.static_attributes)
            self.statics_embedding_net = nn.Identity(
                statics_embedding_output_size,
                statics_embedding_output_size
            )

        # Time series layers.
        self.hindcast_lstm = nn.LSTM(
            input_size=hindcast_embedding_output_size + statics_embedding_output_size,
            hidden_size=cfg.hindcast_hidden_size,
            batch_first=True
        )
        self.forecast_lstm = nn.LSTM(
            input_size=forecast_embedding_output_size + statics_embedding_output_size,
            hidden_size=cfg.forecast_hidden_size,
            batch_first=True
        )

        # State handoff layer.
        self.handoff_net = FC(
            input_size=self.hindcast_hidden_size * 2,
            hidden_sizes=cfg.state_handoff_network.hiddens,
            activation=cfg.state_handoff_network.activation,
            dropout=cfg.state_handoff_network.dropout,
            xavier_init=FC_XAVIER in cfg.weight_init_opts,
        )
        self.handoff_linear = FC(
            input_size=cfg.state_handoff_network.hiddens[-1],
            hidden_sizes=[self.forecast_hidden_size * 2],
            activation='linear',
            dropout=0.0,
            xavier_init=FC_XAVIER in cfg.weight_init_opts,
        )

        # Head layers.
        self.dropout = nn.Dropout(p=cfg.output_dropout)
        self.hindcast_head = get_head(
            cfg=cfg, n_in=self.hindcast_hidden_size, n_out=self.output_size
        )
        self.forecast_head = get_head(
            cfg=cfg, n_in=self.forecast_hidden_size, n_out=self.output_size
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

    def forward(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Perform a forward pass on the EncoderDecoderForecastLSTM model.

        Parameters
        ----------
        data : dict[str, torch.Tensor]
            Dictionary, containing input features as key-value pairs.

        Returns
        -------
        dict[str, torch.Tensor]
            Model outputs as a dictionary.
                - `y_hat`: model predictions of shape [batch size, sequence length, number of target variables]..
                - `y_hindcast_overlap`: Output sequence from hindcast model used for regularization
                    [batch size, overlap_sequence length, number of target variables].
                - `y_forecast_overlap`: Output sequence from forecast model used for regularization
                    [batch size, overlap_sequence length, number of target variables].
        """

        # Run the embedding layers.
        hindcast_features = torch.cat(
            [
                t for f, t in data['x_d_hindcast'].items()
                if f in self.hindcast_inputs
            ], dim=-1)
        forecast_features = torch.cat(
            [
                t for f, t in data['x_d_forecast'].items()
                if f in self.forecast_inputs
            ], dim=-1)

        statics_embeddings = self.statics_embedding_net(data['x_s'])
        hindcast_embeddings = self.hindcast_embedding_net(hindcast_features)
        forecast_embeddings = self.forecast_embedding_net(forecast_features)
        
        hindcast_embeddings = torch.cat(
            [
                hindcast_embeddings,
                statics_embeddings.unsqueeze(1).expand(-1, hindcast_embeddings.size(1), -1)
            ], dim=-1
        )
        forecast_embeddings = torch.cat(
            [
                forecast_embeddings,
                statics_embeddings.unsqueeze(1).expand(-1, forecast_embeddings.size(1), -1)
            ], dim=-1
        )
        
        # Run the hindcast LSTM. This happens in two parts. First, the true hindcast
        # or spin-up, then the part the overlaps with the forecast. This is necessary
        # to extract the hidden and cell states at the point of the handoff.
        spinup_embeddings = hindcast_embeddings[:, : -self.overlap,]
        overlap_embeddings = hindcast_embeddings[:, -self.overlap :,]
        spinup, (h_hindcast, c_hindcast) = self.hindcast_lstm(spinup_embeddings)
        hindcast_overlap, _ = self.hindcast_lstm(
            overlap_embeddings, (h_hindcast, c_hindcast)
        )

        # Handoff from hindcast to forecast.
        x = self.handoff_net(torch.cat([h_hindcast, c_hindcast], -1))
        initial_state = self.handoff_linear(x)
        h_handoff, c_handoff = initial_state.chunk(2, -1)
        h_handoff, c_handoff = h_handoff.contiguous(), c_handoff.contiguous()

        # Run the forecast LSTM.
        forecast, _ = self.forecast_lstm(
            forecast_embeddings, (h_handoff, c_handoff)
        )

        # Run head layers.
        y_spinup = self.hindcast_head(self.dropout(spinup))
        y_hindcast_overlap = self.hindcast_head(self.dropout(hindcast_overlap))
        y_forecast = self.forecast_head(self.dropout(forecast))
        
        # Create the full prediction sequence, and only pull the last `seg_length` timesteps.
        output = {
            key: torch.cat(
                [
                    y_spinup[key], 
                    y_hindcast_overlap[key], 
                    y_forecast[key][:, -self.lead_time :, :]
                ], dim=1
            )[:, -self.seq_length :, :] for key in y_spinup
        }
        
        if self.overlap_output:
            y_forecast_overlap = y_forecast['y_hat'][:, : -self.lead_time, :]
            output.update(
                {
                    'y_hindcast_overlap': y_hindcast_overlap['y_hat'],
                    'y_forecast_overlap': y_forecast_overlap,
                }
            )

        return output

