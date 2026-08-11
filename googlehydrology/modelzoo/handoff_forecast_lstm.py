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

from pathlib import Path

import numpy as np
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
           
        if isinstance(cfg.hindcast_inputs, dict):
            self.hindcast_inputs = [
                f for feats in cfg.hindcast_inputs.values() for f in feats
            ]
        else:
            self.hindcast_inputs = cfg.hindcast_inputs

        if isinstance(cfg.forecast_inputs, dict):
            self.forecast_inputs = [
                f for feats in cfg.forecast_inputs.values() for f in feats
            ]
        else:
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
                input_size=len(self.hindcast_inputs)
            )
            hindcast_embedding_output_size = self.hindcast_embedding_net.output_size
        else:
            hindcast_embedding_output_size = len(self.hindcast_inputs)
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
                input_size=len(self.forecast_inputs)
            )
            forecast_embedding_output_size = self.forecast_embedding_net.output_size
        else:
            forecast_embedding_output_size = len(self.forecast_inputs)
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
        state = getattr(self, '_preloaded_state', None)
        hindcast_initial_state = None
        forecast_initial_state = None
        is_hot_start = False
        if state is not None:
            device = forecast_embeddings.device
            dtype = forecast_embeddings.dtype
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
                hindcast_initial_state = (
                    _to_3d_tensor(h_hind_arr),
                    _to_3d_tensor(c_hind_arr),
                )
            if h_fore_arr is not None and c_fore_arr is not None:
                forecast_initial_state = (
                    _to_3d_tensor(h_fore_arr),
                    _to_3d_tensor(c_fore_arr),
                )

            # Even if overlap is > 0, if seq_length == 0 on a hot start we have 0
            # hindcast elements. We flag this so we can bypass spinup processing.
            if hindcast_embeddings.size(1) == 0:
                is_hot_start = True

        if is_hot_start:
            # Bypass spinup entirely. We shouldn't run hindcast_lstm on 0-length inputs.
            # Spinup and hindcast_overlap outputs are empty for a length-0 hot start.
            spinup = torch.empty(
                hindcast_embeddings.size(0),
                0,
                self.hindcast_hidden_size,
                device=hindcast_embeddings.device,
            )
            hindcast_overlap = torch.empty(
                hindcast_embeddings.size(0),
                0,
                self.hindcast_hidden_size,
                device=hindcast_embeddings.device,
            )

            # The initial state is already at the correct temporal index (end of
            # historical overlap).
            h_handoff, c_handoff = forecast_initial_state
            if forecast_embeddings.size(1) > self.lead_time:
                forecast_embeddings = forecast_embeddings[
                    :, -self.lead_time :, :
                ]
        else:
            # Normal cold-start or save-state propagation
            if self.overlap > 0:
                spinup_embeddings = hindcast_embeddings[:, : -self.overlap]
                overlap_embeddings = hindcast_embeddings[:, -self.overlap :]
            else:
                spinup_embeddings = hindcast_embeddings
                overlap_embeddings = hindcast_embeddings[:, 0:0, :]

            if spinup_embeddings.size(1) > 0:
                if hindcast_initial_state is not None:
                    spinup, (h_hindcast, c_hindcast) = self.hindcast_lstm(
                        spinup_embeddings, hindcast_initial_state
                    )
                else:
                    spinup, (h_hindcast, c_hindcast) = self.hindcast_lstm(
                        spinup_embeddings
                    )
            else:
                spinup = torch.empty(
                    hindcast_embeddings.size(0),
                    0,
                    self.hindcast_hidden_size,
                    device=hindcast_embeddings.device,
                )
                # If spinup is length 0 but overlap > 0, we use loaded initial state
                if hindcast_initial_state is not None:
                    h_hindcast, c_hindcast = hindcast_initial_state
                else:
                    h_hindcast = torch.zeros(
                        1,
                        hindcast_embeddings.size(0),
                        self.hindcast_hidden_size,
                        device=hindcast_embeddings.device,
                    )
                    c_hindcast = torch.zeros(
                        1,
                        hindcast_embeddings.size(0),
                        self.hindcast_hidden_size,
                        device=hindcast_embeddings.device,
                    )

            if overlap_embeddings.size(1) > 0:
                hindcast_overlap, (h_hind_final, c_hind_final) = (
                    self.hindcast_lstm(
                        overlap_embeddings, (h_hindcast, c_hindcast)
                    )
                )
            else:
                hindcast_overlap = torch.empty(
                    hindcast_embeddings.size(0),
                    0,
                    self.hindcast_hidden_size,
                    device=hindcast_embeddings.device,
                )
                h_hind_final, c_hind_final = h_hindcast, c_hindcast

            # Handoff from hindcast to forecast.
            x = self.handoff_net(torch.cat([h_hindcast, c_hindcast], -1))
            initial_state = self.handoff_linear(x)
            h_handoff, c_handoff = initial_state.chunk(2, -1)
            h_handoff, c_handoff = (
                h_handoff.contiguous(),
                c_handoff.contiguous(),
            )

        # Run the forecast LSTM.
        # If hot start, forecast_embeddings is just the lead_time (no overlap),
        # so we just run it directly on the state.
        # But if it's cold start, forecast_embeddings has overlap + lead_time,
        # and runs from the h_handoff (which is from BEFORE overlap).
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
            )[:, -self.seq_length :, :]
            for key in y_forecast
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
        # Run the embedding layers.
        hindcast_features = torch.cat(
            [
                t
                for f, t in data['x_d_hindcast'].items()
                if f in self.hindcast_inputs
            ],
            dim=-1,
        )

        statics_embeddings = self.statics_embedding_net(data['x_s'])
        hindcast_embeddings = self.hindcast_embedding_net(hindcast_features)

        hindcast_embeddings = torch.cat(
            [
                hindcast_embeddings,
                statics_embeddings.unsqueeze(1).expand(
                    -1, hindcast_embeddings.size(1), -1
                ),
            ],
            dim=-1,
        )

        # We run the exact same logic up to the final temporal state (Day D)
        forecast_features = torch.cat(
            [
                t
                for f, t in data['x_d_forecast'].items()
                if f in self.forecast_inputs
            ],
            dim=-1,
        )

        forecast_embeddings = self.forecast_embedding_net(forecast_features)
        forecast_embeddings = torch.cat(
            [
                forecast_embeddings,
                statics_embeddings.unsqueeze(1).expand(
                    -1, forecast_embeddings.size(1), -1
                ),
            ],
            dim=-1,
        )

        # Cold start logic internally to propagate up to Day D
        if self.overlap > 0:
            spinup_embeddings = hindcast_embeddings[:, : -self.overlap]
            overlap_embeddings_hindcast = hindcast_embeddings[
                :, -self.overlap :
            ]

            # Forecast embeddings contains overlap+lead. We only want overlap.
            overlap_embeddings_forecast = forecast_embeddings[
                :, : self.overlap, :
            ]
        else:
            spinup_embeddings = hindcast_embeddings
            overlap_embeddings_hindcast = hindcast_embeddings[:, 0:0, :]
            overlap_embeddings_forecast = forecast_embeddings[:, 0:0, :]

        _, (h_hindcast, c_hindcast) = self.hindcast_lstm(spinup_embeddings)

        # We also run hindcast overlap to save its final state
        if overlap_embeddings_hindcast.size(1) > 0:
            _, (h_hind_final, c_hind_final) = self.hindcast_lstm(
                overlap_embeddings_hindcast, (h_hindcast, c_hindcast)
            )
        else:
            h_hind_final, c_hind_final = h_hindcast, c_hindcast

        # Handoff
        x = self.handoff_net(torch.cat([h_hindcast, c_hindcast], -1))
        initial_state = self.handoff_linear(x)
        h_handoff, c_handoff = initial_state.chunk(2, -1)
        h_handoff, c_handoff = (
            h_handoff.contiguous(),
            c_handoff.contiguous(),
        )

        # Run forecast lstm on just the overlap
        if overlap_embeddings_forecast.size(1) > 0:
            _, (h_fore_final, c_fore_final) = self.forecast_lstm(
                overlap_embeddings_forecast, (h_handoff, c_handoff)
            )
        else:
            h_fore_final, c_fore_final = h_handoff, c_handoff

        np.savez_compressed(
            path,
            h_hindcast=h_hind_final.detach().cpu().numpy(),
            c_hindcast=c_hind_final.detach().cpu().numpy(),
            h_forecast=h_fore_final.detach().cpu().numpy(),
            c_forecast=c_fore_final.detach().cpu().numpy(),
        )

    def load_state_from_disk(self, path: str | Path) -> None:
        """Pre-load a hot start state archive from disk into memory.

        Parameters
        ----------
        path : str | Path
            Path to the .npz state file to load.
        """
        self._preloaded_state = dict(np.load(path, allow_pickle=False))
