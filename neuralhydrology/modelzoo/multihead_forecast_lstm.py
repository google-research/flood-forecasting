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


from typing import Dict

import torch
import torch.nn as nn

from neuralhydrology.modelzoo.basemodel import BaseModel
from neuralhydrology.modelzoo.fc import FC
from neuralhydrology.modelzoo.head import get_head
from neuralhydrology.modelzoo.inputlayer import InputLayer
from neuralhydrology.utils.config import Config


class MultiHeadForecastLSTM(BaseModel):
    """A forecasting model that does not roll out over the forecast horizon.
    
    This is a forecasting model that runs a sequential (LSTM) model up to the forecast issue time, 
    and then directly predicts a sequence of forecast timesteps without using a recurrent rollout.
    Prediction is done with a custom ``FC`` (fully connected) layer, which can include depth.
    Do not use this model with ``forecast_overlap`` > 0.

    Parameters
    ----------
    cfg : Config
        The run configuration.
    
    Raises
    ------
    ValueError if forecast_overlap > 0.
    ValueError if a forecast_network is not specified.
    """
    # specify submodules of the model that can later be used for finetuning. Names must match class attributes
    module_parts = [
        'forecast_mebedding_net',
        'hindcast_embedding_net',
        'hindcast_lstm',
        'forecast_network',
        'hindcast_head',
        'forecast_head'
    ]

    def __init__(self, cfg: Config):
        super(MultiHeadForecastLSTM, self).__init__(cfg=cfg)

        if cfg.forecast_overlap:
            raise ValueError(
                'Forecast overlap cannot be set for a multi-head forecasting model. '
                'Please set to None or remove from config file.'
            )

        # Data sizes for expanding features in the forward pass.
        self.seq_length = cfg.seq_length
        # TODO (future) :: Models assume that all lead times are present up to the longest `lead_time`.
        # ForecastBaseDataset does not require this assumption.
        self.lead_time = cfg.lead_time
        
        # Input embedding layers.
        self.forecast_embedding_net = InputLayer(cfg=cfg, embedding_type='forecast')
        self.hindcast_embedding_net = InputLayer(cfg=cfg, embedding_type='hindcast')
        
        # Time series layers.
        self.hindcast_lstm = nn.LSTM(
            input_size=self.hindcast_embedding_net.output_size,
            hidden_size=cfg.hidden_size,
        )

        self.dropout = nn.Dropout(p=cfg.output_dropout)

        if not cfg.forecast_network:
            raise ValueError('The multihead forecast model requires a forecast network specified in the config file.')

        input_size = self.forecast_embedding_net.output_size*cfg.lead_time + cfg.hidden_size
        forecast_network_output_size = cfg.forecast_network['hiddens'][-1] * cfg.lead_time
        self.forecast_network = FC(
            input_size=input_size,
            hidden_sizes=cfg.forecast_network['hiddens'][:-1] + [forecast_network_output_size],
            activation=cfg.forecast_network['activation'],
            dropout=cfg.forecast_network['dropout']
        )

        self.hindcast_head = get_head(cfg=cfg, n_in=cfg.hidden_size, n_out=self.output_size)
        self.forecast_head = get_head(cfg=cfg, n_in=cfg.forecast_network['hiddens'][-1], n_out=self.output_size)

        self._reset_parameters()

    def _reset_parameters(self):
        """Special initialization of certain model weights."""
        if self.cfg.initial_forget_bias is not None:
            self.hindcast_lstm.bias_hh_l0.data[self.cfg.hidden_size:2 * self.cfg.hidden_size] = self.cfg.initial_forget_bias

    def forward(self, data: dict[str, torch.Tensor | dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Perform a forward pass on the MultiheadForecastLSTM model.
        
        Parameters
        ----------
        data : dict[str, torch.Tensor | dict[str, torch.Tensor]]
            Dictionary, containing input features as key-value pairs.
        
        Returns
        -------
        Dict[str, torch.Tensor]
            Model outputs and intermediate states as a dictionary.
                - y_hat: Predictions over the sequence from the head layer.
        """
        # Run the embedding layers.
        hindcast_embeddings = self.hindcast_embedding_net(data)
        forecast_embeddings = self.forecast_embedding_net(data)

        # Run hindcast part of the lstm.
        lstm_output_hindcast, (h_hindcast, _) = self.hindcast_lstm(input=hindcast_embeddings)
        output_hindcast = self.hindcast_head(self.dropout(lstm_output_hindcast.transpose(0, 1)))['y_hat']

        # Reshape to [batch_size, seq, n_hiddens].
        h_hindcast = h_hindcast.transpose(0, 1).squeeze(dim=1)
        forecast_embeddings = forecast_embeddings.transpose(0, 1)
        batch_size = forecast_embeddings.shape[0]
        forecast_embeddings = forecast_embeddings.reshape(batch_size, -1)

        # Run forecast heads.
        x = torch.cat([h_hindcast, forecast_embeddings], dim=-1)
        x = self.forecast_network(x)
        x = x.view(batch_size, self.lead_time, -1)
        output_forecast = self.forecast_head(self.dropout(x))['y_hat']
        y_hat = torch.cat([output_hindcast, output_forecast], dim=1)[:, -self.seq_length:, ...]

        return {'y_hat': y_hat}
