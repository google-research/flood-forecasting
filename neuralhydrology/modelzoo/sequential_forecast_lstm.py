from typing import Dict

import torch
import torch.nn as nn

from neuralhydrology.modelzoo.basemodel import BaseModel
from neuralhydrology.modelzoo.fc import FC
from neuralhydrology.modelzoo.head import get_head
from neuralhydrology.modelzoo.inputlayer import InputLayer
from neuralhydrology.utils.config import Config


class SequentialForecastLSTM(BaseModel):
    """A forecasting model that uses a single LSTM sequence with multiple embedding layers.

    This is a forecasting model that uses a single sequential (LSTM) model that rolls 
    out through both the hindcast and forecast sequences. The difference between this
    and a standard ``CudaLSTM`` is (1) this model uses both hindcast and forecast
    input features, and (2) it uses a separate embedding network for the hindcast
    period and the forecast period, as defined by ``hindcast_embedding`` and
    ``forecast_embedding`` in the config. Do not use this model with ``forecast_overlap`` > 0.

    Parameters
    ----------
    cfg : Config
        The run configuration.

    Raises
    ------
    ValueError if forecast_overlap > 0
    ValueError if forecast and hindcast embedding nets have different output sizes.
    """
    # specify submodules of the model that can later be used for finetuning. Names must match class attributes
    module_parts = ['hindcast_embedding_net', 'forecast_embedding_net', 'lstm', 'head']

    def __init__(self, cfg: Config):
        super(SequentialForecastLSTM, self).__init__(cfg=cfg)

        if cfg.forecast_overlap:
            raise ValueError(
                'Forecast overlap cannot be set for a sequential LSTM forecast model. '
                'Please set to None or remove from config file.'
            )

        # Data sizes for expanding features in the forward pass.
        self.seq_length = cfg.seq_length
        # TODO (future) :: Models assume that all lead times are present up to the longest `lead_time`.
        # ForecastBaseDataset does not require this assumption.
        
        # Input embedding layers.
        self.forecast_embedding_net = InputLayer(cfg=cfg, embedding_type='forecast')
        self.hindcast_embedding_net = InputLayer(cfg=cfg, embedding_type='hindcast')

        if self.forecast_embedding_net.output_size != self.hindcast_embedding_net.output_size:
            raise ValueError('Forecast and hindcast embedding nets must have the same output size when using a sequential forecast LSTM.')

        self.lstm = nn.LSTM(
            input_size=self.forecast_embedding_net.output_size,
            hidden_size=cfg.hidden_size,
        )

        self.dropout = nn.Dropout(p=cfg.output_dropout)
        self.head = get_head(cfg=cfg, n_in=cfg.hidden_size, n_out=self.output_size)

        self._reset_parameters()

    def _reset_parameters(self):
        """Special initialization of certain model weights."""
        if self.cfg.initial_forget_bias is not None:
            self.lstm.bias_hh_l0.data[self.cfg.hidden_size:2 * self.cfg.hidden_size] = self.cfg.initial_forget_bias

    def forward(self, data: dict[str, torch.Tensor | dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Perform a forward pass on the SequentialForecastLSTM model.

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
        embeddings = torch.cat([hindcast_embeddings, forecast_embeddings], dim=0)

        # Run LSTM.
        lstm_output, _ = self.lstm(input=embeddings)
        y_hat = self.head(self.dropout(lstm_output.transpose(0, 1)))['y_hat'][: ,-self.seq_length:, ...]

        # Run head.
        return {'y_hat': y_hat}
