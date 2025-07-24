from typing import Dict

import torch
import torch.nn as nn

from neuralhydrology.modelzoo.basemodel import BaseModel
from neuralhydrology.modelzoo.fc import FC
from neuralhydrology.modelzoo.head import get_head
from neuralhydrology.modelzoo.inputlayer import InputLayer
from neuralhydrology.utils.config import Config
from neuralhydrology.datautils import mean_embedding_forecast_lstm_datautils


class MeanEmbeddingForecastLSTM(BaseModel):
    """A forecasting model using stacked LSTMs for hindcast and forecast.

    This is a forecasting model that uses two stacked sequential (LSTM) models to handle 
    hindcast vs. forecast. The config parameter ``forecast_overlap`` deterimines the temporal
    overlap of the hindcast and forecast LSTMs.
        
    Outputs from the hindcast LSTM are concatenated to the input sequences to the forecast
    LSTM and shifted in time by the forecast horizon. This causes a lag between the latest
    hindcast data and the newest forecast time point, meaning that forecasts do not get
    information from the most recent hindcast inputs. To solve this, set the
    ``bidirectional_stacked_forecast_lstm`` config parameter to True, so that the
    hindcast LSTM runs bidirectional and therefore all outputs from the hindcast
    LSTM receive information from the most recent hindcast input data. This model supports
    different embedding networks, as defined by ``hindcast_embedding`` and
    ``forecast_embedding`` in the config.

    Parameters
    ----------
    cfg : Config
        The run configuration.
        
    Raises
    ------
    ValueError if `predict_last_n` is longer than the total forecast sequence (overlap + lead time).
    ValueError if `seq_length` is shorter than `forecast_overlap`.
    """
    # Specify submodules of the model that can later be used for finetuning. Names must match class attributes.
    module_parts = ['hindcast_embedding_net', 'forecast_embedding_net', 'forecast_lstm', 'hindcast_lstm', 'head']

    def __init__(self, cfg: Config):
        super(MeanEmbeddingForecastLSTM, self).__init__(cfg=cfg)

        self.config_data = (
            mean_embedding_forecast_lstm_datautils.ConfigData.from_config(cfg)
        )

        self.static_attributes_fc = FC(
            input_size=len(self.config_data.static_attributes),
            hidden_sizes=[100, 100, self.config_data.embedding_size],
            activation=["tanh", "tanh", "linear"],
            dropout=0,
        )

        self.cpc_input_fc = FC(
            input_size=len(self.config_data.cpc_attributes) + self.config_data.embedding_size,
            hidden_sizes=[100, self.config_data.embedding_size],
            activation=["tanh", "linear"],
            dropout=0,
        )


        # Data sizes for expanding features in the forward pass.
        self.seq_length = cfg.seq_length
        self.lead_time = cfg.lead_time
        self.overlap = cfg.forecast_overlap
        # TODO (future) :: Models assume that all lead times are present up to the longest `lead_time`.
        # ForecastBaseDataset does not require this assumption.
        if cfg.predict_last_n > self.lead_time + self.overlap:
            raise ValueError('`predict_last_n` must not be larger than the length of the forecast sequence.')
        if cfg.seq_length < self.overlap:
            raise ValueError('`seq_length` must be larger than `forecast_overlap`.')

        # Input embedding layers.
        self.forecast_embedding_net = InputLayer(cfg=cfg, embedding_type='forecast')
        self.hindcast_embedding_net = InputLayer(cfg=cfg, embedding_type='hindcast')

        # Time series layers.
        self.hindcast_lstm = nn.LSTM(
            input_size=self.hindcast_embedding_net.output_size,
            hidden_size=cfg.hidden_size,
            bidirectional=cfg.bidirectional_stacked_forecast_lstm,
        )

        forecast_input_size = self.forecast_embedding_net.output_size + self.hindcast_lstm.hidden_size
        if self.cfg.bidirectional_stacked_forecast_lstm:
            forecast_input_size += self.hindcast_lstm.hidden_size
        self.forecast_lstm = nn.LSTM(
            input_size=forecast_input_size,
            hidden_size=cfg.hidden_size,
            batch_first=True
        )

        self.dropout = nn.Dropout(p=cfg.output_dropout)
        self.head = get_head(cfg=cfg, n_in=cfg.hidden_size, n_out=self.output_size)

        self._reset_parameters()

    def _reset_parameters(self):
        """Special initialization of certain model weights."""
        if self.cfg.initial_forget_bias is not None:
            self.hindcast_lstm.bias_hh_l0.data[self.cfg.hidden_size:2 * self.cfg.hidden_size] = self.cfg.initial_forget_bias
            self.forecast_lstm.bias_hh_l0.data[self.cfg.hidden_size:2 * self.cfg.hidden_size] = self.cfg.initial_forget_bias


    def forward(self, data: dict[str, torch.Tensor | dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Perform a forward pass on the StackedForecastLSTM model.

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
        forward_data = (
            mean_embedding_forecast_lstm_datautils.ForwardData.from_forward_data(data)
        )
        x_s_fc = self.static_attributes_fc(forward_data.static_attributes)
        static_embeddings_repeated = x_s_fc.unsqueeze(1).repeat(1, self.seq_length, 1)

        cpc_input_concat = torch.cat([forward_data.cpc_data, static_embeddings_repeated], dim=-1)
        cpc_embeddings = self.cpc_input_fc(cpc_input_concat)

        # CPC embeddings (cpc, static)
        # IMERG embeddings (imerg, static)
        # HRES embeddings (hres, static)
        # GraphCast embeddings (graphcast, static)
        # Masked mean hindcast (cpc, imerg, hres, graphcast)
        # Masked mean forcast (hres, graphcast)
        # LSTM hindcast (masked mean hindcast, static)
        # LSTM forecast (lstm hindcast, masked mean forcast, static)
        # Head (lstm forecast)

        # Run the embedding layers.
        hindcast_embeddings = self.hindcast_embedding_net(data)
        forecast_embeddings = self.forecast_embedding_net(data)

        # Run hindcast LSTM.
        hindcast, _ = self.hindcast_lstm(input=hindcast_embeddings)

        # Run forecast LSTM.
        forecast_inputs = torch.cat((forecast_embeddings, hindcast[-self.overlap-self.lead_time:, ...]), dim=-1)
        forecast, _ = self.forecast_lstm(forecast_inputs)

        # Run head.
        return self.head(self.dropout(forecast.transpose(0, 1)))
