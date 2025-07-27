from typing import Dict, Iterable

import math
import torch
import torch.nn as nn

from neuralhydrology.modelzoo.basemodel import BaseModel
from neuralhydrology.modelzoo.fc import FC
from neuralhydrology.modelzoo.head import CMAL
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
    module_parts = [
        "hindcast_embedding_net",
        "forecast_embedding_net",
        "forecast_lstm",
        "hindcast_lstm",
        "head",
    ]

    def __init__(self, cfg: Config):
        super(MeanEmbeddingForecastLSTM, self).__init__(cfg=cfg)

        # Data sizes for expanding features in the forward pass.
        self.seq_length = cfg.seq_length
        self.lead_time = cfg.lead_time
        self.overlap = cfg.forecast_overlap
        # TODO (future) :: Models assume that all lead times are present up to the longest `lead_time`.
        # ForecastBaseDataset does not require this assumption.
        if cfg.predict_last_n > self.lead_time + self.overlap:
            raise ValueError(
                "`predict_last_n` must not be larger than the length of the forecast sequence."
            )
        if cfg.seq_length < self.overlap:
            raise ValueError("`seq_length` must be larger than `forecast_overlap`.")

        self.config_data = (
            mean_embedding_forecast_lstm_datautils.ConfigData.from_config(cfg)
        )

        self.static_attributes_fc = FC(
            input_size=len(self.config_data.static_attributes_names),
            hidden_sizes=[100, 100, self.config_data.embedding_size],
            activation=["tanh", "tanh", "linear"],
            dropout=0,
        )

        self.cpc_input_fc = FC(
            input_size=(
                len(self.config_data.cpc_attributes_names)
                + self.config_data.embedding_size
            ),
            hidden_sizes=[100, self.config_data.embedding_size],
            activation=["tanh", "linear"],
            dropout=0,
        )

        self.imerg_input_fc = FC(
            input_size=(
                len(self.config_data.imerg_attributes_names)
                + self.config_data.embedding_size
            ),
            hidden_sizes=[100, self.config_data.embedding_size],
            activation=["tanh", "linear"],
            dropout=0,
        )

        self.hres_input_fc = FC(
            input_size=(
                len(self.config_data.hres_attributes_names)
                + self.config_data.embedding_size
            ),
            hidden_sizes=[20, 20, 20, self.config_data.embedding_size],
            activation=["tanh", "tanh", "tanh", "linear"],
            dropout=0,
        )

        self.graphcast_input_fc = FC(
            input_size=(
                len(self.config_data.graphcast_attributes_names)
                + self.config_data.embedding_size
            ),
            hidden_sizes=[20, 20, 20, self.config_data.embedding_size],
            activation=["tanh", "tanh", "tanh", "linear"],
            dropout=0,
        )

        self.hindcast_lstm = nn.LSTM(
            input_size=self.config_data.embedding_size * 2,
            hidden_size=512,
            batch_first=True,
        )

        self.forecast_lstm = nn.LSTM(
            input_size=self.config_data.embedding_size * 2 + 512,
            hidden_size=512,
            batch_first=True,
        )

        self.dropout = nn.Dropout(p=cfg.output_dropout)

        self.head = CMAL(
            n_in=512, n_out=3 * 4, n_hidden=100
        )  # Uses `sample_umal` via cfg.

        self._reset_parameters()

    def _reset_parameters(self):
        """Special initialization of certain model weights."""
        if self.cfg.initial_forget_bias is not None:
            self.hindcast_lstm.bias_hh_l0.data[
                self.cfg.hidden_size : 2 * self.cfg.hidden_size
            ] = self.cfg.initial_forget_bias
            self.forecast_lstm.bias_hh_l0.data[
                self.cfg.hidden_size : 2 * self.cfg.hidden_size
            ] = self.cfg.initial_forget_bias

    def _append_static_embeddings(
        self, embeddings: torch.Tensor, *, static_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """Pad the embedding tensor with the embedding of static attributes.

        Parameters
        ----------
        embeddings : torch.Tensor
            Embedding tensor to append the static embedding tensor.

        Returns
        -------
        torch.Tensor
            A new tensor which is a concatentation of both tensors.

        """
        length = embeddings.shape[1]
        static_embeddings_repeated = static_embeddings.unsqueeze(1).repeat(1, length, 1)
        return torch.cat([embeddings, static_embeddings_repeated], dim=-1)

    def _add_nan_padding(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Pad the embedding tensor with nan value.

        Parameters
        ----------
        embeddings : torch.Tensor
            Embedding tensor of all hindcast calculations.

        Returns
        -------
        torch.Tensor
            A new tensor padded with nan in the end, spanning the full sequence length and the lead time.

        """
        # Dimension 0 is the batch size. Note the batch size may change during training.
        batch_size = embeddings.shape[0]
        # Dimension 1 is the time dimension.
        nan_padding_length = self.seq_length + self.lead_time - embeddings.shape[1]
        # Dimension 2 is the length of embedding vector.
        embedding_size = embeddings.shape[2]
        nan_padding = torch.full(
            (batch_size, nan_padding_length, embedding_size), math.nan
        )
        return torch.cat([embeddings, nan_padding], dim=1)

    def _masked_mean_embedding(self, data: Iterable[torch.Tensor]) -> torch.Tensor:
        merged = torch.cat([e.unsqueeze(-1) for e in data], dim=-1)
        return torch.nanmean(merged, dim=-1)

    def forward(
        self, data: dict[str, torch.Tensor | dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
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
        static_embeddings = self.static_attributes_fc(forward_data.static_attributes)

        cpc_input_concat = self._append_static_embeddings(
            forward_data.cpc_data, static_embeddings=static_embeddings
        )
        cpc_embeddings = self.cpc_input_fc(cpc_input_concat)
        cpc_embedding_with_nan = self._add_nan_padding(cpc_embeddings)

        imerg_input_concat = self._append_static_embeddings(
            forward_data.imerg_data, static_embeddings=static_embeddings
        )
        imerg_embeddings = self.imerg_input_fc(imerg_input_concat)
        imerg_embedding_with_nan = self._add_nan_padding(imerg_embeddings)

        hres_input_concat = self._append_static_embeddings(
            forward_data.hres_data, static_embeddings=static_embeddings
        )
        hres_embeddings = self.hres_input_fc(hres_input_concat)

        graphcast_input_concat = self._append_static_embeddings(
            forward_data.graphcast_data, static_embeddings=static_embeddings
        )
        graphcast_embeddings = self.graphcast_input_fc(graphcast_input_concat)

        hindcast_mean_embedding = self._masked_mean_embedding(
            [
                cpc_embedding_with_nan,
                imerg_embedding_with_nan,
                hres_embeddings,
                graphcast_embeddings,
            ]
        )
        hindcast_data_concat = self._append_static_embeddings(
            hindcast_mean_embedding, static_embeddings=static_embeddings
        )
        hindcast, _ = self.hindcast_lstm(input=hindcast_data_concat)

        forecast_mean_embedding = self._masked_mean_embedding(
            [hres_embeddings, graphcast_embeddings]
        )
        forecast_data_concat = self._append_static_embeddings(
            torch.cat([forecast_mean_embedding, hindcast], dim=-1),
            static_embeddings=static_embeddings,
        )
        forecast, _ = self.forecast_lstm(input=forecast_data_concat)

        predictions = self.head(self.dropout(forecast))

        # create deterministic "best guess" prediction:
        # - weighted avg of location parameters (mu),
        # - using the model's learnt mixture weights (pi).
        # Mu's values are the median for each distribution.
        # Pi's values are likewise for the model's confidence for each dist and should sum to 1.
        # So, multiply those element-wise to scale each location mu by factor pi.
        # Sum on the last dim because that's the distribution feature values.
        # For example, suppose n_distributions=3, mu=[10.0, 15.0, 12.0] and [0.1, 0.7, 0.2]. It means the model is most confident in the second dist (most importance in pi). So 0.1*10.0+0.7*15.0+0.2*12.0=1.0+10.5+2.4=13.9.
        y_hat = torch.sum(predictions["pi"] * predictions["mu"], dim=-1, keepdim=True)

        return {"y_hat": y_hat} | predictions
