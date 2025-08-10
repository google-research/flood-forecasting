from typing import Tuple, Dict, Iterable

import dataclasses
import numpy as np
import torch
import torch.nn as nn

from neuralhydrology.modelzoo.basemodel import BaseModel
from neuralhydrology.modelzoo.fc import FC
from neuralhydrology.modelzoo.head import get_head
from neuralhydrology.utils.config import Config


class MeanEmbeddingForecastLSTM(BaseModel):
    """A forecasting model using mean embedding and LSTMs for hindcast and forecast.

    Parameters
    ----------
    cfg : Config
        The run configuration.
    """

    # Specify submodules of the model that can later be used for finetuning. Names must match class attributes.
    module_parts = [
        "static_attributes_fc",
        "cpc_input_fc",
        "imerg_input_fc",
        "hres_input_fc",
        "graphcast_input_fc",
        "hindcast_lstm",
        "forecast_lstm",
        "head",
    ]

    def __init__(self, cfg: Config):
        super(MeanEmbeddingForecastLSTM, self).__init__(cfg=cfg)

        self.seq_length = cfg.seq_length
        self.lead_time = cfg.lead_time

        self.config_data = ConfigData.from_config(cfg)

        # Static attributes
        self.static_attributes_fc = FC(
            input_size=len(self.config_data.static_attributes_names),
            hidden_sizes=[100, 100, self.config_data.embedding_size],
            activation=["tanh", "tanh", "linear"],
            dropout=0,
        )

        # CPC
        self.cpc_input_fc = FC(
            input_size=(
                len(self.config_data.cpc_attributes_names)
                + self.config_data.embedding_size
            ),
            hidden_sizes=[100, self.config_data.embedding_size],
            activation=["tanh", "linear"],
            dropout=0,
        )

        # IMERG
        self.imerg_input_fc = FC(
            input_size=(
                len(self.config_data.imerg_attributes_names)
                + self.config_data.embedding_size
            ),
            hidden_sizes=[100, self.config_data.embedding_size],
            activation=["tanh", "linear"],
            dropout=0,
        )

        # HRES
        self.hres_input_fc = FC(
            input_size=(
                len(self.config_data.hres_attributes_names)
                + self.config_data.embedding_size
            ),
            hidden_sizes=[20, 20, 20, self.config_data.embedding_size],
            activation=["tanh", "tanh", "tanh", "linear"],
            dropout=0,
        )

        # GraphCast
        self.graphcast_input_fc = FC(
            input_size=(
                len(self.config_data.graphcast_attributes_names)
                + self.config_data.embedding_size
            ),
            hidden_sizes=[20, 20, 20, self.config_data.embedding_size],
            activation=["tanh", "tanh", "tanh", "linear"],
            dropout=0,
        )

        # Hindcast LSTM
        self.hindcast_lstm = nn.LSTM(
            input_size=self.config_data.embedding_size * 2,
            hidden_size=self.config_data.hidden_size,
            batch_first=True,
        )

        # Forecast LSTM
        self.forecast_lstm = nn.LSTM(
            input_size=self.config_data.embedding_size * 2
            + self.config_data.hidden_size,
            hidden_size=self.config_data.hidden_size,
            batch_first=True,
        )

        # Head
        self.dropout = nn.Dropout(p=cfg.output_dropout)
        self.head = get_head(
            self.cfg, n_in=self.config_data.hidden_size, n_out=3 * 4, n_hidden=100
        )

        self._reset_parameters()

    def _reset_parameters(self):
        """Special initialization of certain model weights."""
        if self.cfg.initial_forget_bias is not None:
            self.hindcast_lstm.bias_hh_l0.data[
                self.config_data.hidden_size : 2 * self.config_data.hidden_size
            ] = self.cfg.initial_forget_bias
            self.forecast_lstm.bias_hh_l0.data[
                self.config_data.hidden_size : 2 * self.config_data.hidden_size
            ] = self.cfg.initial_forget_bias

    def _append_static_attributes(
        self, embedding: torch.Tensor, *, static: torch.Tensor
    ) -> torch.Tensor:
        """Append static attributes embedding to another embedding tensor."""
        # Dimension 1 is the time dimension. Duplicate static embedding in all time series.
        length = embedding.shape[1]
        static_repeated = static.unsqueeze(1).repeat(1, length, 1)
        return torch.cat([embedding, static_repeated], dim=-1)

    def _add_nan_padding(self, embedding: torch.Tensor) -> torch.Tensor:
        """Pad the embedding tensor with nan value to timespan of hindcast and forecast."""
        # Dimension 0 is the batch size. Note the batch size may change during training.
        batch_size = embedding.shape[0]
        # Dimension 1 is the time dimension. Pad nan to the full sequence length plus lead time.
        nan_padding_length = self.seq_length + self.lead_time - embedding.shape[1]
        # Dimension 2 is the length of embedding vector.
        embedding_size = embedding.shape[2]
        nan_padding = torch.full(
            (batch_size, nan_padding_length, embedding_size),
            np.nan,
            device=embedding.device,
        )
        return torch.cat([embedding, nan_padding], dim=1)

    def _masked_mean(self, tensors: Iterable[torch.Tensor]) -> torch.Tensor:
        """Calculate mean between list of tensors, skipping nan values. Calculates mean of the last dimension.
        All tensors have same dimensions."""
        merged = torch.cat([e.unsqueeze(-1) for e in tensors], dim=-1)
        return torch.nanmean(merged, dim=-1)

    def forward(
        self, data: dict[str, torch.Tensor | dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """Perform a forward pass on the MeanEmbeddingForecastLSTM model.

        Parameters
        ----------
        data : dict[str, torch.Tensor | dict[str, torch.Tensor]]
            Dictionary, containing input features as key-value pairs.

        Returns
        -------
        Dict[str, torch.Tensor]
            Model outputs and intermediate states as a dictionary from CMAL head.
        """
        forward_data = ForwardData.from_forward_data(data, self.config_data)

        static = self._calc_static_attributes(forward_data)
        cpc = self._calc_cpc(forward_data, static)
        imerg = self._calc_imerg(forward_data, static)
        hres = self._calc_hres(forward_data, static)
        graphcast = self._calc_graphcast(forward_data, static)

        hindcast = self._calc_hindcast(static, cpc, imerg, hres, graphcast)
        forecast = self._calc_forecast(hindcast, static, hres, graphcast)

        return self._calc_head(forecast)

    def _calc_static_attributes(self, forward_data: "ForwardData") -> torch.Tensor:
        return self.static_attributes_fc(forward_data.static_attributes)

    def _calc_cpc(
        self, forward_data: "ForwardData", static: torch.Tensor
    ) -> torch.Tensor:
        cpc_input_concat = self._append_static_attributes(
            forward_data.cpc_data, static=static
        )
        cpc = self.cpc_input_fc(cpc_input_concat)
        cpc_with_nan = self._add_nan_padding(cpc)
        return cpc_with_nan

    def _calc_imerg(
        self, forward_data: "ForwardData", static: torch.Tensor
    ) -> torch.Tensor:
        imerg_input_concat = self._append_static_attributes(
            forward_data.imerg_data, static=static
        )
        imerg = self.imerg_input_fc(imerg_input_concat)
        imerg_with_nan = self._add_nan_padding(imerg)
        return imerg_with_nan

    def _calc_hres(
        self, forward_data: "ForwardData", static: torch.Tensor
    ) -> torch.Tensor:
        hres_input_concat = self._append_static_attributes(
            forward_data.hres_data, static=static
        )
        return self.hres_input_fc(hres_input_concat)

    def _calc_graphcast(
        self, forward_data: "ForwardData", static: torch.Tensor
    ) -> torch.Tensor:
        graphcast_input_concat = self._append_static_attributes(
            forward_data.graphcast_data, static=static
        )
        return self.graphcast_input_fc(graphcast_input_concat)

    def _calc_hindcast(
        self,
        static: torch.Tensor,
        cpc_with_nan: torch.Tensor,
        imerg_with_nan: torch.Tensor,
        hres: torch.Tensor,
        graphcast: torch.Tensor,
    ) -> torch.Tensor:
        hindcast_mean = self._masked_mean(
            [cpc_with_nan, imerg_with_nan, hres, graphcast]
        )
        hindcast_data_concat = self._append_static_attributes(
            hindcast_mean, static=static
        )
        hindcast, _ = self.hindcast_lstm(input=hindcast_data_concat)
        return hindcast

    def _calc_forecast(
        self,
        hindcast: torch.Tensor,
        static: torch.Tensor,
        hres: torch.Tensor,
        graphcast: torch.Tensor,
    ) -> torch.Tensor:
        forecast_mean = self._masked_mean([hres, graphcast])
        forecast_data_concat = self._append_static_attributes(
            torch.cat([forecast_mean, hindcast], dim=-1),
            static=static,
        )
        forecast, _ = self.forecast_lstm(input=forecast_data_concat)
        return forecast

    def _calc_head(self, forecast: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.head(self.dropout(forecast))


@dataclasses.dataclass(frozen=True, kw_only=True)
class ConfigData:
    @classmethod
    def from_config(cls, cfg: Config) -> "ConfigData":
        return ConfigData(
            hidden_size=cfg.hidden_size,
            embedding_size=20,
            static_attributes_names=tuple(cfg.static_attributes),
            cpc_attributes_names=_filter_by_prefix(cfg.hindcast_inputs, "cpc_"),
            imerg_attributes_names=_filter_by_prefix(cfg.hindcast_inputs, "imerg_"),
            hres_attributes_names=_filter_by_prefix(cfg.forecast_inputs, "hres_"),
            graphcast_attributes_names=_filter_by_prefix(
                cfg.forecast_inputs, "graphcast_"
            ),
        )

    hidden_size: int
    embedding_size: int
    static_attributes_names: Tuple[str, ...]
    cpc_attributes_names: Tuple[str, ...]
    imerg_attributes_names: Tuple[str, ...]
    hres_attributes_names: Tuple[str, ...]
    graphcast_attributes_names: Tuple[str, ...]


def _filter_by_prefix(names: list[str], prefix: str) -> Tuple[str, ...]:
    return tuple(s for s in names if s.startswith(prefix))


@dataclasses.dataclass(frozen=True, kw_only=True)
class ForwardData:
    @classmethod
    def from_forward_data(
        cls,
        data: dict[str, torch.Tensor | dict[str, torch.Tensor]],
        config_data: ConfigData,
    ) -> "ForwardData":
        return ForwardData(
            static_attributes=data["x_s"],
            cpc_data=_concat_tensors_from_dict(
                data["x_d_hindcast"], keys=config_data.cpc_attributes_names
            ),
            imerg_data=_concat_tensors_from_dict(
                data["x_d_hindcast"], keys=config_data.imerg_attributes_names
            ),
            hres_data=_concat_tensors_from_dict(
                data["x_d_forecast"], keys=config_data.hres_attributes_names
            ),
            graphcast_data=_concat_tensors_from_dict(
                data["x_d_forecast"], keys=config_data.graphcast_attributes_names
            ),
        )

    static_attributes: torch.Tensor
    cpc_data: torch.Tensor
    imerg_data: torch.Tensor
    hres_data: torch.Tensor
    graphcast_data: torch.Tensor


def _concat_tensors_from_dict(
    data: dict[str, torch.Tensor], *, keys: Iterable[str]
) -> torch.Tensor:
    return torch.cat([data[e] for e in keys], dim=-1)
