from typing import Tuple, Dict, Iterable

import dataclasses
import math
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
            hidden_size=512,
            batch_first=True,
        )

        # Forecast LSTM
        self.forecast_lstm = nn.LSTM(
            input_size=self.config_data.embedding_size * 2 + 512,
            hidden_size=512,
            batch_first=True,
        )

        # Head
        self.dropout = nn.Dropout(p=cfg.output_dropout)
        self.head = get_head(self.cfg, n_in=512, n_out=3 * 4, n_hidden=100)

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
        """Append static attributes embedding to another embedding tensor."""
        # Dimension 1 is the time dimension. Duplicate static embeddings in all time series.
        length = embeddings.shape[1]
        static_embeddings_repeated = static_embeddings.unsqueeze(1).repeat(1, length, 1)
        return torch.cat([embeddings, static_embeddings_repeated], dim=-1)

    def _add_nan_padding(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Pad the embedding tensor with nan value to timespan of hindcast and forecast."""
        # Dimension 0 is the batch size. Note the batch size may change during training.
        batch_size = embeddings.shape[0]
        # Dimension 1 is the time dimension. Pad nan to the full sequence length plus lead time.
        nan_padding_length = self.seq_length + self.lead_time - embeddings.shape[1]
        # Dimension 2 is the length of embedding vector.
        embedding_size = embeddings.shape[2]
        nan_padding = torch.full(
            (batch_size, nan_padding_length, embedding_size),
            math.nan,
            device=embeddings.device,
        )
        return torch.cat([embeddings, nan_padding], dim=1)

    def _masked_mean_embedding(
        self, embeddings: Iterable[torch.Tensor]
    ) -> torch.Tensor:
        """Calculate mean between list of tensors, skipping nan values. All
        Tensors are with the same dimensions."""
        merged = torch.cat([e.unsqueeze(-1) for e in embeddings], dim=-1)
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
        forward_data = ForwardData.from_forward_data(data)

        # Static attributes
        static_embeddings = self.static_attributes_fc(forward_data.static_attributes)

        # CPC
        cpc_input_concat = self._append_static_embeddings(
            forward_data.cpc_data, static_embeddings=static_embeddings
        )
        cpc_embeddings = self.cpc_input_fc(cpc_input_concat)
        cpc_embedding_with_nan = self._add_nan_padding(cpc_embeddings)

        # IMERG
        imerg_input_concat = self._append_static_embeddings(
            forward_data.imerg_data, static_embeddings=static_embeddings
        )
        imerg_embeddings = self.imerg_input_fc(imerg_input_concat)
        imerg_embedding_with_nan = self._add_nan_padding(imerg_embeddings)

        # HRES
        hres_input_concat = self._append_static_embeddings(
            forward_data.hres_data, static_embeddings=static_embeddings
        )
        hres_embeddings = self.hres_input_fc(hres_input_concat)

        # GraphCast
        graphcast_input_concat = self._append_static_embeddings(
            forward_data.graphcast_data, static_embeddings=static_embeddings
        )
        graphcast_embeddings = self.graphcast_input_fc(graphcast_input_concat)

        # Hindcast LSTM
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

        # Forecast LSTM
        forecast_mean_embedding = self._masked_mean_embedding(
            [hres_embeddings, graphcast_embeddings]
        )
        forecast_data_concat = self._append_static_embeddings(
            torch.cat([forecast_mean_embedding, hindcast], dim=-1),
            static_embeddings=static_embeddings,
        )
        forecast, _ = self.forecast_lstm(input=forecast_data_concat)

        # Head
        predictions = self.head(self.dropout(forecast))

        return predictions


_STATIC_ATTRIBUTES_NAMES = (
    "area",
    "p_mean",
    "pet_mean_ERA5_LAND",
    "pet_mean_FAO_PM",
    "aridity_ERA5_LAND",
    "aridity_FAO_PM",
    "frac_snow",
    "moisture_index_ERA5_LAND",
    "moisture_index_FAO_PM",
    "seasonality_ERA5_LAND",
    "seasonality_FAO_PM",
    "high_prec_freq",
    "high_prec_dur",
    "low_prec_freq",
    "low_prec_dur",
    "pet_mm_syr",
    "ele_mt_smx",
    "pre_mm_syr",
)

_CPC_ATTRIBUTES_NAMES = ("cpc_precipitation",)

_IMERG_ATTRIBUTES_NAMES = ("imerg_precipitation",)

_HRES_ATTRIBUTES_NAMES = (
    "hres_surface_net_solar_radiation",
    "hres_surface_net_thermal_radiation",
    "hres_surface_pressure",
    "hres_temperature_2m",
    "hres_total_precipitation",
)

_GRAPHCAST_ATTRIBUTES_NAMES = (
    "graphcast_temperature_2m",
    "graphcast_total_precipitation",
    "graphcast_u_component_of_wind_10m",
    "graphcast_v_component_of_wind_10m",
)


@dataclasses.dataclass(frozen=True, kw_only=True)
class ConfigData:
    @classmethod
    def from_config(cls, unused_cfg: Config) -> "ConfigData":
        return ConfigData(
            embedding_size=20,
            static_attributes_names=_STATIC_ATTRIBUTES_NAMES,
            cpc_attributes_names=_CPC_ATTRIBUTES_NAMES,
            imerg_attributes_names=_IMERG_ATTRIBUTES_NAMES,
            hres_attributes_names=_HRES_ATTRIBUTES_NAMES,
            graphcast_attributes_names=_GRAPHCAST_ATTRIBUTES_NAMES,
        )

    embedding_size: int
    static_attributes_names: Tuple[str, ...]
    cpc_attributes_names: Tuple[str, ...]
    imerg_attributes_names: Tuple[str, ...]
    hres_attributes_names: Tuple[str, ...]
    graphcast_attributes_names: Tuple[str, ...]


@dataclasses.dataclass(frozen=True, kw_only=True)
class ForwardData:
    @classmethod
    def from_forward_data(
        cls, data: dict[str, torch.Tensor | dict[str, torch.Tensor]]
    ) -> "ForwardData":
        return ForwardData(
            static_attributes=data["x_s"],
            cpc_data=_concat_tensors_from_dict(
                data["x_d_hindcast"], keys=_CPC_ATTRIBUTES_NAMES
            ),
            imerg_data=_concat_tensors_from_dict(
                data["x_d_hindcast"], keys=_IMERG_ATTRIBUTES_NAMES
            ),
            hres_data=_concat_tensors_from_dict(
                data["x_d_forecast"], keys=_HRES_ATTRIBUTES_NAMES
            ),
            graphcast_data=_concat_tensors_from_dict(
                data["x_d_forecast"], keys=_GRAPHCAST_ATTRIBUTES_NAMES
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
