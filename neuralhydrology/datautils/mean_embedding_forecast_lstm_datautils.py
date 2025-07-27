import dataclasses
from typing import Tuple

import torch

from neuralhydrology.utils.config import Config

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


@dataclasses.dataclass(frozen=True, kw_only=True)
class ConfigData:
    @classmethod
    def from_config(cls, unused_cfg: Config) -> "ConfigData":
        return ConfigData(
            embedding_size=20,
            static_attributes_names=_STATIC_ATTRIBUTES_NAMES,
            cpc_attributes_names=_CPC_ATTRIBUTES_NAMES,
            imerg_attributes_names=_IMERG_ATTRIBUTES_NAMES,
        )

    embedding_size: int
    static_attributes_names: Tuple[str, ...]
    cpc_attributes_names: Tuple[str, ...]
    imerg_attributes_names: Tuple[str, ...]


@dataclasses.dataclass(frozen=True, kw_only=True)
class ForwardData:
    @classmethod
    def from_forward_data(
        cls, data: dict[str, torch.Tensor | dict[str, torch.Tensor]]
    ) -> "ForwardData":
        return ForwardData(
            static_attributes=data["x_s"],
            cpc_data=data["x_d_hindcast"]["cpc_precipitation"],
            imerg_data=data["x_d_hindcast"]["imerg_precipitation"],
            forecast_inputs=_gather_forecast_inputs(data),
        )

    static_attributes: torch.Tensor
    cpc_data: torch.Tensor
    imerg_data: torch.Tensor
    forecast_inputs: torch.Tensor


def _gather_forecast_inputs(
    data: dict[str, torch.Tensor | dict[str, torch.Tensor]],
) -> torch.Tensor:
    forecast_input_tensors = [
        data["x_d_forecast"]["graphcast_temperature_2m"],
        data["x_d_forecast"]["graphcast_total_precipitation"],
        data["x_d_forecast"]["graphcast_u_component_of_wind_10m"],
        data["x_d_forecast"]["graphcast_v_component_of_wind_10m"],
    ]
    return torch.cat(forecast_input_tensors, dim=-1)
