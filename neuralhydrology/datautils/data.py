import dataclasses
from typing import Tuple

from neuralhydrology.utils.config import Config

_STATIC_ATTRIBUTES = (
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

_CPC_ATTRIBUTES = ("cpc_precipitation",)


@dataclasses.dataclass(frozen=True, kw_only=True)
class ConfigData:
    @classmethod
    def from_config(cls, unused_cfg: Config) -> "ConfigData":
        return ConfigData(
            static_attributes=_STATIC_ATTRIBUTES, cpc_attributes=_CPC_ATTRIBUTES
        )

    static_attributes: Tuple[str, ...]
    cpc_attributes: Tuple[str, ...]
