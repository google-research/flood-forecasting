import os
from pathlib import Path
from typing import Dict, Optional, Iterator, Hashable

import pandas as pd
import xarray as xr

SCALER_FILE_NAME = 'scaler.nc'


class Scaler():
    """Scaler for a dataset that contains multiple features.
    
    Parameters
    ----------
    scaler_dir : pathlib.Path
        Directory for loading a pre-calculated scaler or saving this scaler if it is calculated.
    calculate_scaler : bool
        Flag to indicate if the scaler should be computed (the alternative is to load an existing scaler file).
    custom_normalization : Dict[str, Dict[str, float]]
        Feature-specific scaling instructions as a mapping from feature name to centering and/or scaling type.
        See docs for a list of accepted types and their meaning.
    dataset : Optional[xr.Dataset]
        Dataset to use for calculating a new scaler. Cannot be supplied if `calculate_scaler` is False.
        
    Raises
    -------
    ValueError for incompatible loading/calculating instructions.
    """

    def __init__(
        self,
        scaler_dir: Path,
        calculate_scaler,
        custom_normalization: Dict[str, Dict[str, float]] = {},
        dataset: Optional[xr.Dataset] = None,
    ):
        # Consistency check.
        if not calculate_scaler and dataset is not None:
            raise ValueError('Do not pass a dataset if you are loading a pre-calculated scaler.')

        # Load or calculate scaling parameters.
        self.scaler = None
        self.scaler_dir = scaler_dir
        if not calculate_scaler:
            self.load()
        else:
            self._custom_normalization = custom_normalization
            if dataset is not None:
                self.calculate(dataset)

    def load(self):
        scaler_file = self.scaler_dir / SCALER_FILE_NAME
        if os.path.exists(scaler_file):
            with open(scaler_file, 'rb') as f:
                self.scaler = xr.load_dataset(f)
        else:
            raise ValueError("Old scaler files are unsupported")
        self._check_zero_scale()

    def save(self):
        if self.scaler is None:
            raise ValueError('You are trying to save a scaler that has not been computed.')
        os.makedirs(self.scaler_dir, exist_ok=True)
        scaler_file = self.scaler_dir / SCALER_FILE_NAME
        with open(scaler_file, 'wb') as f:
            self.scaler.to_netcdf(f)

    def calculate(
        self,
        dataset: xr.Dataset,
    ):
        # Option for custom scaling for each feature.
        centering_types = {feature: 'mean' for feature in dataset.data_vars}
        scaling_types = {feature: 'std' for feature in dataset.data_vars}
        for feature, norm in self._custom_normalization.items():
            if "centering" in norm:
                centering_types[feature] = norm["centering"]
            if "scaling" in norm:
                scaling_types[feature] = norm["scaling"]

        # Calculate all statistics
        stats = {
            "mean": dataset.mean(skipna=True),
            "std": dataset.std(skipna=True),
            "median": dataset.quantile(q=0.5, skipna=True),
            "min": dataset.min(skipna=True),
            "max": dataset.max(skipna=True),
        }
        stats["minmax"] = stats["max"] - stats["min"]

        # Select the appropriate center and scale statistic for each feature.
        def calc_types(
            types: dict[Hashable, str], fallback_stat: str, none_value: float
        ) -> Iterator[xr.DataArray]:
            for feature in dataset.data_vars:
                a_type = types.get(feature, fallback_stat).lower()
                if a_type == "none":
                    yield xr.DataArray(none_value, name=feature)
                else:
                    yield stats[a_type][feature]

        center = xr.merge(calc_types(centering_types, "mean", 0.0))
        scale = xr.merge(calc_types(scaling_types, "std", 1.0))

        # Combine parameters into a single xarray.Dataset with a 'parameter' coordinate.
        scaler = xr.concat(
            [center, scale, stats["mean"], stats["std"]],
            dim=pd.Index(["center", "scale", "mean", "std"], name="parameter"),
        )

        # Expand the scaler dataset to include 'obs' and 'sim' versions of all variables.
        obs_scaler = scaler.rename({var: f"{var}_obs" for var in scaler.data_vars})
        sim_scaler = scaler.rename({var: f"{var}_sim" for var in scaler.data_vars})
        scaler = xr.merge([scaler, obs_scaler, sim_scaler])

        # Handle cases where part of the scaler is already calculated. Simply add new features.
        if self.scaler is not None:
            self.scaler = xr.merge([self.scaler, scaler])
        else:
            self.scaler = scaler

        self.scaler = self.scaler.compute() # Compute prior to the checks.

        # Ensure that there are no zero-valued scale parameters, as this will cause NaN's.
        self._check_zero_scale()

    def _check_zero_scale(self):
        """Raises an error if the scale is zero for any feature."""
        zero_scale_features = [
            feature for feature, da in self.scaler.sel(parameter=['scale', 'std']).data_vars.items()
            if (da == 0).any()
        ]
        if any(zero_scale_features):
             raise ValueError(f'Zero scale values found for features: {zero_scale_features}.')
        
    def scale(
        self,
        dataset: xr.Dataset,
    ) -> xr.Dataset:
        """Scale a data set with a precaculated_scaler.
        
        $$ unscaled_dataset = (dataset - center) + scale $$

        Applies a linear transformation to the features (data_vars) in an xr.Dataset.
        This transformation is the inverse of the one applied by self.unscale().
        Agnostic to the dimensions and coordinates of the dataset.
        
        Parameters
        ----------
        dataset : xr.Dataset
            Dataset to be scaled.
        
        Returns
        -------
        xr.Dataset
            The new dataset where all scalable features are scaled.
        
        Raises
        ------
        ValueError if the dataset contains features that are not in the scaler parameters.
        """
        missing_features = [feature for feature in dataset if feature not in self.scaler.data_vars]
        if any(missing_features):
            raise ValueError(f'Requesting to scale variables that are not part of the scaler: {missing_features}')
        return (dataset - self.scaler.sel(parameter='center')) / self.scaler.sel(parameter='scale')

    def unscale(
        self,
        dataset: xr.Dataset
    ) -> xr.Dataset:
        """Un-scale a data set with a precalculated scaler.
        
        $$ scaled_dataset = dataset * scale + center $$
        
        Applies a linear transformation to the features (data_vars) in an xr.Dataset.
        This transformation is the inverse of the one applied by self.scale().
        Agnostic to the dimensions and coordinates of the dataset.
        
        Parameters
        ----------
        dataset : xr.Dataset
            Dataset to be un-scaled.
        
        Returns
        -------
        xr.Dataset
            The new dataset where all scalable features are un-scaled. 
        
        Raises
        ------
        ValueError if the dataset contains features that are not in the scaler parameters.
        """
        missing_features = [feature for feature in dataset if feature not in self.scaler.data_vars]
        if any(missing_features):
            raise ValueError(f'Requesting to unscale variables that are not part of the scaler: {missing_features}')
        return dataset * self.scaler.sel(parameter='scale') + self.scaler.sel(parameter='center')
