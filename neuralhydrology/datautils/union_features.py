from typing import Dict

import numpy as np
import xarray as xr


def _expand_lead_times(da: xr.DataArray, lead_times: xr.DataArray) -> xr.DataArray:
    """Expands `da to include a `lead_time` dimension via shifting.

    Prepare a non-forecast da (e.g. 2d obs with dims basin, date) to be used as a
    mask for a multi lead time forecast da (e.g. 3d basin, date, lead_time).

    Time-shift (backward) copies of da for each lead time (n days ago) so it's used
    as if it were the forecast. And thus concat'd along a new lead_time dim.

    Example:
    -------
    Input dates [10, 20, 30] and lead_times of [1 day, 2 days]:
    - For lead_time=1, it shifts by -1: [nan, 10, 20]
      So the value for the second date is the obs from first date.
    - For lead_time=2, it shifts by -2: [nan, nan, 10]
      So the value for the third date is the obs from second date.
    """
    if 'lead_time' in da.dims:
        raise ValueError('Trying to expand a dataarray that already has a lead time.')
    # TODO (future) :: This assumes daily data.
    # Shift past date to present (negative) e.g. for lead time 1D, Jan 02 -> Jan 01.
    lt_das = (da.shift(date=-int(lt / np.timedelta64(1, "D"))) for lt in lead_times)
    lt_da = xr.concat(lt_das, dim="lead_time")
    return lt_da.assign_coords(lead_time=lead_times)  # Label lead_times as lead_time


def _union_features_with_same_dimensions(
  feature_da: xr.DataArray,
  mask_feature_da: xr.DataArray,
) -> xr.DataArray:
    """Fills nans in feature da with from the mask when both are the same size."""
    return feature_da.combine_first(mask_feature_da)


def _union_lead_time_feature_with_non_lead_time_feature(
  feature_da: xr.DataArray,
  mask_feature_da: xr.DataArray,
) -> xr.DataArray:
    """Fills nans in the 3d target forecast with values from the 2d obs feature.

    Assuming feature da has "lead_time" dim but the masking da doesn't (e.g. obs).

    Expands the 2d mask feature da into a 3d one that matches shape and lead times
    of the 3d feature_da (time shifted mask copies for each lead time). Then combined.
    """
    lead_times = feature_da.coords["lead_time"]
    lt_mask_da = _expand_lead_times(mask_feature_da, lead_times)
    return _union_features_with_same_dimensions(feature_da, lt_mask_da)


def _union_non_lead_time_feature_with_lead_time_feature(
  feature_da: xr.DataArray,
  mask_feature_da: xr.DataArray,
) -> xr.DataArray:
    """Fills nans in the 2d feature from the earliest forecast 3d (with lead time) feature.

    Assuming best available data from the forecast to fill missing obs data is the forecast
    with min lead time. (e.g. 1 day ahead forecast may be better than 5 days ahead)

    Aligning forecast times: forecast's "date" is "issue date" when forecast was made, and
    feature's "date" is "valid date" when it's applied to. Shifting forecast data forward
    by lead time to match dates.
    """
    min_lead_time = mask_feature_da["lead_time"].min().item()  # Best forecast
    min_lead_time_mask_feature = mask_feature_da.sel(
        lead_time=min_lead_time, drop=True
    )  # 2d slice
    shift_days = int(min_lead_time / np.timedelta64(1, "D"))  # forecast time aligning
    # Align mask's "issue date" with target feature's "valid date", e.g. forecast issued
    # on Jan 1 for Jan 2 (lead time 1 day) is shifted forward by 1 day to align with Jan 2.
    mask_values = min_lead_time_mask_feature.shift(date=shift_days)
    return _union_features_with_same_dimensions(feature_da, mask_values)


def union_features(
    dataset: xr.Dataset,
    union_feature_mapping: Dict[str, str]
) -> xr.Dataset:
    """Replace missing values in various features with values from another feature.
    
    Works with features that do and do not have a `lead_time` dimension.
    
    Parameters
    ----------
    dataset : xr.Dataset
        Dataset of features to be masked *and* features used for masking.
    union_feature_mapping : Dict[str, str]
        A 1-to-1 mapping from a feature that will be masked by another feature. 
    
    Returns
    -------
    Dataset of all features from the original dataset where the features that are keys in `union_feature_mapping'
        are masked by the features in their respective dictionary values.
    
    Raises
    ------
    ValueError if a masking feature is not found.
    """
    # Collect new unioned features as overrides to the features from the original dataset
    # that were NOT targets of a union operation as they were.
    new_das = {}

    for feature, mask_feature in union_feature_mapping.items():
        if feature == mask_feature:
            continue
        if feature not in dataset.data_vars:
            continue
        if mask_feature not in dataset.data_vars:
            raise ValueError(f'Masking feature {mask_feature} not in dataset.')

        # Mask the feature depending on their dimensions.
        feature_da = dataset[feature]
        mask_feature_da = dataset[mask_feature]

        if 'lead_time' in feature_da.dims and 'lead_time' not in mask_feature_da.dims:
            new_das[feature] = _union_lead_time_feature_with_non_lead_time_feature(
                feature_da, mask_feature_da
            )
        elif 'lead_time' not in feature_da.dims and 'lead_time' in mask_feature_da.dims:
            new_das[feature] = _union_non_lead_time_feature_with_lead_time_feature(
                feature_da, mask_feature_da
            )
        else:
            new_das[feature] = _union_features_with_same_dimensions(
                feature_da, mask_feature_da
            )

    return dataset.update(new_das)
