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



import numpy as np
import pandas as pd
import xarray as xr


def _expand_lead_times(
    da: xr.DataArray, lead_times: xr.DataArray | np.ndarray
) -> xr.DataArray:
    """Expands `da` with a `lead_time` dimension via shifting days back by lead time.

    The shifting generates nans from the end as much as the lead time value is.
    """
    if 'lead_time' in da.dims:
        raise ValueError('Trying to expand a dataarray that already has a lead time.')
    # TODO (future) :: This assumes daily data.
    lt_das = (da.shift(date=-int(lt / np.timedelta64(1, "D"))) for lt in lead_times)
    lt_da = xr.concat(lt_das, dim=pd.Index(data=lead_times, name="lead_time"))
    return lt_da

def _union_features_with_same_dimensions(
  feature_da: xr.DataArray,
  mask_feature_da: xr.DataArray,
) -> xr.DataArray:
    """Mask (align and union) da with mask, taking values from da else from mask for nans."""
    return feature_da.combine_first(mask_feature_da)


def _union_lead_time_feature_with_non_lead_time_feature(
  feature_da: xr.DataArray,
  mask_feature_da: xr.DataArray,
) -> xr.DataArray:
    """Mask the lead-time feature with the non-lead-time feature.

    Assuming feature da has "lead_time" dim but the masking da doesn't (e.g. obs).
    """
    lead_times = feature_da.coords["lead_time"]
    lt_mask_da = _expand_lead_times(mask_feature_da, lead_times)
    return _union_features_with_same_dimensions(feature_da, lt_mask_da)


def _union_non_lead_time_feature_with_lead_time_feature(
  feature_da: xr.DataArray,
  mask_feature_da: xr.DataArray,
) -> xr.DataArray:
    """Mask the non-lead-time feature with the lead-time feature.

    Fills nans in the 2d feature from the earliest forecast 3d (with lead time) feature
    via min lead time.

    Align forecast's "issue date" (when was made) with feature's "valid date" (when applied).
    Shift forecast data forward by lead time to match dates.
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
    union_feature_mapping: dict[str, str]
) -> xr.Dataset:
    """Replace missing values in various features with values from another feature.
    
    Works with features that do and do not have a `lead_time` dimension.
    
    Parameters
    ----------
    dataset : xr.Dataset
        Dataset of features to be masked *and* features used for masking.
    union_feature_mapping : dict[str, str]
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

    dataset.update(new_das)
    return dataset
