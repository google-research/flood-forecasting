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

import logging
import itertools

import pandas as pd
import xarray as xr

LOGGER = logging.getLogger(__name__)


def _skip_all_zero_samples(dataset: xr.Dataset) -> xr.DataArray:
    mask = (dataset != 0).to_array(dim='variable').any(dim='variable')
    if 'lead_time' in dataset.dims:
        mask = mask.any(dim='lead_time')
    return mask


def validate_samples(
    is_train: bool,
    dataset: xr.Dataset,
    sample_dates: pd.DatetimeIndex,
    nan_handling_method: str | None,
    feature_groups: list[list[str]],
    lead_time: int = 0,
    seq_length: int | None = None,
    predict_last_n: int | None = None,
    forecast_overlap: int | None = None,
    min_lead_time: int | None = None,
    forecast_features: list[str] | None = None,
    hindcast_features: list[str] | None = None,
    target_features: list[str] | None = None,
    static_features: list[str] | None = None,
    allzero_samples_are_invalid: bool = False,
) -> xr.DataArray:
    """Validates samples based on the NaN-handling method.

    Parameters
    ----------
    is_train : bool
        Training datasets require a check on target data -- inference datasets do not.
    dataset : xarray.Dataset
        Dataset with any combination of dims (basin, date, lead_time)
    sample_dates : pd.DatetimeIndex
        Sample dates.
    nan_handling_method : str | None
        Name of the NaN-handling method. This can be None, but we require that to be passed explicitly.
    feature_groups : list[list[str]]
        A list of feature groups where each group is a list of features. Used in certain types of NaN-handling.
    lead_time : int
        Sequence length for validating a look-ahead sequence of target variables. Defaults to nowcasts.
    seq_length : int | None
        Sequence length for validating a look-back sequence of hindcast variables. Required if `hindcast_features`
        is not None.
    predict_last_n : int | None
        Sequence length used for calculating loss function. At least one target data must be non-NaN in the last
        `predict_last_n` values of the target sequence. Required if `target_features` is not None.
    forecast_overlap : int | None
        Defines the look-back for forecast data.
    min_lead_time : int | None
        Integer representing the minimum lead time in the forecast data as a number of timesteps.
        Required if forecast_overlap > 0.
    forecast_features : list[str] | None
        List of forecast features to validate. Defaults to None. At least one feature list is required.
    hindcast_features : list[str] | None
        List of hindcast features to validate. Defaults to None. At least one feature list is required.
    target_features : list[str] | None
        List of target features to validate. Defaults to None. At least one feature list is required.
    static_features : list[str] | None
        List of static features to validate. Defaults to None. At least one feature list is required.
    allzero_samples_are_invalid : bool
        Whether to skip all-zero samples (via a mask).

    Returns
    -------
    xarray.DataArray
        Contains the indexes of all valid samples as tuples (basin, date).

    Raises
    -------
    ValueError if no feature lists are provided.
    ValueError if a sequence length (`seq_length`) is not provided for hindcasts.
    ValueError if a forecast overlap sequence length (`forecast_overlap`) is provided
        without a minimum lead time (`min_lead_time`).
    ValueError if a target sequence length (`predict_last_n`) is not provided for targets.
    ValueError if a hindcast or forecast feature is missing from the feature groups.
    """
    if (
        forecast_features is None
        and hindcast_features is None
        and target_features is None
        and static_features is None
    ):
        raise ValueError(
            'At least one feature list is required to validate samples.'
        )

    masks = []

    # Statics must pass an ALL-valid check.
    if static_features:
        LOGGER.debug('static_features')
        masks.append(
            validate_samples_all(dataset=dataset[static_features]).rename(
                'statics'
            )
        )

    # Hindcasts must pass a check that depends on the NaN-handling.
    if hindcast_features:
        LOGGER.debug('hindcast features')
        if seq_length is None:
            raise ValueError(
                'Sequence length is required when validating hindcast data.'
            )

        mask = validate_samples_for_nan_handling(
            dataset=dataset[hindcast_features],
            nan_handling_method=nan_handling_method,
            feature_groups=[hindcast_features],
        )

        masks.append(
            validate_sequence_all(
                mask=mask, seq_length=seq_length, shift_right=0
            ).rename('hindcasts')
        )

    # Forecasts must pass a check that depends on the NaN-handling.
    if forecast_features:
        LOGGER.debug('forecast features')
        masks.append(
            validate_samples_for_nan_handling(
                dataset=dataset[forecast_features],
                nan_handling_method=nan_handling_method,
                feature_groups=[forecast_features],
            ).rename('forecasts')
        )

        if forecast_overlap is not None and forecast_overlap > 0:
            LOGGER.debug('forecast features:overlap')
            if min_lead_time is None:
                raise ValueError(
                    '`min_lead_time`is required when validating a forecast overlap sequence.'
                )

            mask = validate_samples_for_nan_handling(
                dataset=dataset[forecast_features]
                .isel(lead_time=0)
                .squeeze()
                .drop_vars('lead_time'),
                nan_handling_method=nan_handling_method,
                feature_groups=[forecast_features],
            )

            masks.append(
                validate_sequence_all(
                    mask=mask,
                    seq_length=forecast_overlap,
                    shift_right=-min_lead_time,
                ).rename('forecast_overlap')
            )

    # Targets must pass and ANY-valid check.
    if target_features and is_train:
        LOGGER.debug('target features')
        if predict_last_n is None:
            raise ValueError(
                'Target sequence length is required when validating target data.'
            )
        dataset_targets = dataset[target_features]

        if allzero_samples_are_invalid:
            masks.append(
                _skip_all_zero_samples(dataset_targets).rename(
                    'non_zero_targets'
                )
            )

        mask = validate_samples_any(dataset=dataset_targets)
        masks.append(
            validate_sequence_any(
                mask=mask, seq_length=predict_last_n, shift_right=lead_time
            ).rename('targets')
        )

    # Mask any dates that are not valid sample dates.
    LOGGER.debug('invalid sample dates')
    # 1d `True` mask for basin dim with same dims, coords, chunking, as dataset.basin.
    all_basins = xr.ones_like(dataset.basin, dtype=bool)
    # 1d bool mask for date. It's a single dim and coord 'date'.
    is_date_valid = dataset.date.isin(sample_dates)
    # Combine both masks into 2d. The & broadcasts the (date,) with (basin, date,).
    # It's equiv to taking all basins, copying and stacking them, then ANDing them.
    masks.append((all_basins & is_date_valid).rename('dates'))

    LOGGER.debug('valid_sample_mask')
    # masks is a mix of 1d (static) and 2d features' bool masks.
    # we merge them by lazily apply (&) each mask so objects stay small.
    # All masks must be valid according to their own checks for the sample to be valid.
    # 1d `True` mask for date dim with same dims, coords, chunking, as dataset.date.
    all_dates = xr.ones_like(dataset.date, dtype=bool)
    # Create the init valid sample temple. xarray broadcasts 1d with 2d arrays into 2d
    # like basin and date.
    valid_sample_mask = all_basins & all_dates
    for mask in masks:
        # when mask is 2d: element-wise &
        # when mask is 1d (the dim "basin" statics feature): it's exapnded across date
        #                                                    then element-wise &.
        valid_sample_mask = valid_sample_mask & mask

    return valid_sample_mask, masks


def validate_samples_for_nan_handling(
    dataset: xr.Dataset,
    nan_handling_method: str | None,
    feature_groups: list[list[str]],
) -> xr.DataArray:
    """Validates samples based on the NaN-handling method.

    Parameters
    ----------
    dataset : xarray.Dataset
        Dataset with any combination of dims (basin, date, lead_time)
    nan_handling_method : str | None
        Name of the NaN-handling method. This can be None, but we require that to be passed explicitly.
    feature_groups : list[list[str]]
        A list of feature groups where each group is a list of features. Used in certain types of NaN-handling.

    Returns
    -------
    xarray.DataArray
        Boolean valid sample mask.

    Raises
    ------
    ValueError if NaN-handling method requires groups but none are provided.
    ValueError for un-recognized NaN-handling method.
    """
    if not feature_groups and nan_handling_method in [
        'masked_mean',
        'attention',
    ]:
        raise ValueError(
            f'Feature groups is empty for NaN-handling method: {nan_handling_method}.'
        )

    if nan_handling_method is None or nan_handling_method.lower() == 'none':
        return validate_samples_all(dataset)
    elif nan_handling_method.lower() == 'input_replacing':
        return validate_samples_any(dataset)
    elif nan_handling_method.lower() == 'masked_mean':
        return validate_samples_any_all_group(dataset, feature_groups)
    elif nan_handling_method.lower() == 'attention':
        return validate_samples_any_all_group(dataset, feature_groups)
    elif nan_handling_method.lower() == 'unioning':
        return validate_samples_all_any_group(dataset, feature_groups)
    else:
        raise ValueError(
            f'Unrecognized NaN-handling method: {nan_handling_method}.'
        )


def validate_samples_any_all_group(
    dataset: xr.Dataset, feature_groups: list[list[str]]
) -> xr.DataArray:
    """Validates samples using an ANY-valid / ALL-valid criteria over groups.

    A sample is valid if ANY group passes an ALL NaN check.
    This is used for mean_embedding type NaN-handling methods, including attention-based.

    Parameters
    ----------
    dataset : xarray.Dataset
        Dataset with any combination of dims (basin, date, lead_time)
    feature_groups : list[list[str]]
        A list of feature groups where each group is a list of features. Used in certain types of NaN-handling.

    Returns
    -------
    xarray.DataArray
        Boolean valid sample mask.

    Raises
    ------
    ValueError if groups list is empty.
    """
    if not feature_groups:
        raise ValueError('No feature groups provided.')
    group_masks = []
    for idx, group in enumerate(feature_groups):
        group_masks.append(
            validate_samples_all(dataset[group]).rename(str(idx))
        )
    return xr.merge(group_masks).to_array(dim='variable').any(dim='variable')


def validate_samples_all_any_group(
    dataset: xr.Dataset, feature_groups: list[list[str]]
) -> xr.DataArray:
    """Validates samples using an ALL-valid / ANY-valid criteria over groups.

    A sample is valid if ALL groups pass an ANY NaN check.

    Parameters
    ----------
    dataset : xarray.Dataset
        Dataset with any combination of dims (basin, date, lead_time)
    feature_groups : list[list[str]]
        A list of feature groups where each group is a list of features. Used in certain types of NaN-handling.

    Returns
    -------
    xarray.DataArray
        Boolean valid sample mask.
    """
    if not feature_groups:
        raise ValueError('No feature groups provided.')
    group_masks = []
    for idx, group in enumerate(feature_groups):
        group_masks.append(
            validate_samples_any(dataset[group]).rename(str(idx))
        )
    return xr.merge(group_masks).to_array(dim='variable').all(dim='variable')


def validate_samples_any(
    dataset: xr.Dataset,
) -> xr.DataArray:
    """Validates samples using an ANY-valid criteria.

    This is used for input_replacing NaN-handling.

    Parameters
    ----------
    dataset : xarray.Dataset
        Dataset with any combination of dims (basin, date, lead_time)

    Returns
    -------
    xarray.DataArray
        Boolean valid sample mask.
    """
    if 'lead_time' in dataset.dims:
        mask = dataset.isnull().all(dim='lead_time')
    else:
        mask = dataset.isnull()
    return ~mask.to_array(dim='variable').all(dim='variable')


def validate_samples_all(
    dataset: xr.Dataset,
) -> xr.DataArray:
    """Validates samples using an ALL-valid criteria.

    This is used when no NaN-handling is specified.

    Parameters
    ----------
    dataset : xarray.Dataset
        Dataset with any combination of dims (basin, date, lead_time)

    Returns
    -------
    xarray.DataArray
        Boolean valid sample mask.
    """
    if 'lead_time' in dataset.dims:
        mask = dataset.isnull().any(dim='lead_time')
    else:
        mask = dataset.isnull()
    return ~mask.to_array(dim='variable').any(dim='variable')


def validate_sequence_all(
    mask: xr.DataArray, seq_length: int, shift_right: int
) -> xr.DataArray:
    """Validates samples with an ALL-valid criteria for a `date` sequence.

    Parameters
    ----------
    mask : xarray.DataArray
        Boolean mask with a `date` dimension.
    seq_length : int
        Number of timesteps in the time window.
    shift_right : int
        Number of timesteps to shift the rightmost inclusive bound of the sequence time window.

    Returns
    -------
    xarray.DataArray
        Boolean valid sample mask.
    """
    mask = mask.fillna(False)

    # The sliding window is valid if ALL timestemps are True which is what
    # min represents in the window being True (1).
    valid_windows = mask.rolling(date=seq_length, min_periods=seq_length).min()

    # Move the valid windows data along the date dim, to align the validity
    # info of windows with the timestamp of the value to predict.
    # This may create nan values at the end, e.g. [False, True, True]
    # shifted by -1 becomes [True, True, nan].
    # Noting nans are of type float so this casts the type to float.
    shifted_windows = valid_windows.shift(date=-shift_right)

    # Fill the missing values from the shift with `False` so it's equiv
    # to the nan value, so it's considered as an invalid (shifted) window.
    filled_windows = shifted_windows.fillna(False)

    # Ensure the data type is all bool, and this saves memory.
    final_mask = filled_windows.astype(bool)

    return final_mask


def validate_sequence_any(
    mask: xr.DataArray, seq_length: int, shift_right: int
) -> xr.DataArray:
    """Validates samples with an ANY-valid criteria for a `date` sequence.

    Parameters
    ----------
    mask : xarray.DataArray
        Boolean mask with a `date` dimension.
    seq_length : int
        Number of timesteps in the time window.
    shift_right : int
        Number of timesteps to shift the rightmost inclusive bound of the sequence time window.

    Returns
    -------
    xarray.DataArray
        Boolean valid sample mask.
    """
    mask = mask.fillna(False)

    # The sliding window is valid if ANY timestemp is True which is what
    # max represents in the window being True (1).
    valid_windows = mask.rolling(date=seq_length, min_periods=seq_length).max()

    # Move the valid windows data along the date dim, to align the validity
    # info of windows with the timestamp of the value to predict.
    # This may create nan values at the end, e.g. [False, True, True]
    # shifted by -1 becomes [True, True, nan].
    # Noting nans are of type float so this casts the type to float.
    shifted_windows = valid_windows.shift(date=-shift_right)

    # Fill the missing values from the shift with `False` so it's equiv
    # to the nan value, so it's considered as an invalid (shifted) window.
    filled_windows = shifted_windows.fillna(False)

    # Ensure the data type is all bool, and this saves memory.
    final_mask = filled_windows.astype(bool)

    return final_mask
