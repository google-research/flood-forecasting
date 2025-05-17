import itertools
from typing import List, Optional

import numpy as np
import pandas as pd
import xarray as xr


def _flatten_feature_groups(groups: List[List[str]])-> List[str]:
    """Conditionally flattens a list of lists."""
    if not groups:
        return []
    
    if not isinstance(groups, list):
        # This catches feature groups used for frequencies, which is not supported.
        raise ValueError('Feature groups must be supplied as a list.')
    
    if any(isinstance(g, list) for g in groups) and not all(isinstance(g, list) for g in groups):
        raise ValueError('A mix of lists and features was supplied as feature groups.')

    if isinstance(groups[0], list):
        return list(itertools.chain.from_iterable(groups))

    else:
        return groups


def extract_feature_groups(
    groups: List[List[str]],
    features: List[str]
) -> List[List[str]]:
    """Extract relevant feature groups from a larger set of groups.
    
    Parameters
    ----------
    groups : List[List[str]]
        List of feature groups.
    features : List[str]
        Flattened list of features.
        
    Returns
    -------
    List of feature groups including only groups that contain features in the flattened list.
    
    Raises
    ------
    ValueError if not all features in flattened list are grouped.
    ValueError if there are some groups that are fractionally represented in the flattened list.
    ValueError if no groups are found that are comprised of features in the flattened list.
    """
    
    # Return no groups for no features.
    if not features:
        return []
    
    # Extract groups that contain ALL features from the supplied list.
    extracted_groups = [
        group for group in groups
        if all(feature in features for feature in group)
    ]
    if not extracted_groups:
        raise ValueError('No groups were extracted.')

    # Ensure that no group contains SOME but not ALL features in the feature list.
    partial_groups = [
        group for group in groups
        if any(feature in features for feature in group)
        and not all(feature in features for feature in group)
    ]
    if partial_groups:
        raise ValueError('There appear to be mixed groups in the dataset.')
        
    # Ensure that all features are part of some group.
    missing_features = [
        feature for feature in features
        if feature not in _flatten_feature_groups(extracted_groups)
    ]
    if missing_features:
        raise ValueError(f'Not all features are in feature groups: {missing_features}.')                   
    
    return extracted_groups


def validate_samples(
    is_train: bool,
    dataset: xr.Dataset,
    sample_dates: pd.DatetimeIndex,
    nan_handling_method: Optional[str],
    feature_groups: List[List[str]],
    lead_time: int = 0,   
    seq_length: Optional[int] = None,
    predict_last_n: Optional[int] = None,
    forecast_overlap: Optional[int] = None,
    min_lead_time: Optional[int] = None,
    forecast_features: Optional[List[str]] = None,
    hindcast_features: Optional[List[str]] = None,
    target_features: Optional[List[str]] = None,
    static_features: Optional[List[str]] = None,
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
    nan_handling_method : Optional[str]
        Name of the NaN-handling method. This can be None, but we require that to be passed explicitly.
    feature_groups : List[List[str]]
        A list of feature groups where each group is a list of features. Used in certain types of NaN-handling.
    lead_time : int
        Sequence length for validating a look-ahead sequence of target variables. Defaults to nowcasts.
    seq_length : Optional[int]
        Sequence length for validating a look-back sequence of hindcast variables. Required if `hindcast_features`
        is not None.
    predict_last_n : Optional[int]
        Sequence length used for calculating loss function. At least one target data must be non-NaN in the last
        `predict_last_n` values of the target sequence. Required if `target_features` is not None.
    forecast_overlap : Optional[int]
        Defines the look-back for forecast data.
    min_lead_time : Optional[int]
        Integer representing the minimum lead time in the forecast data as a number of timesteps.
        Required if forecast_overlap > 0.
    forecast_features : Optional[List[str]]
        List of forecast features to validate. Defaults to None. At least one feature list is required.
    hindcast_features : Optional[List[str]]
        List of hindcast features to validate. Defaults to None. At least one feature list is required.
    target_features : Optional[List[str]]
        List of target features to validate. Defaults to None. At least one feature list is required.
    static_features : Optional[List[str]]
        List of static features to validate. Defaults to None. At least one feature list is required.

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
    if forecast_features is None and hindcast_features is None and target_features is None and static_features is None:
        raise ValueError('At least one feature list is required to validate samples.')
       
    masks = []
        
    # Statics must pass an ALL-valid check.
    if static_features:
        masks.append(validate_samples_all(dataset=dataset[static_features]).rename('statics'))
        
    # Hindcasts must pass a check that depends on the NaN-handling.
    if hindcast_features:
        if seq_length is None:
            raise ValueError('Sequence length is required when validating hindcast data.')

        hindcast_groups = extract_feature_groups(feature_groups, hindcast_features)
        mask = validate_samples_for_nan_handling(
            dataset=dataset[hindcast_features],
            nan_handling_method=nan_handling_method,
            feature_groups=hindcast_groups
        )

        masks.append(
            validate_sequence_all(
                mask=mask,
                seq_length=seq_length,
                shift_right=0
            ).rename('hindcasts')
        )
        
    # Forecasts must pass a check that depends on the NaN-handling.
    if forecast_features:
        forecast_groups = extract_feature_groups(feature_groups, forecast_features)
        masks.append(
            validate_samples_for_nan_handling(
                dataset=dataset[forecast_features],
                nan_handling_method=nan_handling_method,
                feature_groups=forecast_groups
            ).rename('forecasts')
        )
        
        if forecast_overlap is not None and forecast_overlap > 0:
            if min_lead_time is None:
                raise ValueError('`min_lead_time`is required when validating a forecast overlap sequence.')
            
            mask = validate_samples_for_nan_handling(
                dataset=dataset[forecast_features].isel(lead_time=0).squeeze().drop('lead_time'),
                nan_handling_method=nan_handling_method,
                feature_groups=forecast_groups
            )
            
            masks.append(
                validate_sequence_all(
                    mask=mask,
                    seq_length=forecast_overlap,
                    shift_right=-min_lead_time
                ).rename('forecast_overlap')
            )
            
    # Targets must pass and ANY-valid check.
    if target_features and is_train:
        if predict_last_n is None:
            raise ValueError('Target sequence length is required when validating target data.')
        mask = validate_samples_any(dataset=dataset[target_features])
        masks.append(
            validate_sequence_any(
                mask=mask,
                seq_length=predict_last_n,
                shift_right=lead_time
            ).rename('targets')
        )

    # Mask any dates that are not valid sample dates.
    all_dates = dataset.date.values
    all_basins = dataset.basin.values
    masks.append(
        xr.DataArray(
            [np.in1d(all_dates, sample_dates)]*len(all_basins),
            coords={'basin': all_basins, 'date': all_dates},
            dims=['basin', 'date']
        ).rename('dates')
    )

    # All masks must be valid according to their own checks for the sample to be valid.
    valid_sample_mask = xr.merge(masks).to_array(dim='variable').all(dim='variable')
    
    return valid_sample_mask, masks


def validate_samples_for_nan_handling(
    dataset: xr.Dataset,
    nan_handling_method: Optional[str],
    feature_groups: List[List[str]]
) -> xr.DataArray:
    """Validates samples based on the NaN-handling method.
    
    Parameters
    ----------
    dataset : xarray.Dataset
        Dataset with any combination of dims (basin, date, lead_time)
    nan_handling_method : Optional[str]
        Name of the NaN-handling method. This can be None, but we require that to be passed explicitly.
    feature_groups : List[List[str]]
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
    if not feature_groups and nan_handling_method in ['masked_mean', 'attention']:
        raise ValueError(f'Feature groups is empty for NaN-handling method: {nan_handling_method}.')
    
    if nan_handling_method is None or nan_handling_method.lower() == 'none':
        return validate_samples_all(dataset)
    elif nan_handling_method.lower() == 'input_replacing':
        return validate_samples_any(dataset)
    elif nan_handling_method.lower() == 'masked_mean':
        return validate_samples_any_all_group(dataset, feature_groups)
    elif nan_handling_method.lower() == 'attention':
        return validate_samples_any_all_group(dataset, feature_groups)
    else:
        raise ValueError(f'Unrecognized NaN-handling method: {nan_handling_method}.')

    
def validate_samples_any_all_group(
    dataset: xr.Dataset,
    feature_groups: List[List[str]]
) -> xr.DataArray:
    """Validates samples using an ANY-valid / ALL-valid criteria over groups.
    
    A sample is valid if ANY group passes an ALL NaN check. 
    This is used for mean_embedding type NaN-handling methods, including attention-based.
    
    Parameters
    ----------
    dataset : xarray.Dataset
        Dataset with any combination of dims (basin, date, lead_time)
    feature_groups : List[List[str]]
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
        group_masks.append(validate_samples_all(dataset[group]).rename(str(idx)))
    return xr.merge(group_masks).to_array(dim='variable').any(dim='variable')    


def validate_samples_all_any_group(
    dataset: xr.Dataset,
    feature_groups: List[List[str]]
) -> xr.DataArray:
    """Validates samples using an ALL-valid / ANY-valid criteria over groups.
    
    A sample is valid if ALL groups pass an ANY NaN check. 
    
    Parameters
    ----------
    dataset : xarray.Dataset
        Dataset with any combination of dims (basin, date, lead_time)
    feature_groups : List[List[str]]
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
        group_masks.append(validate_samples_any(dataset[group]).rename(str(idx)))
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
    mask: xr.DataArray,
    seq_length: int,
    shift_right: int
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
    return mask.rolling(
        date=seq_length,
        min_periods=seq_length
    ).reduce(np.all).shift(date=-shift_right).fillna(False).astype(bool)


def validate_sequence_any(
    mask: xr.DataArray,
    seq_length: int,
    shift_right: int
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
    return mask.rolling(
        date=seq_length,
        min_periods=seq_length
    ).reduce(np.any).shift(date=-shift_right).fillna(False).astype(bool)
