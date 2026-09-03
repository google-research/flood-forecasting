"""MultiMet Weather Forcing & Feature Engineering Helpers for Data Assimilation."""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
import xarray as xr

from googlehydrology import datasetzoo
from googlehydrology import evaluation
from googlehydrology import modelzoo
from googlehydrology.modelzoo.basemodel import BaseModel
from googlehydrology import utils


_GLOBAL_MULTIMET_DS_CACHE: Dict[str, xr.Dataset] = {}


def norm_var(scaler: xr.Dataset, var_name: str, val: np.ndarray) -> np.ndarray:
    """Normalize feature values using mean and std from scaler.nc."""
    m_key = f"{var_name}_sim" if f"{var_name}_sim" in scaler else var_name
    s_key = f"{var_name}_std" if f"{var_name}_std" in scaler else (f"{var_name}_sim" if f"{var_name}_sim" in scaler else var_name)

    if m_key in scaler:
        mean_val = float(scaler[m_key].sel(parameter='mean').values)
        std_val = float(scaler[s_key].sel(parameter='std').values)
    else:
        mean_val = 0.0
        std_val = 1.0

    if std_val == 0 or np.isnan(std_val):
        std_val = 1.0

    return (val - mean_val) / std_val


def _get_source_var(union_mapping: dict, target_var: str) -> str:
    """Resolve target variable name to its source product name using union_mapping."""
    if not union_mapping:
        return target_var
    for prod_name, var_dict in union_mapping.items():
        if isinstance(var_dict, dict):
            for src_name, tgt_name in var_dict.items():
                if tgt_name == target_var or src_name == target_var:
                    return src_name
    return target_var


def _get_product_for_var(var_name: str) -> str:
    """Determines MultiMet product directory from variable prefix."""
    v = var_name.lower()
    if v.startswith('graphcast'):
        return 'GRAPHCAST'
    if v.startswith('hres'):
        return 'HRES'
    if v.startswith('cpc'):
        return 'CPC'
    if v.startswith('imerg'):
        return 'IMERG'
    return 'ERA5_LAND'


def prepare_multimet_batch(
    mode: str,
    basin_id: str,
    issue_date: str,
    cfg,
    scaler: xr.Dataset,
    caravan_attrs: pd.DataFrame,
    ds_caravan: xr.Dataset,
    multimet_dir: Path | str = Path('/usr/local/google/home/kruparell/Caravans_MultiMet'),
    hindcast_window_days: int = 365,
    forecast_lead_days: int = 7,
    forecast_product: str = 'HRES',
    fixed_leadtime: int = 0
) -> Dict[str, Union[torch.Tensor, np.ndarray]]:
    """Builds a standardized PyTorch batch dictionary for MeanEmbeddingForecastLSTM with Data Assimilation.

    Parameters
    ----------
    mode : str
        'all_reanalysis', 'reanalysis_and_fixed_leadtime', or 'multimet_0_and_1_to_7'
    basin_id : str
        Basin identifier (e.g. 'camels_12451000')
    issue_date : str
        Forecast issue date t0 (e.g. '2020-12-31')
    cfg : Config
        Google Hydrology run configuration instance
    scaler : xr.Dataset
        Dataset containing feature normalization statistics (scaler.nc)
    caravan_attrs : pd.DataFrame
        DataFrame of catchment static attributes indexed by gauge_id
    ds_caravan : xr.Dataset
        Full Caravan dataset for streamflow observations
    multimet_dir : Path or str
        Path to CaravansMultiMet directory containing Zarr archives
    hindcast_window_days : int
        Number of historical days up to issue_date t0 (default 365)
    forecast_lead_days : int
        Number of forecast lead days after t0 (default 7)
    forecast_product : str
        Forecast product name ('HRES' or 'GRAPHCAST')
    fixed_leadtime : int
        Fixed lead_time index (0-based) for 'reanalysis_and_fixed_leadtime' mode

    Returns
    -------
    dict
        PyTorch batch dictionary ready for model forward pass containing:
        'date', 'x_s', 'x_d', 'x_d_hindcast', 'x_d_forecast', 'y', 'c_n', 'h_n'
    """
    valid_modes = ['all_reanalysis', 'reanalysis_and_fixed_leadtime', 'multimet_0_and_1_to_7']
    if mode not in valid_modes:
        raise ValueError(f"Invalid mode '{mode}'. Must be one of {valid_modes}.")

    multimet_path = Path(multimet_dir)
    t0_dt = pd.to_datetime(issue_date)

    start_date_h = (t0_dt - pd.Timedelta(days=hindcast_window_days - 1)).strftime('%Y-%m-%d')
    end_date_h = t0_dt.strftime('%Y-%m-%d')
    end_date_f = (t0_dt + pd.Timedelta(days=forecast_lead_days)).strftime('%Y-%m-%d')

    dates_h = pd.date_range(start_date_h, end_date_h).strftime('%Y-%m-%d').values
    dates_f = pd.date_range(start_date_h, end_date_f).strftime('%Y-%m-%d').values

    # Extract observed streamflow from Caravan dataset
    obs_discharge = ds_caravan['streamflow'].sel(date=dates_f).values.astype(np.float32)

    # Streamflow normalization parameters
    q_mean = float(scaler['streamflow'].sel(parameter='mean').values)
    q_std = float(scaler['streamflow'].sel(parameter='std').values)

    hindcast_dict = {}
    forecast_dict = {}
    union_map = getattr(cfg, 'union_mapping', {})

    def _get_dataset(prod_name: str) -> xr.Dataset:
        cache_key = f"{prod_name}_{basin_id}"
        if cache_key not in _GLOBAL_MULTIMET_DS_CACHE:
            p = multimet_path / prod_name / 'timeseries.zarr'
            try:
                _GLOBAL_MULTIMET_DS_CACHE[cache_key] = xr.open_zarr(p, consolidated=True, decode_timedelta=False).sel(basin=basin_id)
            except Exception:
                _GLOBAL_MULTIMET_DS_CACHE[cache_key] = xr.open_zarr(p, decode_timedelta=False).sel(basin=basin_id)
        return _GLOBAL_MULTIMET_DS_CACHE[cache_key]

    def _fetch_var(var_name: str, dates_target: np.ndarray, lead_idx: Optional[int] = 0) -> np.ndarray:
        prod = _get_product_for_var(var_name)
        ds = _get_dataset(prod)
        da = ds[var_name].sel(date=dates_target)
        if 'lead_time' in da.dims and lead_idx is not None:
            da = da.isel(lead_time=lead_idx)
        return da.values.astype(np.float32)

    # -------------------------------------------------------------------------
    # Option 1: All Reanalysis (Lead 0 / Observation across entire span)
    # -------------------------------------------------------------------------
    if mode == 'all_reanalysis':
        for group, feat_list in cfg.hindcast_inputs.items():
            for f_name in feat_list:
                src_var = _get_source_var(union_map, f_name)
                val = _fetch_var(src_var, dates_h, lead_idx=0)
                hindcast_dict[f_name] = torch.tensor(norm_var(scaler, f_name, val), dtype=torch.float32).unsqueeze(0).unsqueeze(-1)

        for group, feat_list in cfg.forecast_inputs.items():
            for f_name in feat_list:
                src_var = _get_source_var(union_map, f_name)
                val = _fetch_var(src_var, dates_f, lead_idx=0)
                forecast_dict[f_name] = torch.tensor(norm_var(scaler, f_name, val), dtype=torch.float32).unsqueeze(0).unsqueeze(-1)

    # -------------------------------------------------------------------------
    # Option 2: Reanalysis (Hindcast) & Fixed Leadtime Forecast
    # -------------------------------------------------------------------------
    elif mode == 'reanalysis_and_fixed_leadtime':
        for group, feat_list in cfg.hindcast_inputs.items():
            for f_name in feat_list:
                src_var = _get_source_var(union_map, f_name)
                val = _fetch_var(src_var, dates_h, lead_idx=0)
                hindcast_dict[f_name] = torch.tensor(norm_var(scaler, f_name, val), dtype=torch.float32).unsqueeze(0).unsqueeze(-1)

        for group, feat_list in cfg.forecast_inputs.items():
            for f_name in feat_list:
                src_var = _get_source_var(union_map, f_name)
                val = _fetch_var(src_var, dates_f, lead_idx=fixed_leadtime)
                forecast_dict[f_name] = torch.tensor(norm_var(scaler, f_name, val), dtype=torch.float32).unsqueeze(0).unsqueeze(-1)

    # -------------------------------------------------------------------------
    # Option 3: MultiMet Ensemble (Lead 0 for history, Lead 1..7 for forecast)
    # -------------------------------------------------------------------------
    elif mode == 'multimet_0_and_1_to_7':
        # 1. Hindcast Inputs (days 1..365)
        for group, feat_list in cfg.hindcast_inputs.items():
            for f_name in feat_list:
                src_var = _get_source_var(union_map, f_name)
                val = _fetch_var(src_var, dates_h, lead_idx=0)
                hindcast_dict[f_name] = torch.tensor(norm_var(scaler, f_name, val), dtype=torch.float32).unsqueeze(0).unsqueeze(-1)

        # 2. Forecast Inputs (days 1..372)
        for group, feat_list in cfg.forecast_inputs.items():
            for f_name in feat_list:
                src_var = _get_source_var(union_map, f_name)
                prod = _get_product_for_var(src_var)
                ds = _get_dataset(prod)
                
                lead0_vals = ds[src_var].sel(date=dates_f)
                if 'lead_time' in lead0_vals.dims:
                    lead0_vals = lead0_vals.isel(lead_time=0).values.astype(np.float32)
                    fc_1_7 = ds[src_var].sel(date=issue_date).isel(lead_time=slice(0, forecast_lead_days)).values.astype(np.float32)
                    f_arr = lead0_vals.copy()
                    f_arr[-forecast_lead_days:] = fc_1_7
                else:
                    f_arr = lead0_vals.values.astype(np.float32)
                
                forecast_dict[f_name] = torch.tensor(norm_var(scaler, f_name, f_arr), dtype=torch.float32).unsqueeze(0).unsqueeze(-1)

        obs_discharge = obs_discharge.copy()
        obs_discharge[-forecast_lead_days:] = np.nan

    # -------------------------------------------------------------------------
    # Static Attributes & Target Tensor Formatting
    # -------------------------------------------------------------------------
    if basin_id not in caravan_attrs.index:
        raise ValueError(f"Basin '{basin_id}' not found in static attributes dataset.")
    basin_attrs = caravan_attrs.loc[basin_id]
    static_vals = []
    for attr_name in cfg.static_attributes:
        if attr_name not in basin_attrs or pd.isna(basin_attrs[attr_name]):
            raise ValueError(f"Basin '{basin_id}' is missing required static attribute '{attr_name}'.")
        raw_val = float(basin_attrs[attr_name])
        normed_val = norm_var(scaler, attr_name, raw_val)
        static_vals.append(normed_val)

    x_s_tensor = torch.tensor(static_vals, dtype=torch.float32).unsqueeze(0)
    obs_q_norm = (obs_discharge - q_mean) / q_std
    y_tensor = torch.tensor(obs_q_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)

    batch_data = {
        'date': np.tile(dates_f, (1, 1)),
        'x_s': x_s_tensor,
        'x_d': hindcast_dict,
        'x_d_hindcast': hindcast_dict,
        'x_d_forecast': forecast_dict,
        'y': y_tensor,
        'c_n': torch.zeros((1, 1, cfg.hidden_size), dtype=torch.float32),
        'h_n': torch.zeros((1, 1, cfg.hidden_size), dtype=torch.float32)
    }

    return batch_data


def compute_metrics(obs: np.ndarray, sim: np.ndarray) -> Dict[str, float]:
    """Calculates NSE, KGE, Pearson-r, and RMSE between observation and simulation arrays."""
    valid = ~np.isnan(obs) & ~np.isnan(sim)
    o, s = obs[valid], sim[valid]
    denom = np.sum((o - np.mean(o)) ** 2)
    nse = float(1 - (np.sum((o - s) ** 2) / denom)) if denom != 0 else np.nan
    rmse = float(np.sqrt(np.mean((o - s) ** 2)))
    r = float(np.corrcoef(o, s)[0, 1]) if len(o) > 1 else np.nan
    std_o, std_s = np.std(o), np.std(s)
    kge = float(1 - np.sqrt((r - 1)**2 + (std_s/std_o - 1)**2 + (np.mean(s)/np.mean(o) - 1)**2)) if std_o > 0 and np.mean(o) > 0 else np.nan
    return {'NSE': nse, 'KGE': kge, 'Pearson-r': r, 'RMSE (mm/day)': rmse}


def multimet_batch_generator(
    basin_list,
    issue_dates,
    cfg=None,
    scaler=None,
    caravan_attrs=None,
    ds_streamflow=None,
    mode='multimet_0_and_1_to_7',
    forecast_product='HRES',
    hindcast_window_days=358,
    forecast_lead_days=7,
    fixed_leadtime=0,
    multimet_dir='/usr/local/google/home/kruparell/Caravans_MultiMet'
):
    """Generator yielding (basin_id, issue_date, batch_data) using prepare_multimet_batch."""
    for b_id in basin_list:
        ds_b = ds_streamflow.sel(basin=b_id) if ds_streamflow is not None else None
        for dt in issue_dates:
            batch = prepare_multimet_batch(
                mode=mode,
                basin_id=b_id,
                issue_date=dt,
                cfg=cfg,
                scaler=scaler,
                caravan_attrs=caravan_attrs,
                ds_caravan=ds_b,
                multimet_dir=multimet_dir,
                forecast_product=forecast_product,
                hindcast_window_days=hindcast_window_days,
                forecast_lead_days=forecast_lead_days,
                fixed_leadtime=fixed_leadtime
            )
            yield b_id, dt, batch


def run_multibatch_da_pipeline(
    basin_list: list[str],
    issue_dates: list[str],
    da_cfg: Union[dict, Any],
    model: BaseModel = None,
    cfg: Any = None,
    scaler: xr.Dataset = None,
    caravan_attrs: pd.DataFrame = None,
    ds_streamflow: xr.Dataset = None,
    lead_times: list[int] = [0, 3, 5, 7],
    multimet_dir: Path | str = '/usr/local/google/home/kruparell/Caravans_MultiMet',
) -> Tuple[pd.DataFrame, Dict[str, Dict[int, Dict[str, Any]]]]:
    """Runs Data Assimilation across multiple issue dates and lead times."""
    from googlehydrology.utils.assimilationconfig import AssimilationConfig
    from googlehydrology.evaluation.assimilation import Assimilation

    if isinstance(da_cfg, dict):
        da_cfg = AssimilationConfig(da_cfg)
    assim = Assimilation(da_cfg)

    # Resolve global variables if not explicitly passed
    if model is None or cfg is None or scaler is None or caravan_attrs is None or ds_streamflow is None:
        try:
            import __main__
            if model is None: model = getattr(__main__, 'model', None)
            if cfg is None: cfg = getattr(__main__, 'cfg', None)
            if scaler is None: scaler = getattr(__main__, 'scaler', None)
            if caravan_attrs is None: caravan_attrs = getattr(__main__, 'caravan_attrs', None)
            if ds_streamflow is None: ds_streamflow = getattr(__main__, 'ds_streamflow', getattr(__main__, 'ds_caravan', None))
        except Exception:
            pass

    if model is None or cfg is None or scaler is None or caravan_attrs is None or ds_streamflow is None:
        raise ValueError("model, cfg, scaler, caravan_attrs, and ds_streamflow must not be None.")

    q_mean = float(scaler['streamflow'].sel(parameter='mean').values)
    q_std = float(scaler['streamflow'].sel(parameter='std').values)

    records = []
    batch_hydrographs = {}

    for b_id in basin_list:
        ds_b = ds_streamflow.sel(basin=b_id) if 'basin' in ds_streamflow.dims else ds_streamflow
        for dt in issue_dates:
            batch_hydrographs[dt] = {}
            for L in lead_times:
                batch_L = prepare_multimet_batch(
                    mode='reanalysis_and_fixed_leadtime',
                    basin_id=b_id,
                    issue_date=dt,
                    cfg=cfg,
                    scaler=scaler,
                    caravan_attrs=caravan_attrs,
                    ds_caravan=ds_b,
                    multimet_dir=multimet_dir,
                    forecast_product='HRES',
                    fixed_leadtime=L,
                    hindcast_window_days=358,
                    forecast_lead_days=7
                )
                batch_dates = pd.to_datetime(batch_L['date'][0]).strftime('%Y-%m-%d').values
                obs_q = ds_b['streamflow'].sel(date=batch_dates).values.astype(np.float32)

                with torch.no_grad():
                    b_out = model(batch_L)
                    q_base_norm = b_out['y_hat'][0, :, 0].detach().cpu().numpy()
                q_base = np.maximum(0.1, q_base_norm * q_std + q_mean)

                da_out = assim.assimilate(model, batch_L, verbose=False)
                q_da_norm = da_out['y_hat'][0, :, 0].detach().cpu().numpy()
                q_da = np.maximum(0.1, q_da_norm * q_std + q_mean)

                batch_hydrographs[dt][L] = {
                    'dates': batch_dates,
                    'obs': obs_q,
                    'baseline': q_base,
                    'da_model': q_da
                }

                m_base = compute_metrics(obs_q, q_base)
                m_da = compute_metrics(obs_q, q_da)

                records.append({
                    'Basin ID': b_id,
                    'Issue Date': dt,
                    'Lead Time (Days)': L,
                    'Base NSE': round(m_base['NSE'], 3),
                    'Base KGE': round(m_base['KGE'], 3),
                    'DA NSE': round(m_da['NSE'], 3),
                    'DA KGE': round(m_da['KGE'], 3),
                    'NSE Delta': round(m_da['NSE'] - m_base['NSE'], 3),
                    'KGE Delta': round(m_da['KGE'] - m_base['KGE'], 3)
                })

    df_metrics = pd.DataFrame(records)
    return df_metrics, batch_hydrographs
