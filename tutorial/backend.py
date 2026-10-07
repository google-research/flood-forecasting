# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Hydrology Model Evaluation and Configuration Backend.

This module provides utility functions for managing hydrological model runs,
calculating performance metrics, visualizing geographical data, and 
generating fine-tuning and data assimilation configurations.
"""

# Standard Library Imports
import glob
import os
import re
import shutil
import sys
from typing import Any, Dict, List, Optional, Tuple, Set, Union
import yaml

# Third-Party Library Imports
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from tqdm.notebook import tqdm

# Local Module Imports
# Get the current working directory and go one level up
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from googlehydrology.evaluation import metrics

# --- Model Selection and Data Loading ---

def find_model_run_dirs(base_dir: str) -> Dict[str, str]:
    """
    Identifies directories containing valid test results within a base directory.

    Searches the immediate subdirectories of the base directory for a 
    'test/model_epoch*/test_results.zarr' structure.

    Args:
        base_dir: The root directory to search for model runs.

    Returns:
        A dictionary mapping the subdirectory name (display name) to the full path.
    """
    run_dirs = {}
    
    # List immediate subdirectories
    subdirs = [
        d for d in os.listdir(base_dir) 
        if os.path.isdir(os.path.join(base_dir, d))
    ]

    for subdir in subdirs:
        root = os.path.join(base_dir, subdir)
        test_dir = os.path.join(root, 'test')
        
        if os.path.isdir(test_dir):
            # Check for epoch subdirectories
            epoch_dirs = glob.glob(os.path.join(test_dir, 'model_epoch*'))
            
            for epoch_dir in epoch_dirs:
                if os.path.isdir(os.path.join(epoch_dir, 'test_results.zarr')):
                    run_dirs[subdir] = root
                    break 
                    
    return run_dirs


def read_basin_list(file_path: str) -> Set[str]:
    """
    Reads a text file containing basin IDs and returns them as a set.

    Args:
        file_path: Path to the .txt file with one ID per line.

    Returns:
        A set of unique basin ID strings.
    """
    with open(file_path, 'r') as f:
        ids = [line.strip() for line in f if line.strip()]
    return set(ids)


def tutorial_root() -> str:
    """Returns the absolute path of the `tutorial/` directory that holds this file."""
    return os.path.dirname(os.path.abspath(__file__))


def rebase_tutorial_path(path: Any) -> Any:
    """
    Points a path that was recorded on another machine to the local tutorial folder.

    Run directories that ship with the repository were created elsewhere, so their
    `config.yml` contains absolute paths such as `/home/<user>/flood-forecasting/tutorial/...`.
    If such a path does not exist locally but contains a `/tutorial/` component, the
    prefix in front of `/tutorial/` is replaced by the local tutorial directory.
    Non-string values and paths that already exist are returned unchanged.

    Args:
        path: A path (string) or any other value.

    Returns:
        The (possibly rebased) value.
    """
    if not isinstance(path, str) or os.path.exists(os.path.expanduser(path)):
        return path
    marker = '/tutorial/'
    if marker in path:
        return os.path.join(tutorial_root(), path.split(marker, 1)[1])
    return path


def rebase_config_paths(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Applies `rebase_tutorial_path` to every top-level string value of a run config.

    Args:
        config: A run configuration dictionary.

    Returns:
        A shallow copy of the configuration with rebased paths.
    """
    return {k: rebase_tutorial_path(v) for k, v in config.items()}


def load_model_config_and_basins(run_dir: str) -> Tuple[Dict[str, Any], Set[str], Set[str]]:
    """
    Loads model configuration and its associated training/testing basin sets.

    Args:
        run_dir: Path to the directory containing 'config.yml'.

    Returns:
        A tuple containing (config_dict, train_basin_ids, test_basin_ids).
    """
    config_path = os.path.join(run_dir, 'config.yml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    train_basin_ids = read_basin_list(rebase_tutorial_path(config.get('train_basin_file', '')))
    test_basin_ids = read_basin_list(rebase_tutorial_path(config.get('test_basin_file', '')))

    print(f"Loaded {len(train_basin_ids)} training basin IDs.")
    print(f"Loaded {len(test_basin_ids)} test basin IDs.")

    return config, train_basin_ids, test_basin_ids


# --- Visualization ---

def plot_colored_shapefile(
    gdf: gpd.GeoDataFrame,
    column: str,
    title: str,
    cmap: Optional[str] = None,
    colors: Optional[Dict[Any, str]] = None,
    figsize: Tuple[int, int] = (12, 12),
    missing_kwds: Optional[Dict[str, Any]] = None
):
    """
    Utility to plot a GeoDataFrame with either a colormap or discrete category colors.

    Args:
        gdf: The GeoDataFrame to visualize.
        column: The column name used for coloring.
        title: The plot title.
        cmap: Matplotlib colormap name.
        colors: Optional dictionary mapping column values to specific hex/color strings.
        figsize: Size of the resulting figure.
        missing_kwds: Dictionary of keywords for handling missing data plotting.
    """
    fig, ax = plt.subplots(figsize=figsize)

    if colors:
        for category, color in colors.items():
            subset = gdf[gdf[column] == category]
            if not subset.empty:
                 subset.plot(
                    ax=ax,
                    color=color,
                    label=category,
                    alpha=0.7,
                    edgecolor='black',
                    linewidth=0.5
                )
        ax.legend()
    else:
        gdf.plot(
            ax=ax,
            column=column,
            legend=True,
            cmap=cmap,
            alpha=0.7,
            edgecolor='black',
            linewidth=0.5,
            missing_kwds=missing_kwds
        )

    ax.set_title(title)
    ax.set_axis_off()
    plt.show()


def plot_train_test_shapefile(
    shapefile_path: str,
    train_basin_ids: List[str],
    test_basin_ids: List[str],
    model_name: str
):
    """
    Visualizes the geographical distribution of training and testing basins.

    Args:
        shapefile_path: Path to the basin shapefile.
        train_basin_ids: List of IDs used for training.
        test_basin_ids: List of IDs used for testing.
        model_name: Name of the model for labeling purposes.
    """
    gdf_all_basins = gpd.read_file(shapefile_path)
    id_col = 'gauge_id'
    
    is_train = gdf_all_basins[id_col].isin(train_basin_ids)
    only_test_basin_ids = set(test_basin_ids) - set(train_basin_ids)
    is_test = gdf_all_basins[id_col].isin(only_test_basin_ids)
    
    gdf_all_basins['dataset'] = 'Not Used'
    gdf_all_basins.loc[is_test, 'dataset'] = 'Test'
    gdf_all_basins.loc[is_train, 'dataset'] = 'Train'
    
    dataset_colors = {
        'Train': 'purple',
        'Test': 'orange',
        'Not Used': 'lightgrey'
    }

    plot_colored_shapefile(
        gdf=gdf_all_basins,
        column='dataset',
        title=f"Train & Test Basin Sets: {model_name}",
        colors=dataset_colors,
    )


# --- Metrics and Evaluation ---

DA_SUFFIX = '_data_assimilation'


def load_test_results(run_dir: str, suffix: str = '') -> Tuple[xr.Dataset, int]:
    """
    Finds and loads the test results from the latest available epoch.

    Args:
        run_dir: The model run directory.
        suffix: File-stem suffix of the results to load. The evaluation writes
            regular forecasts to `test_results.zarr` (suffix `''`) and data
            assimilation forecasts to `test_results_data_assimilation.zarr`
            (suffix `DA_SUFFIX`), side by side in the same epoch directory.

    Returns:
        A tuple containing (xarray_dataset, epoch_number).
    """
    search_pattern = os.path.join(run_dir, 'test', 'model_epoch*', f'test_results{suffix}.zarr')
    result_files = glob.glob(search_pattern)
    if not result_files:
        raise FileNotFoundError(
            f"No results found matching {search_pattern}. "
            + ("Run `run infer --assimilate` first." if suffix else "Run `run infer` first.")
        )

    # Extract epoch numbers and find the max
    def get_epoch(path):
        match = re.search(r'model_epoch(\d+)', path)
        return int(match.group(1)) if match else -1

    latest_epoch_path = max(result_files, key=get_epoch)
    epoch_number = get_epoch(latest_epoch_path)
    
    data = xr.open_zarr(latest_epoch_path, consolidated=False)
    return data, epoch_number


def calculate_metrics_for_run(
    sim_data: xr.DataArray,
    obs_data: xr.DataArray
) -> pd.DataFrame:
    """
    Calculates hydrological metrics for each basin and lead time.

    Args:
        sim_data: Simulated streamflow (dims: basin, time_step, date).
        obs_data: Observed streamflow (dims: basin, date).

    Returns:
        A pandas DataFrame with MultiIndex ['basin_id', 'lead_time'].
    """
    all_metrics_results = []
    common_gauges = list(set(sim_data['basin'].values) & set(obs_data['basin'].values))
    metrics_list = metrics.get_available_metrics()

    for gauge_id in tqdm(common_gauges, desc="Processing Gauges"):
        sim_gauge = sim_data.sel(basin=gauge_id, freq='1D').load()
        obs_gauge = obs_data.sel(basin=gauge_id, freq='1D').load()

        lead_times = sim_gauge['time_step'].values

        for lt in lead_times:
            sim_slice = sim_gauge.sel(time_step=lt)
            obs_slice = obs_gauge.sel(time_step=lt)

            calc_res = metrics.calculate_metrics(
                obs=obs_slice,
                sim=sim_slice,
                metrics=metrics_list,
                resolution="1D",
                datetime_coord="date"
            )
            
            if isinstance(calc_res, pd.Series):
                df_metrics = calc_res.to_frame().T
            elif isinstance(calc_res, dict):
                df_metrics = pd.DataFrame([calc_res])
            else:
                df_metrics = calc_res

            df_metrics['basin_id'] = gauge_id
            df_metrics['lead_time'] = lt
            all_metrics_results.append(df_metrics)

    all_basins_df = pd.concat(all_metrics_results, ignore_index=True)
    return all_basins_df.set_index(['basin_id', 'lead_time'])


def load_data_and_metrics(
    model_run_dir: str,
    test_basin_ids: Set[str],
    calculate_statistics: bool = False,
    model_name: str = 'Model',
    suffix: str = ''
) -> Tuple[xr.Dataset, Optional[pd.DataFrame]]:
    """
    Loads test results and associated metrics, calculating them if necessary.

    Args:
        model_run_dir: Path to the model directory.
        test_basin_ids: Set of IDs to filter for.
        calculate_statistics: If True, forces recalculation of metrics.
        model_name: Label for printing progress.
        suffix: Results suffix, see `load_test_results`. Use `DA_SUFFIX` to load
            data assimilation results; the metrics cache gets the same suffix.

    Returns:
        A tuple of (xarray_dataset, metrics_dataframe).
    """
    print(f"Loading {model_name} results from: {model_run_dir} ...", end='')
    model_data, _ = load_test_results(model_run_dir, suffix=suffix)
    print(" simulations loaded successfully.")

    metrics_file_path = os.path.join(model_run_dir, 'test', f'precalculated_metrics{suffix}.csv')

    if calculate_statistics or not os.path.exists(metrics_file_path):
        print(f"Calculating metrics for: {model_name} ...", end='')
        model_metrics = calculate_metrics_for_run(
            model_data['streamflow_sim'], 
            model_data['streamflow_obs']
        )
        os.makedirs(os.path.dirname(metrics_file_path), exist_ok=True)
        model_metrics.to_csv(metrics_file_path)
    else:
        model_metrics = pd.read_csv(metrics_file_path)
        if {'basin_id', 'lead_time'}.issubset(model_metrics.columns):
            model_metrics = model_metrics.set_index(['basin_id', 'lead_time'])

    return model_data, model_metrics


def plot_lead_time_zero_scores(
    metrics_df: pd.DataFrame,
    train_basin_ids: Set[str],
    test_basin_ids: Set[str],
    metric_name: str,
    model_name: str = 'Model'
):
    """
    Plots the distribution of a metric at lead time 0 across basins.

    Args:
        metrics_df: DataFrame with model performance metrics.
        train_basin_ids: Set of IDs used during training.
        test_basin_ids: Set of IDs used during testing.
        metric_name: The metric column to plot (e.g., 'NSE').
        model_name: Label for the plot title.
    """
    SKILL_LEAD_TIME = 0

    if 'lead_time' in metrics_df.index.names:
        scores_lt0 = metrics_df.xs(SKILL_LEAD_TIME, level='lead_time')[metric_name]
    else:
        scores_lt0 = metrics_df[metric_name]

    scores_df = scores_lt0.reset_index()
    scores_df['basin_type'] = scores_df['basin_id'].apply(
        lambda x: 'Train' if x in train_basin_ids else ('Test' if x in test_basin_ids else 'Other')
    )
    scores_df = scores_df.sort_values(by=metric_name, ascending=True)

    plt.close('all')
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = {'Train': 'purple', 'Test': 'orange', 'Other': 'gray'}

    for basin_type, group in scores_df.groupby('basin_type'):
        ax.barh(
            group['basin_id'],
            group[metric_name],
            color=colors.get(basin_type, 'gray'),
            label=basin_type
        )

    ax.set_yticks([])
    ax.invert_yaxis()
    ax.set_xlim([-0.5, 1])
    ax.set_xlabel(f"{metric_name} Score (Lead Time {SKILL_LEAD_TIME})")
    ax.set_ylabel("Basins (sorted)")
    ax.set_title(f"{model_name} Performance: {metric_name}")
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    ax.legend()
    
    plt.tight_layout()
    plt.show()


def plot_metrics_vs_lead_time(
    metrics_by_model: Dict[str, pd.DataFrame],
    basin_id: str,
    metric_name: str,
    title: Optional[str] = None,
):
    """
    Compares the performance of several models across all lead times for one basin.

    Args:
        metrics_by_model: Mapping from a display name (e.g. 'Base Model',
            'Data Assimilation') to a metrics DataFrame indexed by
            ['basin_id', 'lead_time'] as returned by `load_data_and_metrics`.
        basin_id: Specific basin to analyze.
        metric_name: Metric to compare (e.g., 'KGE').
        title: Optional plot title.
    """
    plt.figure(figsize=(10, 6))
    linestyles = ['-', '--', '-.', ':']

    for i, (name, df) in enumerate(metrics_by_model.items()):
        if df is None or basin_id not in df.index.get_level_values('basin_id'):
            continue
        basin_metrics = df.loc[basin_id, metric_name]
        plt.plot(
            basin_metrics.index, basin_metrics.values,
            marker='o', linestyle=linestyles[i % len(linestyles)], label=name,
        )

    plt.title(title or f"{metric_name} vs. Lead Time for Basin {basin_id}")
    plt.xlabel("Lead Time (days)")
    plt.ylabel(f"{metric_name} Score")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.show()


def plot_comparison_metrics_vs_lead_time(
    base_metrics_df: pd.DataFrame,
    finetune_metrics_df: Optional[pd.DataFrame],
    basin_id: str,
    metric_name: str
):
    """
    Compares Base vs Fine-tuned model performance across all lead times.

    Args:
        base_metrics_df: Metrics for the base model.
        finetune_metrics_df: Metrics for the fine-tuned model (optional).
        basin_id: Specific basin to analyze.
        metric_name: Metric to compare (e.g., 'KGE').
    """
    plot_metrics_vs_lead_time(
        {'Base Model': base_metrics_df, 'Fine-Tuned Model': finetune_metrics_df},
        basin_id=basin_id,
        metric_name=metric_name,
    )


def plot_hydrograph_comparison(
    observed: xr.DataArray,
    simulations: Dict[str, xr.DataArray],
    basin_id: str,
    lead_times: Union[int, List[int]] = 0,
    date_range: Optional[Tuple[str, str]] = None,
    title: Optional[str] = None,
):
    """
    Plots observed vs. simulated hydrographs of several models, one panel per lead time.

    Args:
        observed: Observed streamflow with dims (basin, freq, date, time_step), e.g.
            `data['streamflow_obs']` of any loaded results dataset.
        simulations: Mapping from a display name to a simulated streamflow DataArray
            with the same dims as `observed` (e.g. base model, fine-tuned model,
            data assimilation).
        basin_id: Basin to plot.
        lead_times: One or several `time_step` values. 0 is the issue day (nowcast),
            1..7 are the forecast days ahead.
        date_range: Optional ('YYYY-MM-DD', 'YYYY-MM-DD') zoom window.
        title: Optional figure title.
    """
    if isinstance(lead_times, int):
        lead_times = [lead_times]

    fig, axes = plt.subplots(len(lead_times), 1, figsize=(12, 4 * len(lead_times)), sharex=True, squeeze=False)
    colors = ['tab:blue', 'tab:red', 'tab:green', 'tab:purple', 'tab:orange']
    linestyles = ['--', '-.', ':', '-', '--']

    for ax, lt in zip(axes[:, 0], lead_times):
        obs = observed.sel(basin=basin_id, freq='1D', time_step=lt)
        if date_range is not None:
            obs = obs.sel(date=slice(*date_range))
        ax.plot(obs['date'], obs.values, label='Observed', color='black', linewidth=1.5)

        for i, (name, sim) in enumerate(simulations.items()):
            s = sim.sel(basin=basin_id, freq='1D', time_step=lt)
            if date_range is not None:
                s = s.sel(date=slice(*date_range))
            ax.plot(s['date'], s.values, label=name, color=colors[i % len(colors)],
                    linestyle=linestyles[i % len(linestyles)], linewidth=1.2)

        ax.set_ylabel("Streamflow")
        ax.set_title(f"Lead time {lt} day{'s' if lt != 1 else ''}" + (" (issue day)" if lt == 0 else ""))
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc='upper right')

    axes[-1, 0].set_xlabel("Date")
    fig.suptitle(title or f"Hydrograph Comparison for Basin {basin_id}")
    fig.tight_layout()
    plt.show()


# --- Configuration Generation ---

def replace_placeholders(data: Any, basin_id: Union[str, int]) -> Any:
    """
    Recursively replaces 'FINETUNE_BASIN' placeholder in nested dicts/lists.

    Args:
        data: The configuration data structure.
        basin_id: The ID to substitute.

    Returns:
        The updated data structure.
    """
    if isinstance(data, dict):
        return {k: replace_placeholders(v, basin_id) for k, v in data.items()}
    elif isinstance(data, list):
        return [replace_placeholders(elem, basin_id) for elem in data]
    elif isinstance(data, str):
        return data.replace('FINETUNE_BASIN', str(basin_id))
    return data


def create_basin_list_file(basin_id: str, output_dir: str = 'basin-lists') -> str:
    """
    Generates a text file containing a single basin ID for fine-tuning.

    Args:
        basin_id: The basin ID.
        output_dir: Target directory.

    Returns:
        The path to the created file.
    """
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"{basin_id}.txt")
    
    with open(file_path, 'w') as f:
        f.write(f"{basin_id}\n")
    
    print(f"Created basin list file at: {file_path}")
    return file_path


def generate_basin_finetune_config(
    template_path: str, 
    basin_id: str, 
    base_model_dir: str, 
    output_path: str
):
    """
    Creates a specific YAML config for fine-tuning based on a template.

    Args:
        template_path: Path to the template configuration file.
        basin_id: The target basin ID for fine-tuning.
        base_model_dir: Directory of the pre-trained weights.
        output_path: Path to save the new configuration.
    """
    print(f"\nGenerating fine-tuning configuration for basin: {basin_id}")
    
    with open(template_path, 'r') as f:
        config_data = yaml.safe_load(f)
    
    # Apply placeholders
    config_data = replace_placeholders(config_data, basin_id)
    
    # Set paths
    config_data['base_run_dir'] = base_model_dir
    config_data['run_dir'] = base_model_dir

    with open(output_path, 'w') as f:
        yaml.dump(config_data, f, default_flow_style=False)
    
    print(f"Configuration saved to: {output_path}")



def generate_assimilation_config(
    run_dir: str,
    template_path: str,
    output_path: str,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Creates a run configuration with data assimilation enabled for an existing model run.

    Data assimilation does not train anything: it re-uses the trained weights of
    `run_dir` and only needs the extra `assimilate` / `assimilation_config` keys.
    Those keys are read from `template_path` (e.g. `configs/assimilation-config.yml`)
    and merged into a copy of the run's own `config.yml`. The result is a complete
    run configuration that can be passed to the evaluation together with the run
    directory:

        run infer --run-dir <run_dir> --config-file <output_path> --assimilate

    The run's `config.yml` is never modified.

    Args:
        run_dir: Directory of the trained model (contains `config.yml` and weights).
        template_path: YAML file holding the `assimilate` and `assimilation_config` keys.
        output_path: Where to write the merged configuration.
        overrides: Optional keys to set inside `assimilation_config` (e.g.
            `{'assimilation_window': 7, 'regularization_weight': 0.1}`), useful for
            trying out different settings without editing the template.

    Returns:
        The merged configuration dictionary.
    """
    print(f"\nGenerating data assimilation configuration for run: {run_dir}")

    with open(os.path.join(run_dir, 'config.yml'), 'r') as f:
        run_config = yaml.safe_load(f)
    with open(template_path, 'r') as f:
        da_template = yaml.safe_load(f)

    if 'assimilation_config' not in da_template:
        raise ValueError(f"{template_path} does not define an 'assimilation_config' block.")

    # Paths inside a shipped run directory may point to the machine the model was
    # trained on; point them to the local tutorial folder instead.
    config_data = rebase_config_paths(run_config)
    config_data['run_dir'] = os.path.abspath(run_dir)
    config_data['img_log_dir'] = os.path.join(os.path.abspath(run_dir), 'img_log')

    # Overlay the data assimilation keys.
    config_data.update(da_template)
    config_data['assimilation_config'] = dict(da_template['assimilation_config'])
    for key, value in (overrides or {}).items():
        config_data['assimilation_config'][key] = value

    # The stored run config has the training-time dataloader settings; data
    # assimilation runs on the CPU in the tutorial, so keep things simple.
    config_data.setdefault('num_workers', 0)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)

    print("Assimilation settings:")
    for key, value in config_data['assimilation_config'].items():
        print(f"  {key}: {value}")
    print(f"Configuration saved to: {output_path}")
    return config_data


def variant_suffix(variant: str) -> str:
    """Returns the results suffix under which a named DA variant is stored."""
    return f'{DA_SUFFIX}_{variant}'


def archive_assimilation_results(run_dir: str, variant: str) -> None:
    """
    Renames the latest data assimilation outputs of a run to a named variant.

    The evaluation always writes `test_results_data_assimilation.zarr` and
    `test_metrics_data_assimilation.csv`. To keep the outputs of several DA
    settings side by side, this moves them to
    `test_results_data_assimilation_<variant>.zarr` (and likewise for the csv)
    so they can be loaded with `load_data_and_metrics(..., suffix=variant_suffix(variant))`.

    Args:
        run_dir: The model run directory.
        variant: Short name of the DA setting (e.g. 'window7').
    """
    epoch_dirs = glob.glob(os.path.join(run_dir, 'test', 'model_epoch*'))
    for epoch_dir in epoch_dirs:
        for stem, ext in [('test_results', '.zarr'), ('test_metrics', '.csv')]:
            src = os.path.join(epoch_dir, f'{stem}{DA_SUFFIX}{ext}')
            dst = os.path.join(epoch_dir, f'{stem}{variant_suffix(variant)}{ext}')
            if os.path.exists(src):
                if os.path.isdir(dst):
                    shutil.rmtree(dst)
                elif os.path.exists(dst):
                    os.remove(dst)
                shutil.move(src, dst)
                print(f"Archived {os.path.basename(src)} -> {os.path.basename(dst)}")
    # The metrics cache of the generic suffix no longer matches any results.
    cache = os.path.join(run_dir, 'test', f'precalculated_metrics{DA_SUFFIX}.csv')
    if os.path.exists(cache):
        os.remove(cache)


def median_metric_vs_lead_time(
    metrics_by_model: Dict[str, pd.DataFrame],
    metric_name: str,
) -> pd.DataFrame:
    """
    Tabulates the median of a metric across basins per lead time, one column per model.

    Args:
        metrics_by_model: Mapping from display name to a metrics DataFrame indexed
            by ['basin_id', 'lead_time'].
        metric_name: Metric to summarise (e.g. 'NSE').

    Returns:
        A DataFrame indexed by lead time with one column per model.
    """
    columns = {
        name: df[metric_name].groupby(level='lead_time').median()
        for name, df in metrics_by_model.items()
        if df is not None
    }
    return pd.DataFrame(columns)
