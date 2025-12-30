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
generating fine-tuning configurations.
"""

# Standard Library Imports
import glob
import os
import re
import yaml
from typing import Any, Dict, List, Optional, Tuple, Set, Union

# Third-Party Library Imports
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from tqdm.notebook import tqdm

# Local Module Imports
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

    train_basin_ids = read_basin_list(config.get('train_basin_file', ''))
    test_basin_ids = read_basin_list(config.get('test_basin_file', ''))

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

def load_test_results(run_dir: str) -> Tuple[xr.Dataset, int]:
    """
    Finds and loads the test results from the latest available epoch.

    Args:
        run_dir: The model run directory.

    Returns:
        A tuple containing (xarray_dataset, epoch_number).
    """
    search_pattern = os.path.join(run_dir, 'test', 'model_epoch*', 'test_results.zarr')
    result_files = glob.glob(search_pattern)

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
    model_name: str = 'Model'
) -> Tuple[xr.Dataset, Optional[pd.DataFrame]]:
    """
    Loads test results and associated metrics, calculating them if necessary.

    Args:
        model_run_dir: Path to the model directory.
        test_basin_ids: Set of IDs to filter for.
        calculate_statistics: If True, forces recalculation of metrics.
        model_name: Label for printing progress.

    Returns:
        A tuple of (xarray_dataset, metrics_dataframe).
    """
    print(f"Loading {model_name} results from: {model_run_dir} ...", end='')
    model_data, _ = load_test_results(model_run_dir)
    print(" simulations loaded successfully.")

    metrics_file_path = os.path.join(model_run_dir, 'test', 'precalculated_metrics.csv')

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
    ax.set_xlabel(f"{metric_name} Score (Lead Time {SKILL_LEAD_TIME})")
    ax.set_ylabel("Basins (sorted)")
    ax.set_title(f"{model_name} Performance: {metric_name}")
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    ax.legend()
    
    plt.tight_layout()
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
    plt.figure(figsize=(10, 6))

    base_basin_metrics = base_metrics_df.loc[basin_id, metric_name]
    plt.plot(base_basin_metrics.index, base_basin_metrics.values, marker='o', label='Base Model')

    if finetune_metrics_df is not None:
        ft_basin_metrics = finetune_metrics_df.loc[basin_id, metric_name]
        plt.plot(ft_basin_metrics.index, ft_basin_metrics.values, 
                 marker='o', linestyle='--', label='Fine-Tuned Model')

    plt.title(f"{metric_name} vs. Lead Time for Basin {basin_id}")
    plt.xlabel("Lead Time (days)")
    plt.ylabel(f"{metric_name} Score")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
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