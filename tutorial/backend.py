# Standard Library Imports
import glob
import os
import pickle
import re
import sys
import yaml
from typing import Any, Dict, List, Optional, Tuple, Set

# Third-Party Library Imports
import geopandas as gpd
import ipywidgets as widgets
from IPython.display import display, Markdown
from ipywidgets import interactive, VBox
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import xarray as xr

# Local tutorial Module Imports
from googlehydrology.evaluation import metrics


# --- Model Selection ---

def find_model_run_dirs(base_dir: str) -> Dict[str, str]:
    """
    Finds directories containing a 'test/model_epoch*/test_results.zarr' file.
    Only searches the immediate subdirectories of base_dir (depth of 1).
    Returns a dictionary mapping display names to directory paths.
    """
    run_dirs = {}
    
    # Get immediate subdirectories of base_dir
    # This replaces os.walk to ensure we don't go deeper than one level
    subdirs = [
        d for d in os.listdir(base_dir) 
        if os.path.isdir(os.path.join(base_dir, d))
    ]

    for subdir in subdirs:
        root = os.path.join(base_dir, subdir)
        
        # Check if a 'test' subdirectory exists within the current root
        test_dir = os.path.join(root, 'test')
        if os.path.exists(test_dir) and os.path.isdir(test_dir):
            
            # Check for subdirectories named 'model_epoch*' within the 'test' directory
            # Using glob to match the pattern
            epoch_dirs = glob.glob(os.path.join(test_dir, 'model_epoch*'))
            
            for epoch_dir in epoch_dirs:
                # Check if 'test_results.zarr' directory/file exists within the epoch directory
                if os.path.isdir(os.path.join(epoch_dir, 'test_results.zarr')):
                    # Create a user-friendly display name (the name of the subdirectory)
                    display_name = subdir
                    run_dirs[display_name] = root
                    # Stop checking other epochs once a valid one is found for this run
                    break 
                    
    return run_dirs


def read_basin_list(file_path: str) -> Set[str]:
    """Reads a basin list file and returns a set of normalized basin IDs."""
    with open(file_path, 'r') as f:
        ids = [normalize_id(line.strip()) for line in f if line.strip()]
    return set(ids)

def normalize_id(basin_id: str) -> str:
    """Converts basin IDs to a standard string format for comparison."""
    str_id = str(basin_id).split('_')[-1]
    return str_id.zfill(8)

def load_model_config_and_basins(run_dir: str) -> Tuple[Dict[str, Any], Set[str], Set[str]]:
    """Loads the base model config and associated train/test basin lists."""
    config_path = os.path.join(run_dir, 'config.yml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    train_basin_file = config['train_basin_file']
    test_basin_file = config['test_basin_file']

    train_basin_ids = read_basin_list(train_basin_file)
    test_basin_ids = read_basin_list(test_basin_file)

    print(f"Loaded {len(train_basin_ids)} training basin IDs.")
    print(f"Loaded {len(test_basin_ids)} test basin IDs.")

    return config, train_basin_ids, test_basin_ids


def plot_colored_shapefile(
    gdf: gpd.GeoDataFrame,
    column: str,
    title: str,
    cmap: Optional[str] = None,
    colors: Optional[Dict[Any, str]] = None,
    figsize: Tuple[int, int] = (12, 12),
    missing_kwds: Optional[Dict[str, Any]] = None
):
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
    gdf_all_basins = gpd.read_file(shapefile_path)
    shapefile_basin_id_column = 'gauge_id'
    gdf_all_basins['normalized_id'] = gdf_all_basins[shapefile_basin_id_column].apply(normalize_id)
    
    is_train = gdf_all_basins['normalized_id'].isin(train_basin_ids)
    only_test_basin_ids = set(test_basin_ids) - set(train_basin_ids)
    is_test = gdf_all_basins['normalized_id'].isin(only_test_basin_ids)
    
    gdf_all_basins['dataset'] = 'Not Used'
    gdf_all_basins.loc[is_test, 'dataset'] = 'Test'
    gdf_all_basins.loc[is_train, 'dataset'] = 'Train'
    
    # Using purple and orange as requested
    dataset_colors = {
        'Train': 'purple',
        'Test': 'orange',
        'Not Used': 'lightgrey'
    }

    plot_colored_shapefile(
        gdf=gdf_all_basins,
        column='dataset',
        title=f"Train & Test Basin Sets from Base Model",
        colors=dataset_colors,
    )


def load_test_results(run_dir: str) -> Tuple[Dict[str, Any], int]:
    search_path = os.path.join(run_dir, 'test', 'model_epoch*', 'test_results.zarr')
    result_files = glob.glob(search_path)

    latest_epoch_path = max(
        result_files,
        key=lambda path: int(re.search(r'model_epoch(\d+)', path).group(1))
        if re.search(r'model_epoch(\d+)', path) else -1
    )

    match = re.search(r'model_epoch(\d+)', latest_epoch_path)
    epoch_number = int(match.group(1)) if match else 0
    data = xr.open_zarr(latest_epoch_path, consolidated=False)
    return data, epoch_number


def calculate_metrics_for_run(
    sim_data: xr.Dataset,
    obs_data: xr.Dataset
) -> pd.DataFrame:
    """
    Calculates metrics for each gauge and each lead time.

    Assumes sim_data has dimensions ['basin', 'lead_time', 'date']
    and obs_data has dimensions ['basin', 'date'].
    """
    all_metrics_results = []

    # Identify common basins/gauges
    common_gauges = list(set(sim_data['basin'].values) & set(obs_data['basin'].values))
    metrics_list = metrics.get_available_metrics()

    for gauge_id in tqdm(common_gauges, desc="Processing Gauges"):
        # Select data for the specific gauge and frequency
        # Note: We keep the objects as xarray objects to access coordinates easily
        sim_gauge = sim_data.sel(basin=gauge_id, freq='1D').load()
        obs_gauge = obs_data.sel(basin=gauge_id, freq='1D').load()

        # Identify lead times available for this specific simulation data
        # We iterate over lead_time because calculate_metrics expects 1-D arrays
        lead_times = sim_gauge['time_step'].values

        for lt in lead_times:
            # Slice the simulation to a 1-D array for the specific lead time
            # FIX: Add .load() to convert Dask arrays to Numpy arrays to avoid indexing errors
            sim_slice = sim_gauge.sel(time_step=lt)
            obs_slice = obs_gauge.sel(time_step=lt)

            # The metrics function expects 1-D arrays.
            # We pass the sim_slice (which is now 1D over 'date') and the obs_gauge.
            # Depending on your 'metrics' implementation, you might need .to_array()
            # or .values, but usually xarray-aware functions handle DataArrays.
            try:
                calculated_metrics = metrics.calculate_metrics(
                    obs=obs_slice,
                    sim=sim_slice,
                    metrics=metrics_list,
                    resolution="1D",
                    datetime_coord="date"
                )
            except:
                calculated_metrics = {metric: np.nan for metric in metrics_list}

            # Convert result to DataFrame if it's a Series or dict and add identifiers
            if isinstance(calculated_metrics, pd.Series):
                df_metrics = calculated_metrics.to_frame().T
            elif isinstance(calculated_metrics, dict):
                # Convert dictionary to a DataFrame, assuming it represents a single row of metrics
                df_metrics = pd.DataFrame([calculated_metrics])
            else:
                df_metrics = calculated_metrics

            # Add index information to allow for easy concatenation
            df_metrics['basin_id'] = gauge_id
            df_metrics['lead_time'] = lt

            all_metrics_results.append(df_metrics)

    # Combine all individual slices into one large DataFrame
    if not all_metrics_results:
        return pd.DataFrame()

    all_basins_df = pd.concat(all_metrics_results, ignore_index=True)

    # Set the multi-index to match the requested structure
    all_basins_df = all_basins_df.set_index(['basin_id', 'lead_time'])

    return all_basins_df


def load_data_and_metrics(
    model_run_dir: str,
    test_basin_ids: Set[str],
    calculate_statistics: bool = False,
    model_name: str = 'Model'
) -> Tuple[Dict[str, Any], Optional[pd.DataFrame]]:
    print(f"Loading {model_name} results from: {model_run_dir} ...", end='')
    model_data, model_epoch = load_test_results(model_run_dir)
    print(f" simulations loaded successfully.")

    model_metrics = None
    metrics_file_path = os.path.join(model_run_dir, 'test', 'precalculated_metrics.csv')

    if calculate_statistics or not os.path.exists(metrics_file_path):
        print(f"Calculating metrics for: {model_name} ...", end='')
        model_metrics = calculate_metrics_for_run(model_data['streamflow_sim'], model_data['streamflow_obs'])
        os.makedirs(os.path.dirname(metrics_file_path), exist_ok=True)
        model_metrics.to_csv(metrics_file_path)
    else:
        model_metrics = pd.read_csv(metrics_file_path)
        if 'basin_id' in model_metrics.columns and 'lead_time' in model_metrics.columns:
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
    Plots the distribution of a specified metric at lead time 0.
    Updated with Purple (Train) and Orange (Test) colors.
    """
    SKILL_LEAD_TIME = 0

    # Ensure correct indexing and filter data
    try:
        scores_lt0 = metrics_df.xs(SKILL_LEAD_TIME, level='lead_time')[metric_name]
    except KeyError:
        # Fallback if lead_time is not in index (already filtered)
        scores_lt0 = metrics_df[metric_name]

    scores_df = scores_lt0.reset_index()

    # Call local normalize_id function
    scores_df['basin_id'] = scores_df['basin_id'].apply(normalize_id)
    train_norm = set(normalize_id(bid) for bid in train_basin_ids)
    test_norm = set(normalize_id(bid) for bid in test_basin_ids)

    # Categorize
    scores_df['basin_type'] = scores_df['basin_id'].apply(
        lambda x: 'Train' if x in train_norm else ('Test' if x in test_norm else 'Other')
    )

    scores_df = scores_df.sort_values(by=metric_name, ascending=True)

    # Plotting Logic
    plt.close('all')
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # New Colors: Purple and Orange
    colors = {'Train': 'purple', 'Test': 'orange', 'Other': 'gray'}

    for basin_type, group in scores_df.groupby('basin_type'):
        ax.barh(
            group['basin_id'],
            group[metric_name],
            color=colors.get(basin_type, 'gray'),
            label=basin_type
        )

    # Aesthetic adjustments
    max_score = scores_df[metric_name].max()
    ax.set_xlim(right=min([max_score * 1.05, 2]))
    abs_min_score = abs(scores_df[metric_name].min())
    ax.set_xlim(left=max([-(abs_min_score * 0.95), -1]))
    
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
    finetune_metrics_df: Optional[pd.DataFrame], # Fine-tuned metrics are optional
    basin_id: str,
    metric_name: str
):
    """
    Plots the selected metric vs. lead time for the base and fine-tuned models
    for a specific basin.

    This function is crucial for visually comparing how a model's performance (skill)
    changes with increasing lead time, and how fine-tuning impacts this performance
    relative to the base model for a chosen basin. It helps assess whether fine-tuning
    improves predictive skill for the target basin.

    Args:
        base_metrics_df (pd.DataFrame): DataFrame containing metrics for the base model.
                                       Expected to have a MultiIndex including 'basin_id' and 'lead_time'.
        finetune_metrics_df (Optional[pd.DataFrame]): DataFrame containing metrics for the fine-tuned model.
                                                      Expected to have a MultiIndex including 'basin_id' and 'lead_time'.
                                                      Can be None if no fine-tuned model is selected or available.
        basin_id (str): The specific basin ID for which to plot the metrics.
        metric_name (str): The name of the metric column (e.g., 'KGE', 'NSE') to plot.
    """
    plt.figure(figsize=(10, 6))

    # Plot base model metrics for the selected basin
    # We select the row corresponding to the specific basin_id and then the column for the metric_name.
    # The index of this Series will be the lead_times.
    base_basin_metrics = base_metrics_df.loc[basin_id, metric_name]
    plt.plot(base_basin_metrics.index, base_basin_metrics.values, marker='o', label='Base Model')


    # Plot fine-tuned model metrics for the selected basin if available
    # Similar to the base model, we select the metrics for the fine-tuned model.
    # This line uses a different linestyle to distinguish it from the base model.
    finetune_basin_metrics = finetune_metrics_df.loc[basin_id, metric_name]
    plt.plot(finetune_basin_metrics.index, finetune_basin_metrics.values, marker='o', linestyle='--', label='Fine-Tuned Model')


    plt.title(f"{metric_name} vs. Lead Time for Basin {basin_id}")
    plt.xlabel("Lead Time (days)")
    plt.ylabel(f"{metric_name} Score")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend() # Display the legend to differentiate between models
    plt.xticks(base_metrics_df.index.get_level_values('lead_time').unique()) # Ensure lead times are shown as ticks
    plt.show()