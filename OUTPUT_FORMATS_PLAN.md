# Output File Formats Architecture & Unification Plan

## 1. Executive Summary & Recommendation

### What We Currently Use for Outputs

| Output Category | Current Format | Location / Filename | Purpose |
| :--- | :--- | :--- | :--- |
| **Prediction Time Series** | Zarr (partial) | `{run_dir}/{period}/model_epochXXX/{period}_results.zarr` | Simulated streamflow $\hat{y}$, observations $y$, and multi-lead forecast tensors. |
| **Evaluation Metrics** | CSV | `{run_dir}/{period}/model_epochXXX/{period}_metrics.csv` | Tabular performance statistics (NSE, KGE, Pearson-$r$, peak timing, etc.). |
| **Feature Scaler** | Zarr (with NetCDF fallback) | `{run_dir}/scaler.zarr` | Feature normalization centers, scales, and zero thresholds. |
| **Model Weights & Optimizer** | PyTorch (`.pt`) | `{run_dir}/model_epochXXX.pt`, `optimizer_state_epochXXX.pt` | Neural network state dictionaries. |
| **Run Configurations** | YAML | `{run_dir}/config.yml` | Full hyperparameter configuration dump. |
| **Logs & Visualizations** | TensorBoard / PNG / Text | `{run_dir}/output.log`, `events.out.tfevents.*`, `img_log/` | Training curves and hydrograph plots. |

---

## 2. Professional Opinion & Recommendations

### Should We Switch Outputs to Zarr?

**Yes for Structured Time Series Data; Keep Native Formats for Weights and Configs.**

1. **Prediction Time Series (`{period}_results.zarr`): Upgrade to Full First-Class Zarr Support**
   - **Current issue:** Currently, `results.zarr` is only saved if `cfg.inference_mode` is enabled and `period == 'test'`. Running standard `run evaluate` or validation evaluation does not write the structured Zarr store by default unless special flags are set.
   - **Recommendation:** Standardize time series output completely on Zarr. Enable saving `test_results.zarr` whenever `save_results: true` across all evaluation and inference modes.
   - **Consolidated Metadata:** When appending per-basin chunks during evaluation (`append_dim='basin'`), automatically run `zarr.consolidate_metadata(result_file)` at the end of the evaluation loop so users can open the results instantly without metadata warnings.

2. **Evaluation Metrics (`test_metrics.csv`): Keep CSV, Optionally Mirror in Zarr**
   - **Recommendation:** Keep `test_metrics.csv` as a primary output.
   - **Rationale:** Summary metrics across basins (e.g. 615 rows $\times$ 15 columns, ~30 KB) are purely tabular. CSV is immediately human-readable, opens directly in spreadsheets and lightweight pandas scripts, and is universally expected in hydrological benchmarking. We can additionally embed summary statistics into the Zarr dataset attributes (`ds.attrs['median_nse'] = ...`).

3. **Model Weights & Optimizer States (`.pt`): Keep PyTorch `.pt`**
   - **Recommendation:** Keep PyTorch native `.pt` format.
   - **Rationale:** PyTorch state dicts contain neural network layer hierarchies, tensor shapes, and optimizer momentum buffers. Zarr is designed for multidimensional scientific arrays, not model parameter graphs.

4. **Scaler (`scaler.zarr`): Completed**
   - We already upgraded the normalizer to `scaler.zarr`.

---

## 3. Implementation Plan (When Ready to Execute)

### Step 1: Modernize `tester.py` Output Pipeline
- Remove legacy conditions requiring `inference_mode` for Zarr saving.
- Enable `save_results: true` in `run evaluate` to output `{period}_results.zarr`.
- Add post-evaluation metadata consolidation (`zarr.consolidate_metadata`).
- Clean up outdated docstrings in `tester.py` referring to legacy pickle wrappers.

### Step 2: Coordinate & Attribute Standardization in Output Zarr
- Ensure output dataset dimensions and coordinates match the input conventions:
  - Dimensions: `(basin, date, lead_time, freq, [samples])`
  - Attributes: Attach run timestamp, model architecture, and commit hash to `ds.attrs`.

### Step 3: Backward Compatibility & Visualization Tools
- Ensure plotting utilities (`plots.py`, `backend.py`, tutorial notebooks) seamlessly read `{period}_results.zarr` using `xr.open_zarr()`.
- Add unit tests verifying output Zarr creation and integrity during test runs.

---

## 4. Summary Table of Proposed Output Standards

```
run_dir/
  ├── config.yml                  (YAML - human readable config)
  ├── scaler.zarr/                (Zarr - feature normalization parameters)
  ├── model_epoch055.pt           (PyTorch - neural network state dict)
  ├── output.log                  (Text - training & evaluation logs)
  └── test/
        └── model_epoch055/
              ├── test_metrics.csv      (CSV - basin-by-basin performance scores)
              └── test_results.zarr/    (Zarr - consolidated simulated & observed time series)
```
