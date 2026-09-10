# **OpenHydroNet: Riverine Flood Forecasting**

## **🌊 This repository implements the state-of-the-art models that power [Google FloodHub](https://sites.research.google/floods/).**

This is not an officially supported Google product. This project is not eligible for the Google Open Source Software Vulnerability Rewards Program.

The repository provides open-source replication of Google’s global flood-forecasting models. By open-sourcing these models, we aim to foster transparency, enable in-house integration in production systems, and accelerate academic research.

This repository is a fork of [NeuralHydrology](https://github.com/neuralhydrology/neuralhydrology), which has been heavily modified and extended to support forecast sequences using the specific model architectures that are used operationally in the Google FloodHub.

## 📖 Documentation

Detailed instructions on how to configure, train, and evaluate OpenHydroNet models can be found on our official documentation page:
👉 **[openhydronet.readthedocs.io](https://openhydronet.readthedocs.io/)**

Watch our high-level video introduction to the interactive tutorial on YouTube:
[OpenHydroNet Tutorial Video](https://www.youtube.com/watch?v=431Kr3mxidU)

## **Models**

This repository contains implementations of the core models used in Google's production forecasting systems.

### **Mean-Embedding-Forecast-LSTM**

The [Mean Embedding Forecast LSTM](https://github.com/google-research/flood-forecasting/blob/main/googlehydrology/modelzoo/mean_embedding_forecast_lstm.py) is a forecasting model that uses separate embedding networks for hindcast and forecast inputs. It aggregates these inputs using masked means before passing them into respective LSTMs for the hindcast and forecast periods.

* **Status:** **Current production model** (as of December 2025\) for [Google FloodHub](https://sites.research.google/floods/).  
* **Reference:** Gauch, Martin, et al. "[How to deal with missing input data](https://hess.copernicus.org/articles/29/6221/2025/)." *Hydrology and Earth System Sciences* (2025).

### **Handoff-Forecast-LSTM**

The [State Handoff Forecast LSTM](https://github.com/google-research/flood-forecasting/blob/main/googlehydrology/modelzoo/handoff_forecast_lstm.py) is a forecasting model that uses a state-handoff to transition from a hindcast sequence (LSTM) model to a forecast sequence (LSTM) model. The hindcast model runs from the past up to the present (the issue time of the forecast) and then passes the cell state and hidden state of the LSTM into a (nonlinear) handoff network, which is used to initialize a new LSTM that rolls out over the forecast period.

* **Status:** Former production model for [Google FloodHub](https://sites.research.google/floods/).  
* **Reference:** Nearing, Grey, et al. "[Global prediction of extreme floods in ungauged watersheds](https://www.nature.com/articles/s41586-024-07145-1)." *Nature* (2024).

## **Installation**

We recommend using **Conda** to manage dependencies like PyTorch and CUDA.

1. **Create and Activate the Environment:**  


   ```
   # Create the environment from the file in the repo  
   conda env create -f environments/conda.yml

   # Activate the environment (MANDATORY)  
   conda activate googlehydrology  
   ```
    
3. Install the Package:  
   Install in editable mode so that changes to the source code are reflected immediately:  


   ```
   # Run from the root of the repository  
   pip install -e .
   ```

## **🚀 Tutorial Notebook**

The most direct way to explore this repository is through our interactive tutorial: [**OpenHydroNet Tutorial Notebook**](https://colab.research.google.com/github/google-research/flood-forecasting/blob/main/tutorial/OpenHydroNet_Tutorial.ipynb).

**What you will learn:**

* **Model Evaluation:** Load pre-trained Google Hydrology models and calculate performance metrics (NSE, KGE) on real-world basin data.  
* **Fine-Tuning for Performance:** Learn how to fine-tune the `static_embedding_fc` layer. This is a powerful technique for improving predictions on "outlier" basins (e.g., basins with unusual sizes or geology) without retraining the entire model.
* **Visualizing Results:** Compare model hydrographs against observed discharge data.

**Run it now:** 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-research/flood-forecasting/blob/main/tutorial/OpenHydroNet_Tutorial.ipynb)

## **Data Setup**

OpenHydroNet uses the [Caravan](https://www.nature.com/articles/s41597-023-01975-w) dataset for streamflow observations and static catchment attributes.

### **1\. Download Caravan (NetCDF Version)**

A small sample is provided in tutorial/data/Caravan-nc. For full runs:

1. Visit the [Zenodo repository](https://doi.org/10.5281/zenodo.6522634).  
2. Download the **NetCDF version** (Caravan-nc.tar.gz).  
3. Unpack it locally:  

   ```
   mkdir -p ~/data/  
   tar -xvzf Caravan-nc.tar.gz -C ~/data/
   ```

### **2\. MultiMet Data & Extractor**

OpenHydroNet supports the Caravan MultiMet forcing dataset, which enriches hydrological modeling with diverse meteorological nowcasts and weather forecasts.

* **Reference Paper:** Kratzert, Frederik, Martin Gauch, Grey Nearing, et al. *"Caravan MultiMet: Extending Caravan with Multiple Weather Nowcasts and Forecasts."* [arXiv:2411.09459](https://arxiv.org/abs/2411.09459) (2024).
* **Pre-extracted Benchmark Data:** Pre-computed forcing time series for standard Caravan basins are hosted on Google Cloud Storage (`gs://caravan-multimet/v1.1`) and Zenodo ([Part 1](https://zenodo.org/records/14161235), [Part 2](https://zenodo.org/records/14161281)).

#### **New: MultiMet Forcing Extractor (`multimet`)**

This repository now includes the open-source **MultiMet Extractor**, allowing researchers to extract and harmonize meteorological forcings directly from raw gridded weather products into Caravan-compliant Zarr stores.

**Supported Products (Local Serial Run):**
1. **ERA5-Land** (0.1° hourly reanalysis, 15 variables including FAO-56 Penman-Monteith PET)
2. **NOAA CPC Global Precipitation** (0.5° daily gauge-based analysis)
3. **NASA GPM IMERG Early V07** (0.1° satellite precipitation nowcast)
4. **ECMWF IFS HRES** (0.25° 10-day numerical weather prediction forecasts with daily increments)
5. **DeepMind GraphCast** (0.25° 10-day AI weather forecast accumulations)

**Key Differences from the arXiv Paper:**
* **Active Extractor Engine vs. Static Benchmark:** The arXiv paper published a static dataset covering fixed Caravan watersheds through 2023. This new extractor is the **underlying reproducible extraction pipeline**, enabling researchers to extract MultiMet-standard forcings for **any custom basin geometries** and **any time interval**.
* **Zero Proprietary Infrastructure:** The extractor operates entirely on public open-access endpoints (WeatherBench 2 on public GCS, NOAA PSL HTTP, NASA GES DISC, ECMWF Open Data) and standard scientific Python packages (`xarray`, `zarr`, `geopandas`, `scipy`), eliminating reliance on Google-internal compute infrastructure.
* **Vectorized Sparse BLAS Reduction:** Employs exact fractional polygon intersection matrices (`ZonalWeightMatrix`) with compressed `.npz` caching, reducing spatial averaging for thousands of catchments to millisecond matrix operations.
* **Seamless Model Ingestion:** Outputs Zarr v2 stores directly compatible with `googlehydrology.datasetzoo.multimet.Multimet` for training and inference with `MeanEmbeddingForecastLSTM` and `HandoffForecastLSTM`.

**Quickstart:**
```bash
extract-multimet \
  --basins_path test/test_data/shapefiles/us/us_basin_shapes.geojson \
  --output_dir /tmp/multimet_extracted \
  --products CPC,ERA5_LAND,IMERG,HRES,GRAPHCAST \
  --start_date 2020-01-01 \
  --end_date 2020-01-02
```
For in-depth documentation, see the [MultiMet Subdirectory README](googlehydrology/multimet/README.md) and [Sphinx Documentation](docs/source/usage/multimet_extractor.rst).

## **Usage**

The package installs the run command as the primary entry point.

### **Training a Model**
   
   ```
   run train --config-file /path/to/your/training_config_file.yml
   ```

### **Evaluation**

Calculate performance metrics (NSE, KGE) on the test set:
   
   ```
   run evaluate --run-dir /path/to/your/model_run/
   ```

### **Inference**

Generate predictions (without skipping NaN observations):
   
   ```
   run infer --run-dir /path/to/your/model_run/
   ```

## **Configuration**

Experiments are defined by YAML files. Update the following paths in your config (e.g., tutorial/training-config.yml):

* run\_dir: Where weights and logs are saved.  
* train\_basin\_file: Path to the list of basin IDs.  
* targets\_data\_dir / statics\_data\_dir: Path to your local Caravan NetCDF data.  
* dynamics\_data\_dir: Path to forcing data (e.g., gs://caravan-multimet/v1.1).

### **Example Configurations**

The `~/flood-forecasting/example-configs` directory contains reference YAML files that define the experimental setups for different model architectures and datasets.

* **`floodhub-settings-config.yml`**  
  * **Model Architecture:** `mean_embedding_forecast_lstm`  
  * **Dataset:** MultiMet (Global Caravan dataset)  
  * **Description:** This configuration is designed to replicate the training settings of the current (2025) operational FloodHub model as closely as possible within this open-source framework.  
* **`handoff-forecast-lstm-config.yml`**  
  * **Model Architecture:** `handoff_forecast_lstm`  
  * **Dataset:** MultiMet (Global Caravan dataset)  
  * **Description:** Provides the settings used for the former operational model. This configuration aligns with the methodology described in the *Nature* (2024) paper for global ungauged flood prediction.  
* **`camels-multimet-mean-embedding-forecast-lstm-config.yml`**  
  * **Model Architecture:** `mean_embedding_forecast_lstm`  
  * **Dataset:** CAMELS-US (531 basins)  
  * **Description:** A benchmarking configuration for the Mean-Embedding model tailored for the CAMELS-US dataset. It is optimized for evaluating model stability and performance on a standard hydrological benchmark. Our team uses this as a reference point during model development, and it is included in this repository because this is what we use to ensure that any changes to the repository work as expected.  
* **`camels-multimet-handoff-forecast-lstm-config.yml`**  
  * **Model Architecture:** `handoff_forecast_lstm`  
  * **Dataset:** CAMELS-US (531 basins)  
  * **Description:** A benchmarking configuration for the State Handoff model tailored for the CAMELS-US dataset, used to compare the handoff approach against other architectures on US-based basin data.

## **Issue Reporting**

If you encounter bugs, please use the [GitHub Issue Tracker](https://github.com/google-research/flood-forecasting/issues). Provide a clear description, steps to reproduce, and the expected behavior.
