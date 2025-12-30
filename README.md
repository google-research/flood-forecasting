# **Flood Forecasting**

### **🌊 This repository implements the state-of-the-art models that power [Google FloodHub](https://sites.research.google/floods/).**

This is not an officially supported Google product. This project is not eligible for the Google Open Source Software Vulnerability Rewards Program.

The repository provides open-source replication of Google’s global flood-forecasting models. By open-sourcing these models, we aim to foster transparency, enable in-house integration in production systems, and accelerate academic research.

This repository is a fork of [NeuralHydrology](https://github.com/neuralhydrology/neuralhydrology), which has been heavily modified and extended to support forecast sequences using the specific model architectures that are used operationally in the Google FloodHub.

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

The most direct way to explore this repository is through our interactive tutorial: [**GoogleHydrology Evaluation & Fine-Tuning Notebook**](https://www.google.com/search?q=./GoogleHydrology_Evaluation_Notebook_with_Fine_Tuning.ipynb).

**What you will learn:**

* **Model Evaluation:** Load pre-trained Google Hydrology models and calculate performance metrics (NSE, KGE) on real-world basin data.  
* **Fine-Tuning for Performance:** Learn how to fine-tune the `static_attributes_fc` layer. This is a powerful technique for improving predictions on "outlier" basins (e.g., basins with unusual sizes or geology) without retraining the entire model.  
* **Visualizing Results:** Compare model hydrographs against observed discharge data.

**Run it now:** 
<!-- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-research/flood-forecasting/blob/main/GoogleHydrology_Evaluation_Notebook_with_Fine_Tuning.ipynb) -->

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-research/flood-forecasting/blob/gsnearing-tutorial/GoogleHydrology_Evaluation_Notebook_with_Fine_Tuning.ipynb)

## **Data Setup**

GoogleHydrology uses the [Caravan](https://www.nature.com/articles/s41597-023-01975-w) dataset for streamflow observations and static catchment attributes.

### **1\. Download Caravan (NetCDF Version)**

A small sample is provided in tutorial/data/Caravan-nc. For full runs:

1. Visit the [Zenodo repository](https://doi.org/10.5281/zenodo.6522634).  
2. Download the **NetCDF version** (Caravan-nc.tar.gz).  
3. Unpack it locally:  

   ```
   mkdir -p ~/data/  
   tar -xvzf Caravan-nc.tar.gz -C ~/data/
   ```

### **2\. MultiMet Data**

The MultiMet forcing data extension is accessed directly from **Google Cloud Storage**. Ensure your configuration points to: gs://caravan-multimet/v1.1

## **Usage**

The package installs the run command as the primary entry point.

### **Training a Model**
   
   ```
   run train --config-file /path/to/your/training_config_file.yml --gpu 0
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

## **Issue Reporting**

If you encounter bugs, please use the [GitHub Issue Tracker](https://www.google.com/search?q=https://github.com/google-research/flood-forecasting/issues). Provide a clear description, steps to reproduce, and the expected behavior.