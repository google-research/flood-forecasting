# **Google Flood Hub: Pre-trained NeuralHydrology Weights**

This directory contains a pre-trained model run (`google-floodhub-settings-110-epochs`) based on the Google Flood Hub architecture (Mean Embedding Forecast LSTM). This release is intended to accelerate hydrological research, enable warm-started fine-tuning for local catchments, and support Prediction in Ungauged Basins (PUB) experiments.

🚨 **CRITICAL METHODOLOGICAL CAVEAT \- READ BEFORE USING** 🚨

**This model was trained on the FULL historical data period (1982-2023). There is NO temporal holdout/test split.**

Because the model has seen the entire historical timeline during training, **you cannot use these weights to evaluate temporal forecasting performance on historical datasets.** Any standard evaluation of this model on the 1982-2023 period will result in fundamentally invalid, artificially inflated performance metrics due to in-sample evaluation (data leakage).

Please see the **"Appropriate Use Cases"** section below for instructions on how to properly utilize these weights for scientifically rigorous research.

## **0\. Compatibility Note (version 1.13.0)**

Version 1.13.0 of `googlehydrology` corrected an off-by-one error in how forecast lead times were aligned with hindcast inputs and targets (GitHub issue #332). The weights in this directory were **retrained with the corrected code**. Weights produced by earlier versions of the code base (including the previous release of this model) are not compatible with version 1.13.0 and later: they expect inputs shifted by one day and will produce degraded forecasts. Likewise, do not load these weights with code older than 1.13.0.

## **1\. Model Overview**

We are releasing a pre-trained global baseline model trained on the MultiMet Caravan dataset configuration (excluding the CHIRPS precipitation product):

### **Full Basin Baseline (`google-floodhub-settings-110-epochs`)**

* **Purpose:** A generalized global baseline that captures the widest possible variety of hydrological behaviors, topologies, and climates available in the dataset.  
* **Configuration:** [`./example-configs/floodhub-settings-config.yml`](../example-configs/floodhub-settings-config.yml)  
* **Training Data:** The complete standard basin list (`15,955` listed basins; `10,137` evaluable basins with observations).  
  * [`./example-configs/multimet-basins-list-without-chirps.txt`](../example-configs/multimet-basins-list-without-chirps.txt)

## **2\. Contents of the Release**

To ensure seamless integration with the OpenHydroNet framework, we are releasing the complete runtime directory (`google-floodhub-settings-110-epochs/`) rather than an isolated weight file. The folder contains:

* **Model Weights (`model_epoch110.pt`):** The trained neural network parameters at epoch 110.  
* **Pre-Computed Scalers (`scaler.zarr/`):** The exact feature and target scalers (mean/std) computed across the global training dataset. *This is critical:* when you fine-tune this model on local data, OpenHydroNet will load this scaler to ensure your local inputs are normalized consistently with the pre-trained features.  
* **Original Configuration (`config.yml`):** The exact hyperparameters, input variable lists, and static attributes used to generate the run, ensuring full reproducibility.  
* **Evaluation Metrics (`test/model_epoch110/test_metrics.csv`):** Full-dataset (`10,137`-basin) in-sample evaluation metrics.

## **3\. Appropriate & Inappropriate Use Cases**

To ensure the integrity of your research, please adhere to the following usage guidelines.

### **❌ Inappropriate Uses (Do Not Do This)**

* **Historical Benchmarking:** Running standard inference on the training period (1982-2023) and reporting the NSE/KGE or other skill scores.  
* **Direct Operational Deployment:** Using these exact weights for live forecasting without rigorous local validation and fine-tuning.

### **✅ Appropriate Uses (Recommended)**

* **Fine-Tuning (Transfer Learning):** Using this model to initialize a network, followed by training on a localized, heavily instrumented dataset (e.g., with local weather radar or higher resolution DEMs).  
* **Spatial Generalization (PUB):** Evaluating the model on *spatially held-out* basins. If you have basins that were completely excluded from the training list, you can evaluate the model's ability to generalize to those ungauged locations during the 1982-2023 period.  
* **Future Inference:** Running forward-looking inference on data generated strictly after the training period cutoff (post-2023).

## **4\. How to Use for Fine-Tuning**

The primary intended use case for this model is transfer learning via fine-tuning. The OpenHydroNet codebase supports this, with an example given in the tutorial directory **(`~/tutorial/OpenHydroNet_Tutorial.ipynb`)**.

Instead of initializing random weights, you can have your new model load our pre-trained weights and pre-computed scalers by setting the `base_run_dir` parameter in a new fine-tuning config file. Please follow the procedure outlined in the tutorial.

### **Example Fine-Tuning Configuration**

Create a new configuration file for your local basins (e.g., `finetune_config.yml`). Add the `base_run_dir` argument pointing to the extracted run directory you downloaded from this repository:

\# Please note that this is just an example of a fine tuning config file.  
\# You will need to modify this for your own data.

\# \--- Fine-Tuning specific arguments \---  
\# Point this to the directory containing the pre-trained model  
base\_run\_dir: /path/to/downloaded/google-floodhub-settings-110-epochs

\# Fine-tuning parameters  
epochs: 30                    \# Require fewer epochs since we are warm-starting  
initial\_learning\_rate: 0.0001 \# Use a lower learning rate to avoid destroying pre-trained features  
learning\_rate\_strategy: ReduceLROnPlateau

\# \--- Standard configurations \---  
train\_basin\_file: /path/to/your/local\_finetune\_basins.txt  
train\_start\_date: 01/01/1990  
train\_end\_date: 31/12/2015

\# FOR FINE-TUNING, YOU MUST HAVE A VALID TEST SPLIT  
test\_basin\_file: /path/to/your/local\_finetune\_basins.txt  
test\_start\_date: 01/01/2016  
test\_end\_date: 31/12/2023

**To run the fine-tuning process:**

python googlehydrology/run.py train \--config-file finetune\_config.yml

### **Note on Data Scaling**

When fine-tuning using `base_run_dir`, OpenHydroNet will automatically load the dataset Scaler from our pre-trained directory. It is strictly required that the new fine-tuning dataset uses the exact same input variables as the pre-trained model.