# **Google Flood Hub: Pre-trained NeuralHydrology Weights**

This repository contains two sets of pre-trained model runs based on the Google Flood Hub architecture (Mean Embedding Forecast LSTM). These releases are intended to accelerate hydrological research, enable warm-started fine-tuning for local catchments, and support Prediction in Ungauged Basins (PUB) experiments.

🚨 **CRITICAL METHODOLOGICAL CAVEAT \- READ BEFORE USING** 🚨

**These models were trained on the FULL historical data period (1982-2023). There is NO temporal holdout/test split.**

Because the models have seen the entire historical timeline during training, **you cannot use these weights to evaluate temporal forecasting performance on historical datasets.** Any standard evaluation of these models on the 1982-2023 period will result in fundamentally invalid, artificially inflated performance metrics due to in-sample evaluation (data leakage).

Please see the **"Appropriate Use Cases"** section below for instructions on how to properly utilize these weights for scientifically rigorous research.

## **1\. Model Variants**

We are releasing two distinct sets of pre-trained models, both trained using the multimet dataset configuration (excluding the CHIRPS precipitation product).

### **Model A: Full Basin Baseline (google-floodhub-base)**

* **Purpose:** A generalized global baseline that captures the widest possible variety of hydrological behaviors, topologies, and climates available in the dataset.  
* **Configuration:** [Config File (./example-configs/config\_full\_baseline.yml)](http://docs.google.com/example-configs/config_full_baseline.yml)  
* **Training Data:** The complete standard basin list.  
  * [Basin List (./example-configs/multimet-basins-list-without-chirps.txt)](http://docs.google.com/example-configs/multimet-basins-list-without-chirps.txt)

### **Model B: High-Skill Filtered (google-floodhub-nse-filtered)**

* **Purpose:** A high-signal model that uses a pre-training step (google-floodhub-base) as an approximate filter for data quality. By removing fundamentally unpredictable, heavily regulated, or severely noisy catchments from the training objective, this model focuses purely on learning physically consistent rainfall-runoff relationships. It is recommended as a starting point for fine-tuning in well-behaved basins.  
* **Configuration:** [Config File (./example-configs/config\_filtered\_nse\_gt\_0.5.yml)](http://docs.google.com/example-configs/config_filtered_nse_gt_0.5.yml)  
* **Training Data:** A strict subset of basins that demonstrated a Nash-Sutcliffe Efficiency (NSE) \> 0.5 during evaluation from the baseline run.  
  * [Basin List (./example-configs/filtered\_basins\_nse\_gt\_0.5.txt)](http://docs.google.com/example-configs/filtered_basins_nse_gt_0.5.txt)

## **2\. Contents of the Release**

To ensure seamless integration with the OpenHydroNet framework, we are releasing the complete runtime directories for both models, rather than isolated weight files. Each folder contains:

* **Model Weights (model\_eopchXXX.pt):** The fully trained neural network parameters.  
* **Pre-Computed Scalers (train\_data/):** The exact feature and target scalers (mean/std) computed across the global training dataset. *This is critical:* when you fine-tune these models on local data, NeuralHydrology will load this scaler to ensure your local inputs are normalized perfectly consistently with the pre-trained features.  
* **Optimizer States (optimizer\_state\_eopchXXX.pt)::** The saved states of the Adam optimizer. This allows you to resume training without inducing momentum shocks to the weights during the first few epochs.  
* **Original Configuration (config.yml):** The exact hyperparameters, input variable lists, and static attributes used to generate the run, ensuring full reproducibility.

## **3\. Appropriate & Inappropriate Use Cases**

To ensure the integrity of your research, please adhere to the following usage guidelines.

### **❌ Inappropriate Uses (Do Not Do This)**

* **Historical Benchmarking:** Running standard inference on the training period (1982-2023) and reporting the NSE/KGE or other skill scores.  
* **Direct Operational Deployment:** Using these exact weights for live forecasting without rigorous local validation and fine-tuning.

### **✅ Appropriate Uses (Recommended)**

* **Fine-Tuning (Transfer Learning):** Using these models to initialize a network, followed by training on a localized, heavily instrumented dataset (e.g., with local weather radar or higher resolution DEMs).  
* **Spatial Generalization (PUB):** Evaluating the model on *spatially held-out* basins. If you have basins that were completely excluded from the training list, you can evaluate the model's ability to generalize to those ungauged locations during the 1982-2023 period.  
* **Future Inference:** Running forward-looking inference on data generated strictly after the training period cutoff (post-2023).

## **4\. How to Use for Fine-Tuning**

The primary intended use case for these models is transfer learning via fine-tuning. The OpenHydroNet codebase supports this, with an example given in the tutorial directory **(\~/tutorial/OpenHydroNet\_Tutorial.ipynb)**.

Instead of initializing random weights, you can have your new model to load our pre-trained weights and pre-computed scalers by setting the base\_run\_dir parameter in a new fine tuning config file. Please follow the procedure outlined in the tutorial.

### **Example Fine-Tuning Configuration**

Create a new configuration file for your local basins (e.g., finetune\_config.yml). Add the base\_run\_dir argument pointing to the extracted run directory you downloaded from this repository:

\# Please note that this is just an example of a fine tuning config file.  
\# You will need to modify this for your own data.

\# \--- Fine-Tuning specific arguments \---  
\# Point this to the directory containing Model A or Model B  
base\_run\_dir: /path/to/downloaded/google-floodhub-nse-filtered

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

When fine-tuning using base\_run\_dir, NeuralHydrology will automatically load the dataset Scaler from our pre-trained directory. It is strictly required that the new fine-tuning dataset uses the exact same input variables as the pre-trained model.