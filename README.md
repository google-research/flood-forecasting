# **Flood Forecasting**

This is not an officially supported Google product. This project is not eligible for the Google Open Source Software Vulnerability Rewards Program.

The repository provides open-source replication of Google’s state-of-the-art flood-forecasting models. By open-sourcing these models, we aim to foster transparency, enable in-house integration in production systems, and accelerate academic research.

## **Features**

* Implementation of hydrology forecast models for gauged and ungauged basins.  
* Data pipelines reading from public datasets: [Caravan](https://www.nature.com/articles/s41597-023-01975-w) and [Caravan MultiMet](https://arxiv.org/abs/2411.09459).  
* Pipelines for training, fine-tuning with historical data, and assimilating real-time data.  
* The coding style and workflows in this repository are based on the open source GoogleHydrology library. All models are implemented using PyTorch.

## **Installation**

The recommended way to install the package is via conda to ensure all dependencies (including PyTorch with CUDA support) are handled correctly.

1. **Create the environment:**  
   Use the provided environment files located in the environments/ folder:  
   ```
   # For CUDA 11.8 support  
   conda env create -f environments/conda.yml

   # Activate the environment  
   conda activate googlehydrology
   ```

2. **Install the package:**  
   Install the package in editable mode to develop and run scripts. Ensure you are in the root directory of the repository (where setup.py is located):  
   ```
   pip install -e
   ```

## **Usage**

This repository provides two main entry points installed by the package: run (mapped to googlehydrology/run.py) for single experiments and schedule-runs (mapped to googlehydrology/run\_scheduler.py) for multi-GPU parallel experiments.

For a step-by-step walkthrough, refer to the **tutorial/googlehydrology-tutorial.ipynb** notebook.

### **1\. Training and Evaluation (run)**

The run command handles training, evaluation, inference, and fine-tuning.

**Training a model:**
To train a model, you must provide a configuration file (.yml).  
```
run train --config-file tutorial/config.yml --gpu 0
```

**Continuing training:** 
To resume training from a specific run directory (e.g., after an interruption):  
```
run continue_training --run-dir runs/my_experiment_run1/ --gpu 0
```

**Fine-tuning:**
To fine-tune a pre-trained model on a new set of basins or parameters. The config file must point to the base\_run\_dir.  
```
run finetune --config-file tutorial/finetune_config.yml --gpu 0
```

**Evaluation & Inference**

* **Evaluate:** Calculates metrics (e.g., NSE, KGE, etc.) on the test set.  
  ```
  run evaluate --run-dir runs/my_experiment_run1/ --period test
  ```

* **Infer:** Runs the model in inference mode (saves outputs to disk, does not skip NaN observations).  
  ```
  run infer --run-dir runs/my_experiment_run1/ --period test
  ```

### **2\. Scheduling Multiple Runs (schedule-runs)**

The schedule-runs command allows you to queue multiple experiments across one or multiple GPUs. It balances the load by checking active processes.

**Training multiple configurations:**  
Point the scheduler to a directory containing multiple .yml config files.  
```
schedule-runs train --directory ./configs/experiment_set_1 --gpu-ids 0 1 2 --runs-per-gpu 2
```

**Evaluating multiple runs:**  
Point the scheduler to a directory containing run sub-folders.  
```
schedule-runs evaluate --directory ./runs/experiment_set_1 --gpu-ids 0 --runs_per_gpu 4
```

## **Configuration**

Experiments are defined by YAML configuration files. Example configuration files are available in the tutorial/ directory (e.g., `tutorial/config.yml`).

Key sections include:

* **General:** experiment\_name, run\_dir, device  
* **Data:** dataset, data\_dir, dynamic\_inputs, target\_variables  
* **Model:** model (e.g., handoff\_forecast\_lstm), head (e.g., regression, cmal), hidden\_size  
* **Training:** epochs, batch\_size, optimizer, loss

## **Issue reporting**

If you encounter any bugs or have feature requests, please use the [GitHub Issue Tracker](https://www.google.com/search?q=https://github.com/googlehydrology/googlehydrology/issues).

**Describe the Bug:** A clear and concise description of what the bug is.  
**To Reproduce:** Steps to reproduce the behavior. E.g. which data did you use, what were the commands that you executed, did you modify the code, etc.  
**Expected Behavior:**  A clear and concise description of what you expected to happen.