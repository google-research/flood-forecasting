Tutorial: Evaluation and Fine-Tuning
====================================

This tutorial provides a guide to training and evaluating Google's global flood forecasting models and adapting them to local basin conditions.

The primary goal of this example workflow is to bridge the gap between a large-scale pre-trained model and specific local performance by fine-tuning the layers responsible for interpreting geographic features.

Interactive Version
-------------------

You can run this tutorial interactively in your browser using Google Colab. This is the recommended way to get started as it provides a pre-configured environment with all necessary dependencies.

.. image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/google-research/flood-forecasting/blob/main/GoogleHydrology_Evaluation_Notebook_with_Fine_Tuning.ipynb
   :alt: Open In Colab

Core Concepts
-------------

The tutorial covers three main technical workflows:

1. **Model Evaluation**
   Learn how to load pre-trained weights from the Google Hydrology model zoo and calculate standard hydrological performance metrics, including:

   * **NSE** (Nash-Sutcliffe Efficiency)
   * **KGE** (Kling-Gupta Efficiency)

2. **Fine-Tuning the Static Embedding**
   A key highlight of this tutorial is the fine-tuning of the ``static_attributes_fc`` (fully connected) layer.

   Many global models struggle with "outlier" basins—those with unique geological or geographic features not well-represented in the broad training set. Instead of retraining the entire LSTM, we optimize only the static attribute encoder. This allows the model to learn a better "representation" of the specific basin's characteristics (like area or slope) while keeping the temporal forecasting logic intact.

3. **Hydrograph Visualization**
   The tutorial includes tools to plot model predictions against observed discharge data to visually assess timing, peak magnitude, and baseflow accuracy.

This notebook is designed as an educational exercise rather than a performance benchmark. To ensure the code runs quickly in a standard environment (like Google Colab), the experiment is restricted to a "toy" dataset of only 5 training basins. Because State-of-the-Art (SOTA) global models typically require data from hundreds or thousands of basins to learn universal hydrologic behaviors and relationships, this 5-basin model will **not** yield state-of-the-art results. Specifically, a model trained on such a small sample size lacks the "experience" to understand basins in different climates or terrains. You will observe that performance metrics (NSE/KGE) on the 3 "ungauged" basins (basins not seen during training) are significantly lower than the training set. This is expected behavior from a model trained on a small (5-basin) dataset.

Prerequisites
-------------

To run this tutorial locally, ensure you have:

* **Python 3.12**
* The ``googlehydrology`` package installed (follow the :doc:`install instructions </usage/quickstart>`).

Data Requirements
-----------------

All data required for the tutorial is included in this repository.

Notebook File
-------------

The source notebook for this tutorial can be found in the root of the repository:
``flood-forecasting/tutorial/GoogleHydrology_Evaluation_Notebook_with_Fine_Tuning.ipynb``
