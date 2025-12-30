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

This tutorial is structured to provide insights into five core areas of hydrologic machine learning:

1. **Train a Base Models**
   Train a hydrological model on a (small) set of basins. This serves as a foundation for further specialization.

2. **Fine-Tune for a Specific Basin**
   Understand the process of adapting a pre-trained model to improve its performance on a specific target or region that might be "out-of-distribution" compared to the base model's training data.

3. **Running ``googlehydrology`` Models**
   Gain practical experience generating and understanding the command-line arguments for training (``train``), fine-tuning (``finetune``), and performing inference (``infer``). You will learn how to prepare configuration files and execute these operations in a terminal environment.

4. **Model Performance Analysis**
   Quantitatively evaluate model performance using standard metrics and qualitatively assess predictions through hydrograph comparisons. Key metrics covered include:

   * **NSE** (Nash-Sutcliffe Efficiency)
   * **KGE** (Kling-Gupta Efficiency)

5. **Impact of Static Attributes**
   Explore the role of static basin attributes (like basin area) and learn how targeted fine-tuning of the ``static_attributes_fc`` embedding layer of the :ref:`Mean Embedding Forecast LSTM <mean-embedding-forecast-lstm>` can address performance discrepancies.

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
