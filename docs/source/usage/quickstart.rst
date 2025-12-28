===========
Quick Start
===========

This guide will help you get up and running with the **GoogleHydrology Flood Forecasting** repository.

---------------
Obtain the Code
---------------

To get started, you need to download the source code to your local machine. This ensures you have access to the environment definitions, example configurations, and the tutorial notebook.

Option A: Via GitHub Cloning (Recommended)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you use git, clone the repository to access the full source and development history:

.. code-block:: bash

   git clone https://github.com/google-research/flood-forecasting.git
   cd flood-forecasting

Option B: Via Zipball
^^^^^^^^^^^^^^^^^^^^^

If you do not use git, you can download the source code as a zip file:

.. code-block:: bash

   # Download the source code zip file
   curl -L https://github.com/google-research/flood-forecasting/zipball/master -o flood-forecasting.zip

   # Extract the archive
   unzip flood-forecasting.zip

   # Enter the resulting directory (folder name may vary based on the specific commit)
   cd google-research-flood-forecasting-*

----------------------------------
Prerequisites & Environment Setup
----------------------------------

A Python environment with specific dependencies (like PyTorch and CUDA) is required. We recommend using **Conda** to manage these dependencies automatically.

Using Conda (Recommended)
^^^^^^^^^^^^^^^^^^^^^^^^^

The environment file is located in the ``environments/`` directory of the code you just obtained.

.. code-block:: bash

   # Create the environment from the file in the repo
   conda env create -f environments/conda.yml

   # Activate the environment (MANDATORY)
   conda activate googlehydrology

Manual Setup
^^^^^^^^^^^^

If you prefer not to use Conda, ensure you have **Python >= 3.12** and install the dependencies listed in ``environments/rtd_requirements.txt`` using your preferred package manager.

------------
Installation
------------

Once your environment is active, install the package in **editable mode**. This allows you to run the model scripts and have any changes you make to the code reflected immediately.

.. code-block:: bash

   # Run this from the root of the flood-forecasting directory
   pip install -e .

----------
Data Setup
----------

GoogleHydrology uses the `Caravan <https://www.nature.com/articles/s41597-023-01975-w>`_ dataset for streamflow observations and static catchment attributes.

Download Caravan (NetCDF Version)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A small amount of data is provided in the ``~/tutorial/data/Caravan-nc`` folder. This data is sufficient for running the tutorial example. For more comprehensive model runs, it is necessary to download the Caravan dataset locally.

1. Navigate to the `Zenodo repository <https://doi.org/10.5281/zenodo.6522634>`_.
2. Download the **NetCDF version** of the dataset (e.g., ``Caravan-nc.tar.gz``). 
   
   .. note:: 
      Do not use the CSV version for standard training as it is significantly slower.

3. Unpack the tarball file into a local directory (e.g., ``~/data/caravan/``).

.. code-block:: bash

   # Create the directory and unpack the data
   mkdir -p ~/data/
   tar -xvzf Caravan-nc.tar.gz -C ~/data/

MultiMet Data
^^^^^^^^^^^^^

The MultiMet forcing data extension is accessed directly from **Google Cloud Storage** during runtime. You do not need to download it; ensure your configuration file's ``dynamics_data_dir`` argument points to: ``gs://caravan-multimet/v1.1``

----------------------
Training Configuration
----------------------

To train a model, you must create or modify a YAML configuration file. An example is provided in the ``tutorial/`` directory (``training-config.yml``).

Understanding the Dataset Splits
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^



* **Training Set (5-basin set):** Core portion used by the algorithm to identify patterns and learn relationships between inputs and streamflow.
* **Test Set (8-basin set):** A final, independent portion never "seen" during training. In this example, it includes the 5 training basins plus 3 additional "ungauged" basins to test generalization.

Understanding Time Periods
^^^^^^^^^^^^^^^^^^^^^^^^^^

* **Training Period:** 01/01/2000 to 31/12/2020 (Historical learning).
* **Validation/Test Period:** 01/01/2022 to 31/12/2024 (Objective performance estimation).

It is normal practice to keep the Validation and Test periods distinct to avoid information leakage, ensuring the model reflects performance on truly novel data. Please notice that this was **not** done for the toy example in the tutorial.

Local Path Requirements
^^^^^^^^^^^^^^^^^^^^^^^

Update these arguments in your configuration file (``~/tutorial/configs/training-config.yml``) to match your data source:

=====================  ============================================================================================================
Argument               Description
=====================  ============================================================================================================
**run_dir**            Directory where weights, logs, and config copies are saved (e.g., ``~/tutorial/run/``).

**train_basin_file**   Path to plain text files containing lists of basin IDs.

**targets_data_dir**   Use the tutorial sample (``~/tutorial/data/Caravan-nc/``) OR the unpacked full dataset, wherever you put it.

**statics_data_dir**   Use the tutorial sample (``~/tutorial/data/Caravan-nc/``) OR the unpacked full dataset, wherever you put it.

**dynamics_data_dir**   Path to the forcing data. For MultiMet, use the cloud bucket: ``gs://caravan-multimet/v1.1``.
=====================  ============================================================================================================

-----
Usage
-----

Training a model
^^^^^^^^^^^^^^^^

.. code-block:: bash

   run train --config-file ~/tutorial/training-config.yml --gpu 0

Evaluation
^^^^^^^^^^

To calculate performance metrics on the test set:

.. code-block:: bash

   run evaluate --run-dir /path/to/your/model_run/

Inference
^^^^^^^^^

To generate predictions without skipping NaN observations:

.. code-block:: bash

   run infer --run-dir /path/to/your/model_run/