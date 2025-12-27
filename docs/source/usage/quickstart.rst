Quick Start
===========

Prerequisites
-------------
As a first step you need a Python environment with all required dependencies.
The recommended way is to use Mini-/Anaconda and to create a new environment using the predefined environment file in environments/ <https://github.com/google-research/flood-forecasting/tree/master/environments>__.

.. code-block:: bash

    conda env create -f environments/conda.yml
    conda activate googlehydrology

If you prefer to not use Mini-/Anaconda, make sure you have a Python environment with Python >= 3.12 with all packages installed that are listed in `rtd_requirements.txt`. The next steps should be executed from within this Python environment.

Installation
------------
There are two ways to install GoogleHydrology: Editable or non-editable.

If you do not expect to edit the code or add your own models or datasets, you can use the non-editable installation.
To install the latest release from PyPI:

.. code-block:: bash

    pip install googlehydrology

To install the package directly from the current master branch of this repository, including any changes that are not yet part of a release, run:

.. code-block:: bash

    pip install git+https://github.com/google-research/flood-forecasting.git

If you want to try implementing your own models or datasets, you'll need an editable installation.
For this, start by downloading or cloning the repository to your local machine.
If you use git, you can run:

.. code-block:: bash

    git clone https://github.com/google-research/flood-forecasting.git

If you don't know git, you can also download the code from `here <https://github.com/google-research/flood-forecasting/zipball/master>` and extract the zip-file.
After you cloned or downloaded the zip-file, you'll end up with a directory called "flood-forecasting". The source code for the model and all training and inference pipelines is in the `~/flood-forecasting/googlehydrology` subdirectory.
Next, we'll go to that directory and install a local, editable copy of the package:

.. code-block:: bash

    cd googlehydrology
    pip install -e .

The installation procedure (both the editable and the non-editable version) adds the package to your Python environment and installs three bash scripts: `run`, `schedule-runs` and `results-ensemble`.

Data
----
The model is configured to run with the Caravan dataset and the MultiMet extension. The original Caravan dataset provides static attributes and streamflow data, and the MultiMet dataset provides meteorological forcing data with forecast lead times. 
The MultiMet dataset does NOT need to be downloaded -- it can be read directly
from a Google Cloud Storage bucket here: gs://caravan-multimet/v1.1. 
The Caravan dataset must be downloaded to your local machine. It can be downloaded from the Zenodo repository here: https://zenodo.org/records/15529786.

Training a model
----------------
To train a model, first prepare a configuration file. An example configuration file is provided in the tutorial directory.
Once you have a configuration file, run the training pipeline:

.. code-block:: bash

    run train --config-file /path/to/config.yml

You can optionally specify a GPU to use (overriding the config file):

.. code-block:: bash

    run train --config-file /path/to/config.yml --gpu 0

Continuing Training
-------------------
To resume a run that was interrupted or to continue training a model for more epochs:

.. code-block:: bash

    run continue_training --run-dir /path/to/run_dir/

Fine-tuning
-----------
To fine-tune a pre-trained model on a new dataset or with different settings:

.. code-block:: bash

    run finetune --config-file /path/to/finetune_config.yml

Note that the ``finetune_config.yml`` must contain the ``base_run_dir`` argument pointing to the pre-trained model and ``finetune_modules`` specifying which parts of the model to update. An example finetuning configuration file is provided in the tutorial directory.

Evaluating a model
------------------
To evaluate a trained model on the test set (calculating metrics and optionally saving results):

.. code-block:: bash

    run evaluate --run-dir /path/to/run_dir/

If the optional argument ``--epoch N`` (where N is the epoch to evaluate) is not specified, the weights of the last epoch are used.

Inference Mode
--------------
To run the model in inference mode (generating predictions without skipping NaN observations and saving all model outputs):

.. code-block:: bash

    run infer --run-dir /path/to/run_dir/

Scheduling Multiple Runs
------------------------
If you want to train multiple models, you can make use of the ``schedule-runs`` command.
Place all configs in a folder, then run:

.. code-block:: bash

    schedule-runs train --directory /path/to/config_dir/ --runs-per-gpu X --gpu-ids Y Z

* ``X``: How many models should be trained in parallel on a single GPU.
* ``Y Z``: Space-separated list of GPU IDs to use (e.g., ``0 1``).

To evaluate all runs in a specific directory:

.. code-block:: bash

    schedule-runs evaluate --directory /path/to/config_dir/ --runs-per-gpu X --gpu-ids Y

Ensemble Results
----------------
To merge the predictions of a number of runs (stored in ``$DIR1``, ...) into one averaged ensemble, use the ``results-ensemble`` script:

.. code-block:: bash

    results-ensemble --run-dirs $DIR1 $DIR2 ... --output-dir /path/to/output/directory --metrics NSE MSE ...

``--metrics`` specifies which metrics will be calculated for the averaged predictions. A full list of metrics is in `~/flood-forecasting/googlehydrology/evaluation/metrics.py`.