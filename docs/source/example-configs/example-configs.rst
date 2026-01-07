Example Configuration Files
===========================

The ``~/flood-forecasting/example-configs`` directory contains reference configuration files in YAML format. These files define the experimental setups for different model architectures and datasets used in Google's production forecasting systems.

Overview
~~~~~~~~

The configurations provided here serve two primary purposes:

* **Operational Replication:** Replicating the exact or near-exact settings used by the Google FloodHub production models.
* **Benchmarking:** Providing stable baselines for model development and evaluation on standard datasets like CAMELS-US.

Configuration Files
~~~~~~~~~~~~~~~~~~~
.. list-table::
   :widths: 30 20 50
   :header-rows: 1

   * - File Name
     - Model Architecture
     - Description

   * - ``floodhub-settings-config.yml``
     - :mod:`~googlehydrology.modelzoo.mean_embedding_forecast_lstm`
     - Near-identical replication of the current (2025) production FloodHub model.

   * - ``handoff-forecast-lstm-config.yml``
     - :mod:`~googlehydrology.modelzoo.handoff_forecast_lstm`
     - Configuration for the former production model and methodology described in the Nature (2024) paper.

   * - ``camels-multimet-mean-embedding-forecast-lstm-config.yml``
     - :mod:`~googlehydrology.modelzoo.mean_embedding_forecast_lstm`
     - Optimized for benchmarking on the CAMELS-US (671 basins) dataset. Used as a core development reference.

   * - ``camels-multimet-handoff-forecast-lstm-config.yml``
     - :mod:`~googlehydrology.modelzoo.handoff_forecast_lstm`
     - Benchmarking configuration for the State Handoff model tailored for the CAMELS-US dataset.


Operational Models
~~~~~~~~~~~~~~~~~~

FloodHub 2025 (Mean Embeddings)
-------------------------------

The ``floodhub-settings-config.yml`` file defines the current operational standard. It utilizes the :mod:`~googlehydrology.modelzoo.mean_embedding_forecast_lstm` model, which handles multiple meteorological products by averaging their latent representations.

FloodHub 2024 (State Handoff)
-----------------------------

The ``handoff-forecast-lstm-config.yml`` file provides the settings for the previous operational generation. It focuses on transferring internal LSTM states (hidden and cell states) from a hindcast period to a forecast period via the :mod:`~googlehydrology.modelzoo.handoff_forecast_lstm` architecture.

Benchmarking & Development
~~~~~~~~~~~~~~~~~~~~~~~~~~

The configurations prefixed with ``camels-multimet-`` are specifically designed for testing on the CAMELS-US dataset. These CAMELS experiments are used internally by the Google team to verify that system updates do not degrade model performance.
