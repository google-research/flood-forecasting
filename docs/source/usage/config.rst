Configuration Arguments
=======================

This page provides a list of possible configuration arguments. 
Check out the file `tutorial/config.yml` for an example of how a config file could look like.

General experiment configurations
---------------------------------

-  ``experiment_name``: Defines the name of your experiment that will be used as a folder name (+ date-time string), as well as the name in TensorBoard. Curly brackets insert other configuration arguments into the experiment name. For example, ``experiment_name: batch-size-is-{batch_size}`` will yield ``batch-size-is-42`` if the ``batch_size`` argument is set to *42*. Furthermore, ``{random_name}`` provides a randomly named string (e.g., ``yellow-frog``).
-  ``run_dir``: Full or relative path to where the run directory is stored. If empty, runs are stored in ``${current_working_dir}/runs/``.
-  ``train_basin_file``: Full or relative path to a text file containing the training basins (use dataset basin id, one id per line).
-  ``validation_basin_file``: Full or relative path to a text file containing the validation basins.
-  ``test_basin_file``: Full or relative path to a text file containing the test basins.
-  ``train_start_date``: Start date of the training period (``DD/MM/YYYY``). Can be a list of dates to specify multiple periods.
-  ``train_end_date``: End date of the training period (``DD/MM/YYYY``).
-  ``validation_start_date``: Start date of the validation period (``DD/MM/YYYY``).
-  ``validation_end_date``: End date of the validation period (``DD/MM/YYYY``).
-  ``test_start_date``: Start date of the test period (``DD/MM/YYYY``).
-  ``test_end_date``: End date of the test period (``DD/MM/YYYY``).
-  ``seed``: Fixed random seed. If empty, a random seed is generated.
-  ``device``: Device to use, e.g., ``cuda:0``, ``cpu``, or ``mps``.
-  ``cache``: A dictionary with keys ``enabled`` (bool) and ``byte_limit`` (int) to control opportunistic caching of data.
-  ``use_swap_memory``: True/False. Whether to enable ``distributed.p2p.storage.disk`` explicitly for dask.
-  ``load_target_features_parallel_processes``: Integer. Number of processes to use to load target features' netCDF files in parallel.

Validation settings
-------------------

-  ``validate_every``: Interval (in epochs) at which validation is performed.
-  ``validate_n_random_basins``: Integer specifying how many random basins to use per validation.
-  ``metrics``: List of metrics to calculate. See :py:mod:`googlehydrology.evaluation.metrics`. Can also be a dictionary mapping target variables to lists of metrics.
-  ``save_validation_results``: True/False. If True, stores validation results to disk as a pickle file.

Evaluation settings
-------------------

-  ``inference_mode``: True/False. If True, saves observed data and model output to disk and does not skip dates with missing observations.
-  ``tester_sample_reduction``: ``mean`` or ``median``. How to reduce multiple samples (e.g., from MC-Dropout or CMAL) during evaluation.

General model configuration
---------------------------

-  ``model``: Defines the model class (e.g., ``cudalstm``, ``handoff_forecast_lstm``, ``mean_embedding_forecast_lstm``).
-  ``head``: The prediction head (``regression``, ``cmal``, ``cmal_deterministic``).
-  ``hidden_size``: Hidden size of the model (number of LSTM states).
-  ``initial_forget_bias``: Initial value of the forget gate bias.
-  ``output_dropout``: Dropout applied to the output of the LSTM.
-  ``weight_init_opts``: List of weight initialization options (e.g., ``lstm-ih-xavier``, ``fc-xavier``).
-  ``compile``: True/False. Whether to compile the model using ``torch.compile``.

Regression head
~~~~~~~~~~~~~~~
-  ``output_activation``: Activation on the output neuron (``linear``, ``relu``, ``softplus``).
-  ``mc_dropout``: True/False. Whether Monte-Carlo dropout is used during inference.

CMAL head
~~~~~~~~~
-  ``n_distributions``: Number of distributions for the CMAL head.
-  ``n_samples``: Number of samples generated per time-step.
-  ``cmal_deterministic``: True/False. Use deterministic 10-point sampling (mean + 9 quantiles) for CMAL.
-  ``negative_sample_handling``: ``clip``, ``truncate``, or ``none``.
-  ``negative_sample_max_retries``: Max retries for ``truncate`` sampling.

Forecast Model settings
~~~~~~~~~~~~~~~~~~~~~~~
-  ``forecast_hidden_size``: Hidden size for the forecast LSTM (defaults to ``hidden_size``).
-  ``hindcast_hidden_size``: Hidden size for the hindcast LSTM (defaults to ``hidden_size``).
-  ``state_handoff_network``: Configuration for the handoff network (see Embedding network settings).
-  ``forecast_overlap``: Integer number of timesteps where forecast overlaps with hindcast.

Embedding network settings
--------------------------
Used for static/dynamic inputs or specific model components like ``state_handoff_network``. Defined as a dictionary:

-  ``type``: ``fc`` (fully connected).
-  ``hiddens``: List of integers defining layer sizes.
-  ``activation``: Activation function (``tanh``, ``sigmoid``, ``linear``, ``relu``).
-  ``dropout``: Dropout rate.

Available keys for embeddings: ``statics_embedding``, ``dynamics_embedding``, ``hindcast_embedding``, ``forecast_embedding``.

Training settings
-----------------

-  ``optimizer``: Optimizer to use (``Adam``, ``AdamW``, ``SGD``, etc.).
-  ``loss``: Loss function (``MSE``, ``NSE``, ``RMSE``, ``CMALLoss``).
-  ``target_loss_weights``: List of weights for multi-target training.
-  ``regularization``: List of regularization terms (e.g., ``tie_frequencies``, ``forecast_overlap``).
-  ``learning_rate_strategy``: ``ConstantLR``, ``StepLR``, or ``ReduceLROnPlateau``.
-  ``initial_learning_rate``: Float. Starting learning rate.
-  ``learning_rate_drop_factor``: Factor by which to reduce the learning rate.
-  ``learning_rate_epochs_drop``: Epochs to wait before dropping LR.
-  ``batch_size``: Mini-batch size.
-  ``epochs``: Number of training epochs.
-  ``max_updates_per_epoch``: Optional limit on weight updates per epoch.
-  ``clip_gradient_norm``: Max norm for gradient clipping.
-  ``target_noise_std``: Standard deviation of Gaussian noise added to labels during training.
-  ``allow_subsequent_nan_losses``: Number of allowed consecutive NaN losses before stopping.
-  ``save_weights_every``: Interval (in epochs) to save model weights.

Data settings
-------------

-  ``dataset``: Dataset class to use (currently only ``multimet`` is supported, but users can add their own).
-  ``data_dir``: Root directory of the dataset.
-  ``statics_data_dir``, ``dynamics_data_dir``, ``targets_data_dir``: Optional overrides for specific data types.
-  ``load_as_csv``: True/False. Whether to force loading data from CSVs instead of binary formats (e.g., NetCDF/Zarr).
-  ``dynamic_inputs``: List of dynamic input variables.
-  ``forecast_inputs``: Dynamic features for the forecast period (used in forecast models).
-  ``hindcast_inputs``: Dynamic features for the hindcast period (used in forecast models).
-  ``target_variables``: List of target variables to predict.
-  ``static_attributes``: List of static attributes to use.
-  ``seq_length``: Input sequence length (or hindcast length for forecast models).
-  ``lead_time``: Forecast lead time (integer).
-  ``predict_last_n``: Number of time steps (counted backwards) used for loss calculation.
-  ``timestep_counter``: True/False. Adds a counting integer sequence as input for forecasts.
-  ``nan_handling_method``: ``masked_mean``, ``input_replacing``, or ``attention``. Strategy for handling missing input data.
-  ``nan_handling_pos_encoding_size``: Size of positional encoding for NaN handling methods.