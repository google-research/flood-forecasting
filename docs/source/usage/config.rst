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

-  ``validate_every``: Integer that specifies in which interval a validation is performed. If empty, no validation is done during training.
-  ``validate_n_random_basins``: Integer specifying how many random basins to use per validation.
-  ``metrics``: List of metrics to calculate. See :py:mod:`googlehydrology.evaluation.metrics`. Can also be a dictionary mapping target variables to lists of metrics.
-  ``save_validation_results``: True/False. If True, stores validation results to disk in a Zarr store.

Evaluation settings
-------------------

-  ``inference_mode``: True/False. If True, saves observed data and model output to disk and does not skip dates with missing observations.
-  ``tester_sample_reduction``: ``mean`` or ``median``. How to reduce multiple samples (e.g., from MC-Dropout or CMAL) during evaluation.

General model configuration
---------------------------

-  ``model``: Defines the model class (``handoff_forecast_lstm``, ``mean_embedding_forecast_lstm``).
-  ``head``: The prediction head (``regression``, ``cmal``, ``cmal_deterministic``).
-  ``hidden_size``: Hidden size of the model (number of LSTM states).
-  ``initial_forget_bias``: Initial value of the forget gate bias. A larger value (like 3) helps the model learn long-timescale dependencies.
-  ``output_dropout``: Dropout applied to the output of the LSTM.
-  ``weight_init_opts``: List of weight initialization options (e.g., ``lstm-ih-xavier``, ``fc-xavier``).
-  ``compile``: True/False. Whether to compile the model using ``torch.compile``. This 

Regression head
~~~~~~~~~~~~~~~
-  ``output_activation``: Activation on the output neuron (``linear``, ``relu``, ``softplus``).
-  ``mc_dropout``: True/False. Whether Monte-Carlo dropout is used during inference.

CMAL head
~~~~~~~~~
-  ``n_distributions``: Number of distributions for the CMAL head.
-  ``n_samples``: Number of samples generated per time-step.
-  ``cmal_deterministic``: True/False. Use deterministic 10-point sampling (mean + 9 quantiles) for CMAL.
-  ``negative_sample_handling``: Approach for handling negative sampling. Possible values are `none` for doing nothing, `clip` for clipping the values at zero, and `truncate` for resampling values that were drawn below zero. If the last option is chosen, the additional argument `negative_sample_max_retries` controls how often the values are resampled.
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

-  ``type``: (default ``fc``): Type of the embedding net. Currently, only ``fc`` for fully-connected net is supported.
-  ``hiddens``: List of integers that define the number of neurons per layer in the fully connected network. The last number is the number of output neurons. Must have at least length one.
-  ``activation``: activation function of the network. Supported values are: ``tanh``, ``sigmoid``, ``linear``, and ``relu``.
-  ``dropout``: Dropout rate.

Available keys for embeddings: ``statics_embedding``, ``dynamics_embedding``, ``hindcast_embedding``, ``forecast_embedding``.

Training settings
-----------------

-  ``optimizer``: Optimizer to use (``Adam``, ``AdamW``, ``SGD``, etc.).
-  ``loss``: Loss function (``MSE``, ``NSE``, ``RMSE``, ``CMALLoss``).
-  ``target_loss_weights``: A list of float values specifying the per-target loss weight, when training on multiple targets at once. Can be combined with any loss. By default, the weight of each target is ``1/n`` with ``n`` being the number of target variables. The order of the weights corresponds to the order of the ``target_variables``.
-  ``regularization``: List of regularization terms (currently, only ``forecast_overlap`` is supported for the ``handoff_forecast_lstm``).
-  ``learning_rate_strategy``: ``ConstantLR``, ``StepLR``, or ``ReduceLROnPlateau``.
-  ``initial_learning_rate``: Float. Starting learning rate.
-  ``learning_rate_drop_factor``: Factor by which to reduce the learning rate.
-  ``learning_rate_epochs_drop``: Epochs to wait before dropping LR.
-  ``batch_size``: Mini-batch size.
-  ``epochs``: Number of training epochs.
-  ``num_workers``: Number of (parallel) threads used in the data loader.
-  ``max_updates_per_epoch``: Optional limit on weight updates per epoch. Use `< 1` to go through all data in every epoch.
-  ``clip_gradient_norm``: Max norm for gradient clipping. Leave empty for not clipping.
-  ``target_noise_std``: Standard deviation of Gaussian noise added to labels during training. Set to zero or
   leave empty to *not* add noise.
-  ``allow_subsequent_nan_losses``: Number of allowed consecutive NaN losses before stopping.
-  ``save_weights_every``: Interval (in epochs) over which the weights of the model
   are stored to disk. ``1`` means to store the weights after each
   epoch, which is the default if not otherwise specified.

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
-  ``seq_length``: Hindcast sequence length for forecast models.
-  ``lead_time``: Forecast lead time (integer).
-  ``predict_last_n``: Number of time steps (counted backwards) used for loss calculation.
-  ``timestep_counter``: True/False. Adds a counting integer sequence as input for forecasts.
-  ``nan_handling_method``: ``masked_mean``, ``input_replacing``, or ``attention``. Strategy for handling missing input data.
-  ``nan_handling_pos_encoding_size``: Size of positional encoding for NaN handling methods.
-  ``lazy_load``: Whether to access data lazily rather than load all in-memory. Each batch is loaded dynamically. Default: `False`.

Finetune settings
-----------------

Ignored if ``mode != finetune``

-  ``finetune_modules``: List of model parts that will be trained
   during fine-tuning. Only parts listed here will be
   updated during finetuning. Check the documentation of each model to see a list
   of available module parts.

Logger settings
---------------

-  ``log_interval``: Interval at which the training loss is logged, 
   by default 10.

-  ``log_tensorboard``: True/False. If True, writes logging results into
   TensorBoard file. The default, if not specified, is True.

-  ``log_n_figures``: If a (integer) value greater than 0, saves the
   predictions as plots of that n specific (random) basins during
   validations.

-  ``log_loss_every_nth_update``: Refresh rate of logging of the loss value
   every n iterations. For example for `20`, the loss logging would be
   updated every 20 iterations (updates) during training. Logging loss has
   performance cost (waits to transfer memory from GPU to CPU instead of
   additional iterations). For example, for multimet_mean_embedding_forecast,
   a value of 5 saves 50ms per iteration on average which translates to 1.5h
   given 2000 updates for 30 epocs.

-  ``save_git_diff``: If set to True and GoogleHydrology is a git repository
   with uncommitted changes, the git diff will be stored in the run directory.
   When using this option, make sure that your run and data directories are either
   not located inside the git repository, or that they are part of the ``.gitignore`` file.
   Otherwise, the git diff may become very large and use up a lot of disk space.
   To make sure everything is configured correctly, you can simply check that the
   output of ``git diff HEAD`` only contains your code changes.