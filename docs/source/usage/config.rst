Configuration Arguments
=======================

This page provides a list of possible configuration arguments. 
Check out the file `tutorial/configs/train-config.yml` for an example of how a config file could look like.

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
-  ``assimilate``: True/False (default False). If True, data assimilation is run before each forecast during ``evaluate`` / ``infer``. The command-line flag ``--assimilate`` sets this to True regardless of the config value. Requires ``assimilation_config``.
-  ``assimilation_config``: Nested block with the data assimilation settings. See `Data assimilation settings`_ below.

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
-  ``regularization``: List of regularization terms. Supported are ``forecast_overlap`` (for the ``handoff_forecast_lstm``) and ``bg_embedding`` (the background term used by data assimilation; see :py:func:`googlehydrology.training.get_regularization_obj`).
-  ``learning_rate_strategy``: ``ConstantLR``, ``StepLR``, or ``ReduceLROnPlateau``.
-  ``initial_learning_rate``: Float. Starting learning rate.
-  ``learning_rate_drop_factor``: Factor by which to reduce the learning rate.
-  ``learning_rate_epochs_drop``: Epochs to wait before dropping LR.
-  ``batch_size``: Mini-batch size.
-  ``epochs``: Number of training epochs.
-  ``num_workers``: Number of (parallel) threads used in the data loader.
-  ``max_updates_per_epoch``: Optional limit on weight updates per epoch. Use `< 1` to go through all data in every epoch.
-  ``clip_gradient_norm``: Positive float specifying the max norm for gradient
   clipping. Leave empty to disable clipping. When enabled, each epoch logs the
   count and percentage of finite pre-clip (unscaled) gradient norms exceeding
   this threshold, plus the median, 90th, and 99th percentiles (excluding
   NaN-loss steps and counting non-finite AMP norms separately). With
   TensorBoard enabled, epoch summaries are also written under
   ``train/gradient_clipping/{threshold,checked_steps,finite_steps,nonfinite_steps,clipped_steps,clipped_fraction,norm_median,norm_p90,norm_p99}``
   (``clipped_fraction`` in ``[0, 1]`` and percentiles are omitted when
   ``finite_steps == 0``).
-  ``target_noise_std``: Standard deviation of Gaussian noise added to labels during training. Set to zero or
   leave empty to *not* add noise.
-  ``nan_sequence_probability``: Float in ``[0.0, 1.0]`` or mapping of feature
   group name to float in ``[0.0, 1.0]`` (default ``0.0``; ``null`` also maps
   to ``0.0``). Probability during training of setting an entire hindcast or
   forecast sequence of a feature group to ``NaN``. When specified as a mapping,
   unlisted groups default to ``0.0`` (useful for pinning specific products such
   as ``hres``), and shared groups use the same probability in both hindcast and
   forecast windows. Applied independently to the hindcast window and the
   forecast horizon (keeping shared groups synchronized across the overlap
   prefix of ``x_d_forecast``). Groups are processed in a random order per
   sample while guaranteeing per active input pool (hindcast pool, forecast
   pool, and shared groups in the forecast horizon when forecast-only groups
   also exist) that every timestep with at least one valid group before dropout
   retains at least one valid group afterward. Effective drop rates may be lower
   than nominal in small pools where the survivor guarantee binds, and feature
   sets across distinct groups in the same pool must be disjoint. Requires a
   model with ``supports_missing_inputs = True`` (e.g.,
   ``mean_embedding_forecast_lstm``). Finetuning runs inherit this setting from
   the base run unless overridden (set to ``0.0`` or ``null`` to disable).
-  ``nan_streak_probability``: Float in ``[0.0, 1.0]`` or mapping of feature
   group name to float in ``[0.0, 1.0]`` (default ``0.0``). Probability during
   training of dropping a contiguous sub-period of a feature group to ``NaN``,
   inspired by Google's operational FloodHub training setup. Each streak draws
   ``start ~ Uniform{0, ..., L - 1}`` and ``duration ~ Uniform{1, ..., L}``,
   clamped at ``end = min(start + duration, L)``, so later timesteps in the
   window have higher dropout probability. When both
   ``nan_sequence_probability`` and ``nan_streak_probability`` are non-zero,
   they are sampled independently per window and combined with logical OR. Uses
   the same windowing, per-pool survivor guarantee (which can split streaks at
   individual timesteps), and finetune inheritance as
   ``nan_sequence_probability``.
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
-  ``lead_time``: Forecast lead time (integer). See `Temporal alignment of forecasts`_ below.
-  ``predict_last_n``: Number of time steps (counted backwards) used for loss calculation.
-  ``timestep_counter``: True/False. Adds a counting integer sequence as input for forecasts.
-  ``nan_handling_method``: ``masked_mean``, ``input_replacing``, or ``attention``. Strategy for handling missing input data.
-  ``nan_handling_pos_encoding_size``: Size of positional encoding for NaN handling methods.
-  ``lazy_load``: Whether to access data lazily rather than load all in-memory. Each batch is loaded dynamically. Default: `False`.

NetCDF input format
-------------------

In addition to Zarr stores (and legacy Caravan ``attributes/<subdataset>/*.csv``
and ``timeseries/netcdf/<subdataset>/<basin_id>.nc`` directories), the training
and inference pipeline can read local NetCDF (``.nc`` / ``.nc4``) files directly
when ``statics_data_dir``, ``targets_data_dir``, or ``dynamics_data_dir`` (or
``data_dir``) points to a directory containing NetCDF files. Zarr stores,
legacy Caravan directory trees (``attributes/``, ``timeseries/``), and
per-product subdirectories are checked first so existing workflows are
unchanged.

Supported directory layouts (non-recursive; hidden dotfiles such as ``._*.nc``
are ignored):

1. **Multi-basin files:** One or more ``.nc`` / ``.nc4`` files directly in the
   configured directory where every file contains a ``basin`` dimension (for
   example, one combined file, non-overlapping time-split files such as
   ``2020.nc`` and ``2021.nc``, or disjoint basin-batch files). Supported for
   **statics**, **targets**, and **dynamics**.
2. **Per-basin files (``<basin_id>.nc`` or ``<basin_id>.nc4``):** One file per
   basin directly in the configured directory, without a ``basin`` dimension
   inside each file (an optional 0D scalar ``basin`` coordinate is ignored).
   Only files matching the configured basin list are opened. Supported for
   **targets** and **dynamics**.

For ``dynamics_data_dir``, users may provide either a **flat directory**
(``dynamics_data_dir/*.nc`` using Layout 1 or 2) or **product subdirectories**
(``dynamics_data_dir/<PRODUCT>/*.nc``, mirroring the MultiMet Zarr layout, where
each product subfolder uses Layout 1 or 2). Mixing multi-basin and per-basin
files among the loaded files in the same directory raises a ``ValueError``.

Required NetCDF schema:

-  **Static attributes (``statics_data_dir``):** Multi-basin files with
   dimension ``(basin)`` and variable names matching ``static_attributes``.
-  **Targets (``targets_data_dir``):** Dimensions ``(basin, date)`` in
   multi-basin files, or ``(date)`` in per-basin files, with variable names
   matching ``target_variables``.
-  **Hindcast inputs (``dynamics_data_dir``):** Dimensions ``(basin, date)``
   (or ``(basin, date, lead_time)``) in multi-basin files, or ``(date)`` (or
   ``(date, lead_time)``) in per-basin files, with variable names matching
   ``hindcast_inputs`` (and any ``union_mapping`` fallback variables).
-  **Forecast inputs (``dynamics_data_dir``):** Dimensions
   ``(basin, date, lead_time)`` in multi-basin files, or ``(date, lead_time)``
   in per-basin files, with variable names matching ``forecast_inputs``.
-  **Coordinates & precision:** Coordinate names must be exact:

   -  ``basin``: unique string basin IDs (not numeric IDs).
   -  ``date``: unique, non-overlapping daily ``datetime64`` timestamps stamped
      at midnight ``00:00:00`` UTC (no ``NaT``, duplicate, or non-midnight
      values).
   -  ``lead_time``: strictly increasing, contiguous whole-day durations
      starting at ``>= 1 day`` (e.g. ``1, 2, ...`` days) encoded with CF
      duration units (such as ``units="days"``). Only ``lead_time`` is decoded
      as ``timedelta64``, so non-``lead_time`` variables carrying time units
      (e.g., ``units="days"``) remain numeric.

   Numeric and boolean data variables (e.g. ``float64``, ``int64``, ``bool``)
   are lazily cast to ``float32`` on read. Missing variables, basins, or
   invalid coordinates raise an error immediately.

.. tip::
   For large multi-basin datasets, prefer multi-basin NetCDF files over
   thousands of per-basin files (especially when ``lazy_load: True`` is
   enabled), and use separate ``statics_data_dir``, ``targets_data_dir``, and
   ``dynamics_data_dir`` paths to avoid scanning unrelated files.

Temporal alignment of forecasts
-------------------------------

Forecast runs follow the Caravan-MultiMet convention (Shalev et al., 2026,
Section 2.1). All daily data are left-labelled in UTC, so the value stored under
date ``D`` covers ``[D 00:00, D+1 00:00)``. Forecast products are indexed by
issue date, and their ``lead_time`` coordinate is 1-indexed: ``lead_time = 1
day`` on issue date ``D`` covers the same calendar day ``D``, ``lead_time = 2
days`` covers ``D + 1``, and so on.

For a sample issued on date ``D`` the dataset therefore builds:

-  **Hindcast inputs** (2D nowcast products, and forecast products used as
   hindcast inputs at their first lead time) covering the ``seq_length``
   completed days ``[D - seq_length, ..., D - 1]``.
-  **Forecast inputs** issued on ``D`` across lead times ``1 .. lead_time``,
   valid on ``[D, ..., D + lead_time - 1]``, optionally preceded by
   ``forecast_overlap`` days of first-lead-time forecasts ending on ``D - 1``.
-  **Targets** (and the ``date`` array of a sample) ending on
   ``D + lead_time - 1``.

Result files written by evaluation are indexed by the issue date ``D`` and by
``time_step``: ``time_step = k >= 1`` is the forecast with ``lead_time = k``
(valid on ``D + k - 1``), while ``time_step <= 0`` are hindcast days
(``time_step = 0`` is ``D - 1``).

.. note::
   Before version 1.13.0, 2D hindcast inputs and targets were shifted one day
   later relative to forecast inputs, and ``time_step = k`` was valid on
   ``D + k``. Models trained with earlier versions should be retrained.

Data assimilation settings
--------------------------

Data assimilation (DA) is only supported by the ``mean_embedding_forecast_lstm``.
It is run during ``evaluate`` / ``infer`` when ``assimilate: True`` is set in
the config or the command-line flag ``--assimilate`` is given
(``run evaluate --run-dir <run> --assimilate``). For every forecast issue date,
the selected latent model components are optimized by gradient descent so that
the model output matches the observations in a window of
``assimilation_window`` time steps ending at the issue date, while a background
term keeps them close to their unassimilated (prior) values. The model weights
are never changed. See :py:mod:`googlehydrology.evaluation.assimilation` and
:py:class:`googlehydrology.utils.assimilationconfig.AssimilationConfig`.

All DA settings live in the nested ``assimilation_config`` block. The block is
validated strictly (unknown keys raise a ``ValueError``). The run config's keys
(``seq_length``, ``lead_time``, ``target_variables``, ...) are inherited by the
DA config, **except** the training keys ``epochs``, ``initial_learning_rate``,
``optimizer``, ``loss`` and ``clip_gradient_norm``: these are never inherited
and take the values of the DA block, or the DA defaults listed below.
``predict_last_n`` is not inherited either; on the DA config it is forced to
``assimilation_window`` so that the DA loss evaluates the assimilation window
rather than the forecast horizon.

-  ``assimilation_components``: Required. Either a list of component names, or a
   dictionary mapping component names to per-component options. Each option
   dict may set ``regularization_weight`` (float >= 0, defaults to the top-level
   value) and ``initial_learning_rate`` (float > 0, defaults to the top-level
   value); an empty value (``static_embedding:``) uses the defaults. Component
   names must be in the model's ``supported_assimilation_components``; for the
   ``mean_embedding_forecast_lstm`` these are ``static_embedding``,
   ``hindcast_embedding`` and ``forecast_embedding``. The ``hindcast_embedding``
   and ``forecast_embedding`` both include the shared-group contribution, so
   select both to assimilate all dynamic information.
-  ``assimilation_window``: Required, positive integer. Number of observed time
   steps before the issue time that are assimilated. The window ends at the
   issue time, so forecast steps are never assimilated. If ``seq_length`` is
   defined, it must satisfy ``1 <= assimilation_window <= seq_length - lead_time``.
-  ``initial_learning_rate``: **Required**, float > 0. Constant learning rate of
   the DA optimizer for all components (unless overridden per component). Not
   inherited from the training learning rate.
-  ``regularization_weight``: Float >= 0, default ``0.0``. Default weight of the
   background term for all components. The term is normalized by the scale of
   each sequence's prior, so the weight is dimensionless.
-  ``epochs``: Non-negative integer, default ``100``. Maximum number of
   optimization steps per batch.
-  ``optimizer``: String, default ``Adam``. Optimizer name, as in the training
   settings.
-  ``loss``: One of ``MSE``, ``NSE``, ``CMAL`` (case-insensitive), default
   ``MSE``. Loss between the model prediction and the observations in the
   window. ``CMAL`` requires a CMAL head (``n_distributions`` in the run config).
-  ``clip_gradient_norm``: Float > 0 or empty, default ``None`` (no clipping).
   Maximum gradient norm, applied to every sequence independently.
-  ``early_stopping_tolerance``: Float > 0 or empty, default ``None``
   (disabled). A sequence stops being optimized once the relative error of its
   prediction at the last window step, ``|pred - obs| / max(obs - y_min, 0.05)``
   in the scaled target space, is at most this value.

Example::

    assimilate: False  # or True; `--assimilate` also turns DA on
    assimilation_config:
      assimilation_components:
        hindcast_embedding:
          regularization_weight: 0.5
          initial_learning_rate: 0.05
        static_embedding:
      assimilation_window: 30
      initial_learning_rate: 0.01
      regularization_weight: 0.1
      epochs: 50
      loss: MSE
      clip_gradient_norm: 1.0
      early_stopping_tolerance: 0.05

DA outputs never overwrite the regular evaluation outputs: the file stems get
the suffix ``_data_assimilation``, i.e. ``test_metrics_data_assimilation.csv``
and ``test_results_data_assimilation.zarr`` (the Zarr store is always written
when DA is on, also in ``evaluate`` mode). Figures and hot-start state files are
suffixed in the same way. If the run's ``config.yml`` has no
``assimilation_config``, add the block to a copy of the config and pass it via
``--config-file`` together with ``--run-dir``; for ``evaluate`` / ``infer`` the
file given by ``--config-file`` replaces the run's ``config.yml`` entirely.

Hot-start settings
------------------

These options save and load LSTM hidden and cell states between evaluation or
inference runs so you can skip full hindcast spin-up.

-  ``save_state``: ``True``/``False`` (default ``False``). When ``True``,
   non-training evaluation and inference runs (``validation``, ``test``, and
   ``infer``; skipped for ``period='train'``) write each basin's final LSTM
   states to ``<run_dir>/hot_start_states/state_<basin>.<ext>``. Saving
   reloadable states requires ``seq_length > 0`` and ``batch_size: 1``.
-  ``save_state_format``: ``npz`` (default) or ``csv``. Controls the file format
   written when ``save_state`` is ``True``, and also sets the preferred format
   when ``hot_start_path`` points to a directory containing both ``.npz`` and
   ``.csv`` files for the same basin. The ``csv`` format writes finite
   ``float32`` values as a plain table with columns
   ``state,layer,batch,unit,value``.
-  ``hot_start_path``: Path to a saved state file (``.npz`` or ``.csv``,
   applied to all basins) or a directory of per-basin state files
   (``state_<basin>.<ext>`` or ``<basin>.<ext>``, matched case-insensitively on
   extension and preferring ``save_state_format`` if multiple formats exist;
   missing files fall back to a cold start). Hot-start runs require
   ``batch_size: 1``.

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

-  ``save_git_diff``: If set to True and OpenHydroNet is a git repository
   with uncommitted changes, the git diff will be stored in the run directory.
   When using this option, make sure that your run and data directories are either
   not located inside the git repository, or that they are part of the ``.gitignore`` file.
   Otherwise, the git diff may become very large and use up a lot of disk space.
   To make sure everything is configured correctly, you can simply check that the
   output of ``git diff HEAD`` only contains your code changes.
