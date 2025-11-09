# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable

import numpy as np
import torch
import torch.cuda
from numba import njit
from torch.distributions import Categorical

from googlehydrology.datautils.scaler import Scaler
from googlehydrology.utils.config import Config
from googlehydrology.utils import cmal_deterministic


def sample_pointpredictions(
    model: 'BaseModel',
    data: dict[str, torch.Tensor],
    n_samples: int,
    scaler: Scaler,
) -> dict[str, torch.Tensor]:
    """Point prediction samplers for the different uncertainty estimation approaches.

    This function provides different point sampling functions for the different uncertainty estimation approaches
    (e.g. Gaussian Mixture Models (GMM), Countable Mixtures of Asymmetric Laplacians (CMAL), Monte-Carlo Dropout (MCD);
    note: MCD can be combined with the others, by setting `mc_dropout` to `True` in the configuration file).

    There are also options to handle negative point prediction samples that arise while sampling from the uncertainty
    estimates. This functionality currently supports (a) 'clip' for directly clipping values at zero and
    (b) 'truncate' for resampling values that are below zero.

    Parameters
    ----------
    model : BaseModel
        The googlehydrology model from which to sample from.
    data : dict[str, torch.Tensor]
        Dictionary, containing input features as key-value pairs.
    n_samples : int
        The number of point prediction samples that should be created.
    scaler : Scaler
        Scaler of the run.

    Returns
    -------
    dict[str, torch.Tensor]
        Dictionary, containing the sampled model outputs for the `predict_last_n` (config argument) time steps of
        each frequency.
    """

    if model.cfg.head.lower() == 'cmal':
        samples = sample_cmal(model, data, n_samples, scaler)
    elif model.cfg.head.lower() == 'cmal_deterministic':
        samples = sample_cmal_deterministic(model, data)
    elif model.cfg.head.lower() == 'regression':
        samples = sample_mcd(
            model, data, n_samples, scaler
        )  # regression head assumes mcd
    else:
        raise NotImplementedError(
            f'Sampling mode not supported for head {model.cfg.head.lower()}!'
        )

    return samples


def _subset_target(
    parameter: dict[str, torch.Tensor], n_target: int, steps: int
) -> dict[str, torch.Tensor]:
    # determine which output neurons correspond to the n_target target variable
    start = n_target * steps
    end = (n_target + 1) * steps
    parameter_sub = parameter[:, :, start:end]

    return parameter_sub


def _handle_negative_values(
    cfg: Config,
    values: torch.Tensor,
    sample_values: Callable,
    scaler: Scaler,
    nth_target: int,
) -> torch.Tensor:
    """Handle negative samples that arise while sampling from the uncertainty estimates.

    Currently supports (a) 'clip' for directly clipping values at zero and (b) 'truncate' for resampling values
    that are below zero.

    Parameters
    ----------
    cfg : Config
        The run configuration.
    values : torch.Tensor
        Tensor with the sampled values.
    sample_values : Callable
        Sampling function to allow for repeated sampling in the case of truncation-handling.
    scaler : Scaler
        Scaler of the run.
    nth_target : int
        Index of the sampled target variable in cfg.target_variables.

    Returns
    -------
    torch.Tensor
        Bound values according to user specifications.
    """
    center = (
        scaler.scaler[cfg.target_variables[nth_target]]
        .sel(parameter='center')
        .item()
    )
    scale = (
        scaler.scaler[cfg.target_variables[nth_target]]
        .sel(parameter='scale')
        .item()
    )
    normalized_zero = -torch.tensor(center / scale).to(values)
    if cfg.negative_sample_handling.lower() == 'clip':
        values = torch.clamp(values, min=normalized_zero)
    elif cfg.negative_sample_handling.lower() == 'truncate':
        values_smaller_zero = values < normalized_zero
        try_count = 0
        while torch.any(values_smaller_zero.flatten()):
            values[values_smaller_zero] = sample_values(values_smaller_zero)
            values_smaller_zero = values < normalized_zero
            try_count += 1
            if try_count >= cfg.negative_sample_max_retries:
                break
    elif (
        cfg.negative_sample_handling is None
        or cfg.negative_sample_handling.lower() == 'none'
    ):
        pass
    else:
        raise NotImplementedError(
            f'The option {cfg.negative_sample_handling} is not supported for handling negative samples!'
        )

    return values


def _sample_asymmetric_laplacians(
    ids: list[int],
    m_sub: torch.Tensor,
    b_sub: torch.Tensor,
    t_sub: torch.Tensor,
) -> torch.Tensor:
    # The ids are used for location-specific resampling for 'truncation' in '_handle_negative_values'
    m_sub_ids = m_sub[ids]
    # sample uniformly between zero and 1
    prob = torch.FloatTensor(m_sub_ids.shape).uniform_(0, 1).to(m_sub.device)
    t_sub_ids = t_sub[ids]
    t_sub_ids_c = 1 - t_sub[ids]
    b_sub_ids = b_sub[ids]
    values = torch.where(
        prob < t_sub_ids,  # needs to be in accordance with the loss
        m_sub_ids + ((b_sub_ids * torch.log(prob / t_sub_ids)) / t_sub_ids_c),
        m_sub_ids
        - ((b_sub_ids * torch.log((1 - prob) / t_sub_ids_c)) / t_sub_ids),
    )
    return values.flatten()


class _SamplingSetup:
    def __init__(
        self, model: 'BaseModel', data: dict[str, torch.Tensor], head: str
    ):
        # make model checks:
        cfg = model.cfg
        if not cfg.head.lower() == head.lower():
            raise NotImplementedError(
                f'{head} sampling not supported for the {cfg.head} head!'
            )

        dropout_modules = [model.dropout.p]

        # Certain models don't have embedding_net(s)
        implied_statics_embedding, implied_dynamics_embedding = None, None
        if hasattr(model, 'forecast_embedding_net'):
            implied_forecast_statics_embedding = (
                model.forecast_embedding_net.statics_embedding_p_dropout
            )
            implied_forecast_dynamics_embedding = (
                model.forecast_embedding_net.dynamics_embedding_p_dropout
            )
            dropout_modules += [
                implied_forecast_statics_embedding,
                implied_forecast_dynamics_embedding,
            ]
        if hasattr(model, 'hindcast_embedding_net'):
            implied_hindcast_statics_embedding = (
                model.hindcast_embedding_net.statics_embedding_p_dropout
            )
            implied_hindcast_dynamics_embedding = (
                model.hindcast_embedding_net.dynamics_embedding_p_dropout
            )
            dropout_modules += [
                implied_hindcast_statics_embedding,
                implied_hindcast_dynamics_embedding,
            ]
        if hasattr(model, 'embedding_net'):
            implied_statics_embedding = (
                model.embedding_net.statics_embedding_p_dropout
            )
            implied_dynamics_embedding = (
                model.embedding_net.dynamics_embedding_p_dropout
            )
            dropout_modules += [
                implied_statics_embedding,
                implied_dynamics_embedding,
            ]
        # account for transformer
        implied_transformer_dropout = None
        if cfg.model.lower() == 'transfomer':
            implied_transformer_dropout = cfg.transformer_dropout
            dropout_modules.append(implied_transformer_dropout)

        max_implied_dropout = max(dropout_modules)
        # check lower bound dropout:
        if cfg.mc_dropout and max_implied_dropout <= 0.0:
            raise RuntimeError(f"""{cfg.model} with `mc_dropout` activated requires a dropout rate larger than 0.0
                               The current implied dropout-rates are:
                                  - model: {cfg.output_dropout}
                                  - statics_embedding: {implied_statics_embedding}
                                  - dynamics_embedding: {implied_dynamics_embedding}
                                  - statics_forecast_embedding: {implied_forecast_statics_embedding}
                                  - dynamics_forecast_embedding: {implied_forecast_dynamics_embedding}
                                  - statics_hindcast_embedding: {implied_hindcast_statics_embedding}
                                  - dynamics_hindcast_embedding: {implied_hindcast_dynamics_embedding}
                                  - transformer: {implied_transformer_dropout}""")
        # check upper bound dropout:
        if cfg.mc_dropout and max_implied_dropout >= 1.0:
            raise RuntimeError(f"""The maximal dropout-rate is 1. Please check your dropout-settings:
                               The current implied dropout-rates are:
                                  - model: {cfg.output_dropout}
                                  - statics_embedding: {implied_statics_embedding}
                                  - dynamics_embedding: {implied_dynamics_embedding}
                                  - statics_forecast_embedding: {implied_forecast_statics_embedding}
                                  - dynamics_forecast_embedding: {implied_forecast_dynamics_embedding}
                                  - statics_hindcast_embedding: {implied_hindcast_statics_embedding}
                                  - dynamics_hindcast_embedding: {implied_hindcast_dynamics_embedding}
                                  - transformer: {implied_transformer_dropout}""")

        # assign setup properties:
        self.cfg = cfg
        self.device = next(model.parameters()).device
        self.number_of_targets = len(cfg.target_variables)
        self.mc_dropout = cfg.mc_dropout
        self.predict_last_n = cfg.predict_last_n

        # determine appropriate frequency suffix:
        if len(self.cfg.use_frequencies) > 1:
            self.freq_suffixes = [f'_{freq}' for freq in cfg.use_frequencies]
        else:
            self.freq_suffixes = ['']

        self.batch_size_data = data[f'y{self.freq_suffixes[0]}'].shape[0]


def _get_frequency_last_n(
    predict_last_n: dict[str, int] | int,
    freq_suffix: str,
    use_frequencies: list[str],
):
    if isinstance(predict_last_n, int):
        frequency_last_n = predict_last_n
    else:
        if freq_suffix != '':
            frequency_last_n = predict_last_n[freq_suffix[1:]]
        else:
            frequency_last_n = predict_last_n[use_frequencies[0]]
    return frequency_last_n


def sample_mcd(
    model: 'BaseModel',
    data: dict[str, torch.Tensor],
    n_samples: int,
    scaler: Scaler,
) -> dict[str, torch.Tensor]:
    """MC-Dropout based point predictions sampling.

    Naive sampling. This function does `n_samples` forward passes for each sample in the batch. Currently it is
    only useful for models with dropout, to perform MC-Dropout sampling.
    Note: Calling this function will force the model to train mode (`model.train()`) and not set it back to its original
    state.

    The negative sample handling currently supports (a) 'clip' for directly clipping sample_points at zero and (b)
    'truncate' for resampling sample_points that are below zero. The mode can be defined by the config argument
    'negative_sample_handling'.

    Parameters
    ----------
    model : BaseModel
        A model with a non-probabilistic head.
    data : dict[str, torch.Tensor]
        Dictionary, containing input features as key-value pairs.
    n_samples : int
        Number of samples to generate for each input sample.
    scaler : Scaler
        Scaler of the run.

    Returns
    -------
    dict[str, torch.Tensor]
        Dictionary, containing the sampled model outputs for the `predict_last_n` (config argument) time steps of
        each frequency.
    """
    setup = _SamplingSetup(model, data, model.cfg.head)

    # force model into train mode for mc_dropout:
    if setup.mc_dropout:
        model.train()

    # sample for different frequencies and targets:
    samples = {}
    for freq_suffix in setup.freq_suffixes:
        sample_points = []
        frequency_last_n = _get_frequency_last_n(
            setup.cfg.predict_last_n, freq_suffix, setup.cfg.use_frequencies
        )

        x_d = data[f'x_d{freq_suffix}']
        some_key = list(x_d)[0]
        ids = list(range(x_d[some_key].shape[0]))

        for nth_target in range(setup.number_of_targets):
            # unbound sampling:
            def _sample_values(ids: list[int]) -> torch.Tensor:
                # The ids are used for location-specific resampling for 'truncation' in '_handle_negative_values'
                target_values = torch.zeros(
                    len(ids), frequency_last_n, n_samples
                )
                for i in range(
                    n_samples
                ):  # forward-pass for each frequency separately to guarantee independence
                    prediction = model(data)
                    value_buffer = prediction[f'y_hat{freq_suffix}'][
                        :, -frequency_last_n:, 0
                    ]
                    target_values[ids, -frequency_last_n:, i] = (
                        value_buffer.detach().cpu()
                    )
                return target_values

            values = _sample_values(ids)

            # bind values and add to sample_points:
            values = _handle_negative_values(
                setup.cfg, values, _sample_values, scaler, nth_target
            )
            sample_points.append(values)

        # add sample_points to dictionary of samples:
        freq_key = f'y_hat{freq_suffix}'
        samples.update({freq_key: torch.stack(sample_points, 2)})

    return samples


def sample_cmal_deterministic(
    model: 'BaseModel', data: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Sample 10 point predictions with the Countable Mixture of Asymmetric Laplacians (CMAL) head.

    Note: If the config setting 'mc_dropout' is true this function will force the model to train mode (`model.train()`)
    and not set it back to its original state.

    Parameters
    ----------
    model : BaseModel
        A model with a CMAL head.
    data : dict[str, torch.Tensor]
        Dictionary, containing input features as key-value pairs.

    Returns
    -------
    dict[str, torch.Tensor]
        Dictionary, containing the sampled model outputs for the `predict_last_n` (config argument) time steps of
        each frequency. The shape of the output tensor for each frequency is
        ``[batch size, predict_last_n, n_samples]``.
    """
    setup = _SamplingSetup(model, data, 'cmal_deterministic')

    # force model into train mode if mc_dropout
    if setup.mc_dropout:
        model.train()

    # Make predictions (forward pass). For CMAL head those are dist params and
    # not point predictions.
    pred = model(data)

    # Map output frequencies to final sample tensors:
    samples = {}

    # Loop over all model output frequencies (e.g., 'daily', 'hourly').
    for freq_suffix in setup.freq_suffixes:
        mu = pred[f'mu{freq_suffix}']  # means
        b = pred[f'b{freq_suffix}']  # scales
        tau = pred[f'tau{freq_suffix}']  # asymmetries
        pi = pred[f'pi{freq_suffix}']  # weights

        sample_points = [
            cmal_deterministic.generate_predictions(mu, b, tau, pi)
            .detach()
            .to('cpu', non_blocking=True)
        ]
        samples.update({f'y_hat{freq_suffix}': torch.stack(sample_points, 2)})

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    return samples


def sample_cmal(
    model: 'BaseModel',
    data: dict[str, torch.Tensor],
    n_samples: int,
    scaler: Scaler,
) -> dict[str, torch.Tensor]:
    """Sample point predictions with the Countable Mixture of Asymmetric Laplacians (CMAL) head.

    This function generates `n_samples` CMAL sample points for each entry in the batch. Concretely, the model is
    executed once (forward pass) and then the sample points are generated by sampling from the resulting mixtures.
    General information about CMAL can be found in [#]_.

    The negative sample handling currently supports (a) 'clip' for directly clipping sample_points at zero and (b)
    'truncate' for resampling sample_points that are below zero. The mode can be defined by the config argument
    'negative_sample_handling'.

    Note: If the config setting 'mc_dropout' is true this function will force the model to train mode (`model.train()`)
    and not set it back to its original state.

    Parameters
    ----------
    model : BaseModel
        A model with a CMAL head.
    data : dict[str, torch.Tensor]
        Dictionary, containing input features as key-value pairs.
    n_samples : int
        Number of samples to generate for each input sample.
    scaler : Scaler
        Scaler of the run.

    Returns
    -------
    dict[str, torch.Tensor]
        Dictionary, containing the sampled model outputs for the `predict_last_n` (config argument) time steps of
        each frequency. The shape of the output tensor for each frequency is
        ``[batch size, predict_last_n, n_samples]``.

    References
    ----------
    .. [#] D.Klotz, F. Kratzert, M. Gauch, A. K. Sampson, G. Klambauer, S. Hochreiter, and G. Nearing:
        Uncertainty Estimation with Deep Learning for Rainfall-Runoff Modelling. arXiv preprint arXiv:2012.14295,
        2020.
    """
    setup = _SamplingSetup(model, data, 'cmal')

    # force model into train mode if mc_dropout
    if setup.mc_dropout:
        model.train()

    # Make predictions (forward pass). For CMAL head those are dist params and
    # not point predictions.
    pred = model(data)

    # Map output frequencies to final sample tensors:
    samples = {}

    # Loop over all model output frequencies (e.g., 'daily', 'hourly').
    for freq_suffix in setup.freq_suffixes:
        # Get the number of time steps to predict for the current frequency
        frequency_last_n = _get_frequency_last_n(
            setup.cfg.predict_last_n, freq_suffix, setup.cfg.use_frequencies
        )

        # Extract the four parameters of the CMAL distributions.
        m = pred[f'mu{freq_suffix}']  # location means
        b = pred[f'b{freq_suffix}']  # scales
        t = pred[f'tau{freq_suffix}']  # asymmetries
        p = pred[f'pi{freq_suffix}']  # mixture weights

        sample_points = []  # (for each target parameter)
        for nth_target in range(
            setup.number_of_targets
        ):  # e.g. streamflow, temp
            # Slice each full param tensor from the model's concat'd params to get
            # only the portion relevant to the current target.
            m_target = _subset_target(
                m[:, -frequency_last_n:, :],
                nth_target,
                setup.cfg.n_distributions,
            )
            b_target = _subset_target(
                b[:, -frequency_last_n:, :],
                nth_target,
                setup.cfg.n_distributions,
            )
            t_target = _subset_target(
                t[:, -frequency_last_n:, :],
                nth_target,
                setup.cfg.n_distributions,
            )
            p_target = _subset_target(
                p[:, -frequency_last_n:, :],
                nth_target,
                setup.cfg.n_distributions,
            )

            assert (
                m_target.shape
                == b_target.shape
                == t_target.shape
                == p_target.shape
            )
            batch_size, time_steps, n_dist = m_target.shape  # WLOG

            # Make [batch, sample, time, dist] (expanded) tensor views of the targets.
            # Unsqueeze to add a dim for samples: [batch, rime, dist] -> [batch, 1, time, dist].
            # Expand to repeat the new dim without allocating new memory for it. So:
            #     [batch, 1, time, dist] -> [batch, sample, time, dist].
            m_exp = m_target.unsqueeze(1).expand(-1, n_samples, -1, -1)
            b_exp = b_target.unsqueeze(1).expand(-1, n_samples, -1, -1)
            t_exp = t_target.unsqueeze(1).expand(-1, n_samples, -1, -1)
            p_exp = p_target.unsqueeze(1).expand(-1, n_samples, -1, -1)

            # Distribute:

            # Prepare data for the Categorical dist.
            # Categorical dist descrbies a random event with a fixed number of results
            # where each one has a probability. Number of outcomes here is from 0 to
            # n_dist-1, and probabilities are given by pi.
            # Replace nan with uniform probability. Later those nans will be restored.
            p_invalid = torch.isnan(p_exp)
            p_safe = torch.where(
                p_invalid, torch.ones_like(p_exp) / n_dist, p_exp
            )

            dist = torch.distributions.Categorical(probs=p_safe)

            # Sample:

            # Draw dist index for each sample and time step from the dist dim, to get
            # [batch, sample, time].
            # And add a dim to match the shape of the expanded params for gathering,
            # for selecting which dist it is gathered from.
            choices = dist.sample().unsqueeze(-1)
            # For each param point, select from [batch, sample, time, dist] using
            # choices index which is [batch, sample, time, 1] to find which dist
            # to use, gathering that param using the index for it.
            # Then squeeze out the dist dim.
            m_sub = torch.gather(m_exp, dim=3, index=choices).squeeze(-1)
            b_sub = torch.gather(b_exp, dim=3, index=choices).squeeze(-1)
            t_sub = torch.gather(t_exp, dim=3, index=choices).squeeze(-1)

            def sample_values(ids: torch.Tensor) -> torch.Tensor:
                return _sample_asymmetric_laplacians(ids, m_sub, b_sub, t_sub)

            # Generate an initial value for every single pos via a mask of all `True`s,
            # with the _sample_asymmetric_laplacians helper.
            values_unbound = sample_values(
                ids=torch.ones_like(m_sub, dtype=torch.bool)
            )
            # Reshape it back since the helper has flattened dims.
            values_unbound = values_unbound.reshape(
                batch_size, n_samples, time_steps
            )
            # Restore nans (squeezing out the dist dim from which it was gathered)
            was_nan_mask = torch.gather(
                p_invalid, dim=3, index=choices
            ).squeeze(-1)
            values_unbound[was_nan_mask] = torch.nan

            values = _handle_negative_values(  # Resample as needed
                setup.cfg,
                values_unbound,
                sample_values=sample_values,
                scaler=scaler,
                nth_target=nth_target,
            )
            # Swap [batch, sample, time] to [batch, time, sample]
            values = values.permute(0, 2, 1).detach().cpu()
            sample_points.append(values)

        # torch.stack results for all targets into a single tensor for this freq.
        # It stacks into a new dim for the targets at dim 2, so shape should be
        # [batch, time, sample] -> [batch, time, target, sample].
        samples.update(
            {f'y_hat{freq_suffix}': torch.stack(sample_points, dim=2)}
        )

    return samples


@njit
def bernoulli_subseries_sampler(
    data: np.ndarray,
    missing_fraction: float,
    mean_missing_length: float,
    start_sampling_on: bool = True,
) -> np.ndarray:
    """Samples a timeseries according to a pair of Bernoulli processes.

    The objective is to sample subsequences of a given timeseries under two criteria:
        1)  Expected long-term sample ratio (i.e., the total fraction of points in a time series
            that are not sampled): `missing_fraction`.
        2)  Expected length of continuous subsequences sampled from the timeseries:
            `mean_missing_length`.

    This is done by sampling two Bernoulli processes with different rate parameters. One
    process samples on-shifts and one process samples off-shifts. An 'on-shift' occurs
    when the state of the sampler transitions from 'off' (not sampling) to 'on' (sampling),
    and vice-versa. The rate parameters for the on-shift and off-shift processes are
    derived from the input parameters explained above.

    Parameters
    ----------
    data : np.ndarray
        Time series data to be sampled. Must be (N, 1) where N is the length of the timeseries.
    missing_fraction : float
        Expected total fraction of points in a time series that are not sampled.
    mean_missing_length : float
        Expected length of continuous subsequences of un-sampled data from the timeseries.
    start_sampling_on: bool
        Whether to start with the sampler turned "on" (True) or "off" (False) at the first
        timestep of the timeseries.

    Returns
    -------
    A copy of the timeseries with NaN's replacing elements that were not sampled.
    """
    # Check if sampling ratio is one or zero. Avoids a divide-by-zero error.
    if missing_fraction == 0:
        return data
    if missing_fraction == 1:
        return np.full(data.shape, np.nan)

    # Check that the input data is a 1-d timeseries.
    if not (data.ndim == 1 or (data.ndim == 2 and data.shape[-1] == 1)):
        raise ValueError('Shape of timeseries data must be N or (N, 1).')

    # Check that the distribution parameters make sense.
    if mean_missing_length < missing_fraction / (1 - missing_fraction):
        raise ValueError(
            'Incompatible distribution parameters in timeseries sampling. Must be: ',
            'mean_missing_length >= missing_fraction / (1-missing_fraction).',
        )
    if missing_fraction < 0 or missing_fraction > 1:
        raise ValueError('Missing fraction must be in [0,1]')

    if mean_missing_length <= 0:
        raise ValueError('Mean missing length must be > 0.')

    # Derive Bernoulli rate parameters.
    on_shift_rate = 1 / mean_missing_length
    off_shift_rate = on_shift_rate * missing_fraction / (1 - missing_fraction)

    # Initialize storage for the samples.
    sampled_data = np.full(data.shape, np.nan)
    sampled_data[0] = data[0]
    if not start_sampling_on:
        sampled_data[0] = np.nan

    # Bernoulli sampling.
    up_switches = np.random.binomial(n=1, p=on_shift_rate, size=data.shape)
    down_switches = np.random.binomial(n=1, p=off_shift_rate, size=data.shape)

    # Sampling -> stochastic process.
    for n in range(1, len(data)):
        if np.isnan(sampled_data[n - 1]):
            if up_switches[n]:
                sampled_data[n] = data[n]
        else:
            if not down_switches[n]:
                sampled_data[n] = data[n]

    return sampled_data
