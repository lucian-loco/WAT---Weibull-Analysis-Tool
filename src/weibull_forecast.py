#!/usr/bin/python3
"""
weibull_forecast.py
=====================

Direct analytical delta-method confidence interval for the expected number of failures over future time windows.

Main idea
---------
For currently installed assets with ages a_i and forecast horizon delta, define

    p_i(theta, delta) = 1 - S(a_i + delta; theta) / S(a_i; theta)

Then the expected number of failures is

    E_N(theta, delta) = sum_i p_i(theta, delta)

This script computes a direct delta-method confidence interval for E_N by:

1. Re-fitting the selected Weibull model with MLE
2. Computing the covariance matrix from the Hessian of the negative log-likelihood
3. Differentiating E_N(theta, delta) with respect to the parameter vector theta
4. Applying the scalar delta method:
       Var(E_N) ≈ grad(E_N)^T Cov(theta) grad(E_N)
5. Forming
       E_N ± z_(1-alpha/2) * sqrt(Var(E_N))

This is an analytical alternative to the conservative envelope bounds based on pointwise SF intervals.

Important notes
---------------
- For Weibull 3P this implementation follows your current analytical CI logic:
  gamma is treated as a fixed shift in the delta-method variance propagation for the forecast functional as well.
  Therefore only the 2x2 covariance of (log_alpha, log_beta) on gamma-shifted data is used.
- For Mixture the proportion parameter remains on linear scale p, while alpha and beta parameters are on log scale,
  matching your weibull_ci.py design.
- Installed assets are expected to have CURRENT_STATE == 'I'.

Dependencies: `autograd` (exact gradients of the scalar E_N functional and the negative log-likelihood Hessian),
`scipy.stats` (normal quantiles), the `reliability` package's `Fit_Weibull_*`/`Fit_Everything` classes, and
`weibull_ci._compute_covariance` (shared covariance computation with the plotting module's confidence-bound logic).

Author: Lucian Groha
"""
import warnings
import numpy as np
import pandas as pd
import scipy.stats
import autograd
import autograd.numpy as anp
from reliability.Fitters import Fit_Weibull_2P, Fit_Weibull_3P, Fit_Weibull_CR, Fit_Weibull_Mixture, Fit_Everything
from data_weibull import get_failures_and_suspensions
from weibull_ci import _compute_covariance
from utils import get_logger, DataError, ThresholdError
from weibull_evaluation import compare_best_distribution

logger = get_logger(__name__)



#-----------------------------------------------------------------------------------------------------------------------
# Function for fitting the data to every available Weibull distribution | This function is important because the import from weibull.py is not possible!
#-----------------------------------------------------------------------------------------------------------------------
def weibull_fit_best(part, sort_by='BIC', data=None):
    """
    Fit every applicable distribution from the `reliability` package's `Fit_Everything` to a part's data and
    return the comparison table, excluding non-Weibull distributions and any Weibull variant the data is statistically
    too sparse to support.

    Duplicated here (rather than imported) from `weibull_analysis.py` because importing that module directly is not
    possible in this context (see inline comment in the source).

    The exclusion logic mirrors `weibull_analysis.weibull_fit_best` exactly, based on distinct failure-time count
    and total failure count:
    - < 3 distinct failure times: excludes Weibull_3P, Weibull_CR, Weibull_Mixture.
    - < 4 distinct failure times: excludes Weibull_CR, Weibull_Mixture.
    - < 5 distinct failure times and < 16 total failures: excludes Weibull_CR, Weibull_Mixture.
    - < 5 distinct failure times and >= 16 total failures: excludes Weibull_Mixture only.
    - >= 5 distinct failure times and < 16 total failures: excludes Weibull_CR, Weibull_Mixture.
    - >= 5 distinct failure times and >= 16 total failures: no additional exclusions.

    Non-Weibull distributions (Normal, Gamma, Loglogistic, Lognormal, Gumbel, Exponential, Beta, Weibull_DS)
    are always excluded.

    Parameters
    ----------
    part : str
        Part identifier; used for data lookup (if `data` is None) and error messages.
    sort_by : str, optional
        Metric used by `Fit_Everything` to rank the fitted distributions (e.g. 'BIC', 'AICc'). Default: 'BIC'.
    data : dict, optional
        Pre-fetched data dict with 'failures'/'suspensions' keys. If None, fetched internally for `part`
        via `get_failures_and_suspensions`.

    Returns
    -------
    tuple
        (wb_data_fit_all, wb_best_distribution_name, data, fit_status):
        - wb_data_fit_all : pandas.DataFrame, goodness-of-fit results for every non-excluded distribution.
        - wb_best_distribution_name : str, name of the top-ranked distribution per `sort_by`.
        - data : dict, the (possibly cleaned) failures/suspensions data used.
        - fit_status : dict, per Weibull-variant success flag and optimizer used
          (for Weibull_2P/3P/CR/Mixture that were not excluded).

    Raises
    ------
    ThresholdError
        If there are fewer than 2 total failures.
    RuntimeError
        If `part` is falsy, or if `Fit_Everything` raises an exception.

    Notes
    -----
    Suspension records with RUNNING_TIME == 0 are dropped with a UserWarning.
    A FutureWarning from pandas about all-NA DataFrame concatenation is suppressed.
    """
    if not part:
        raise RuntimeError('Invalid request ("part" not specified)')

    warnings.filterwarnings("ignore", category=FutureWarning, message="The behavior of DataFrame concatenation with empty or all-NA entries is deprecated")

    if data is None:
        data = get_failures_and_suspensions(part)

    failure_size = len(data['failures'])
    distinct_failure_count = len(set(data['failures']))
    # suspension_size = len(data['suspensions'])
    # sample_size = failure_size + suspension_size

    if failure_size < 2:
        raise ThresholdError('Not enough failures (more than 2) to perform Weibull in data for "{0}"'.format(part))

    # Prevent zeros in the right censored data
    if data.get('suspensions') is not None and any(t == 0 for t in data['suspensions']):
        data['suspensions'] = [t for t in data['suspensions'] if t > 0]
        with warnings.catch_warnings():
            warnings.simplefilter('always', UserWarning)
            warnings.warn(f'The suspension data contained zeros as running_time. These assets have been ignored. Data need to be checked for {part}.', UserWarning)

    if data.get('suspensions') is None or len(data['suspensions']) == 0:
        data['suspensions'] = None

    # Weibull Analysis
    # see https://reliability.readthedocs.io/en/latest/API/Fitters.html for parameters description

    base_exclude = ['Weibull_DS', 'Normal_2P', 'Gamma_2P', 'Loglogistic_2P',
                   'Gamma_3P', 'Lognormal_2P', 'Lognormal_3P', 'Loglogistic_3P',
                   'Gumbel_2P', 'Exponential_2P', 'Exponential_1P', 'Beta_2P']

    if distinct_failure_count < 3:
        logger.warning(f'Less than 3 distinct failure times for "{part}"! It is not possible to fit the Weibull_3P, Weibull_Mixture and Weibull_CR. '
                       f'Therefore, these models will not be used for the fitting.')

        exclude = base_exclude + ['Weibull_3P', 'Weibull_CR', 'Weibull_Mixture']

    elif distinct_failure_count < 4:
        logger.warning(f'Less than 4 distinct failure times for "{part}": Weibull_CR and Weibull_Mixture excluded for the fitting.')

        exclude = base_exclude + ['Weibull_CR', 'Weibull_Mixture']

    elif distinct_failure_count < 5:
        if failure_size < 16:
            logger.warning(f'Less than 5 distinct failures and less than 16 failures in total for "{part}": Weibull_CR and Weibull_Mixture excluded for the fitting.')
            exclude = base_exclude + ['Weibull_CR', 'Weibull_Mixture']
        else:
            logger.warning(f'Less than 5 distinct failures but more than 16 failures in total for "{part}": Weibull_Mixture excluded for the fitting.')
            exclude = base_exclude + ['Weibull_Mixture']

    else:
        if failure_size < 16:
            logger.warning(f'Less than 16 failures in total for "{part}": Weibull_CR and Weibull_Mixture excluded for the fitting.')
            exclude = base_exclude + ['Weibull_CR', 'Weibull_Mixture']
        else:
            exclude = base_exclude

    try:
        wb = Fit_Everything(failures=data['failures'], right_censored=data['suspensions'],
                            sort_by=sort_by,
                            show_probability_plot=False, show_histogram_plot=False, show_PP_plot=False, show_best_distribution_probability_plot=False,
                            exclude=exclude,
                            print_results=False,
                            method='MLE', optimizer='Best')
    except Exception as e:
        raise RuntimeError(f'Weibull fitting all distributions failed for "{part}": {e}')

    fit_status = {}

    for dist in ["Weibull_2P", "Weibull_3P", "Weibull_CR", "Weibull_Mixture"]:
        if dist in wb.excluded_distributions:
            continue
        optimizer = getattr(wb, f"{dist}_optimizer", None)
        fit_status[dist] = {"success": optimizer is not None,
                            "optimizer": optimizer,
                            }

    wb_data_fit_all = wb.results
    wb_best_distribution_name = wb.best_distribution_name
    #print(wb_data_fit_all.to_string())

    return wb_data_fit_all, wb_best_distribution_name, data, fit_status


# ----------------------------------------------------------------------------------------------------------------------
# Model re-fit
# ----------------------------------------------------------------------------------------------------------------------
def _refit_model(model_name, failures, suspensions):
    """
    Re-fit the specified Weibull model via MLE, matching the fitter settings used elsewhere in the pipeline,
    for use as the basis of the forecast's parameter vector and covariance matrix.

    Parameters
    ----------
    model_name : str
        Name of the distribution to fit: 'Weibull_2P', 'Weibull_3P', 'Weibull_Mixture', or 'Weibull_CR'.
    failures : array-like
        Observed failure times.
    suspensions : array-like or None
        Right-censored (suspension) times; treated as None if empty.

    Returns
    -------
    reliability fitter object
        The fitted model instance (`Fit_Weibull_2P`, `Fit_Weibull_3P`, `Fit_Weibull_Mixture`, or `Fit_Weibull_CR`),
        fit via MLE with the 'best' optimizer and no confidence bounds computed internally (CI_type='none' where
        applicable, since bounds are computed separately by this module's delta method).

    Raises
    ------
    ValueError
        If `model_name` is not one of the four recognized distribution names.
    """
    right_censored = suspensions if suspensions is not None and len(suspensions) > 0 else None

    common_kwargs = dict(failures=failures,
                         right_censored=right_censored,
                         show_probability_plot=False,
                         print_results=False,
                         method='MLE',
                         optimizer='best'
    )

    if model_name == 'Weibull_2P':
        return Fit_Weibull_2P(**common_kwargs, CI=0.95, CI_type='none')

    if model_name == 'Weibull_3P':
        return Fit_Weibull_3P(**common_kwargs, CI=0.95, CI_type='none')

    if model_name == 'Weibull_Mixture':
        return Fit_Weibull_Mixture(**common_kwargs, CI=0.95)

    if model_name == 'Weibull_CR':
        return Fit_Weibull_CR(**common_kwargs, CI=0.95)

    raise ValueError(f'Unknown model name for re-fit: {model_name}')


# ----------------------------------------------------------------------------------------------------------------------
# Parameter vector and covariance, consistent with weibull_ci.py
# ----------------------------------------------------------------------------------------------------------------------
def _get_params_and_covariance(fit, failures, right_censored=None):
    """
    Extract the MLE parameter vector (on the sampling/log scale used throughout `weibull_ci.py`) and its asymptotic
    covariance matrix for a fitted Weibull model, for use in delta-method variance propagation.

    Parameterizations returned:
    - Weibull 2P:      [log_alpha, log_beta]
    - Weibull 3P:      [log_alpha, log_beta] (gamma treated as a fixed shift; covariance is computed from a Weibull_2P
                       log-likelihood on gamma-shifted data, consistent with `weibull_ci.py`)
    - Weibull Mixture: [log_a1, log_b1, log_a2, log_b2, p] (proportion p on linear scale)
    - Weibull CR:      [log_a1, log_b1, log_a2, log_b2]

    Parameters
    ----------
    fit : reliability fitter object
        An already-fitted `Fit_Weibull_2P`, `Fit_Weibull_3P`, `Fit_Weibull_Mixture`, or `Fit_Weibull_CR` instance.
    failures : array-like
        Observed failure times (original scale), used to reconstruct the negative log-likelihood
        for the Hessian computation.
    right_censored : array-like, optional
        Right-censored (suspension) times (original scale).

    Returns
    -------
    tuple
        (params, cov): `params` is the MLE parameter vector on the scale described above; `cov` is the corresponding
        covariance matrix (or None if it could not be computed — see `weibull_ci._compute_covariance`).

    Raises
    ------
    ValueError
        If `fit` is not one of the four supported fitter types.
    """
    T_f = np.asarray(failures, dtype=float)
    T_rc = np.asarray(right_censored, dtype=float) if right_censored is not None else np.array([])

    if isinstance(fit, Fit_Weibull_2P):
        params = np.array([np.log(fit.alpha), np.log(fit.beta)], dtype=float)

        def neg_loglik(p):
            return Fit_Weibull_2P.LL(anp.exp(p), T_f, T_rc)

        cov = _compute_covariance(neg_loglik, params)
        return params, cov

    if isinstance(fit, Fit_Weibull_3P):
        gamma_hat = fit.gamma
        T_f_shifted = T_f - gamma_hat
        T_rc_shifted = T_rc - gamma_hat if len(T_rc) > 0 else np.array([])

        params = np.array([np.log(fit.alpha), np.log(fit.beta)], dtype=float)

        def neg_loglik_2p(p):
            return Fit_Weibull_2P.LL(anp.exp(p), T_f_shifted, T_rc_shifted)

        cov = _compute_covariance(neg_loglik_2p, params)
        return params, cov

    if isinstance(fit, Fit_Weibull_Mixture):
        params = np.array([np.log(fit.alpha_1), np.log(fit.beta_1),
                           np.log(fit.alpha_2), np.log(fit.beta_2), fit.proportion_1], dtype=float)

        def neg_loglik(p):
            p_orig = anp.array([anp.exp(p[0]), anp.exp(p[1]), anp.exp(p[2]), anp.exp(p[3]), p[4]])

            return Fit_Weibull_Mixture.LL(p_orig, T_f, T_rc)

        cov = _compute_covariance(neg_loglik, params)
        return params, cov

    if isinstance(fit, Fit_Weibull_CR):
        params = np.array([np.log(fit.alpha_1), np.log(fit.beta_1),
                           np.log(fit.alpha_2), np.log(fit.beta_2)], dtype=float)

        def neg_loglik(p):
            p_orig = anp.array([anp.exp(p[0]), anp.exp(p[1]), anp.exp(p[2]), anp.exp(p[3])])
            return Fit_Weibull_CR.LL(p_orig, T_f, T_rc)

        cov = _compute_covariance(neg_loglik, params)
        return params, cov

    raise ValueError(f'Unsupported fit object type: {type(fit).__name__}')


# ----------------------------------------------------------------------------------------------------------------------
# Survival functions on autograd scale
# ----------------------------------------------------------------------------------------------------------------------
def _sf_2p_autograd(t, params):
    log_alpha, log_beta = params[0], params[1]
    alpha = anp.exp(log_alpha)
    beta = anp.exp(log_beta)
    t = anp.maximum(t, 0.0)

    return anp.exp(-((t / alpha) ** beta))


def _sf_3p_autograd_shifted(t_shifted, params):
    """
    Weibull 3P survival function using already gamma-shifted times:
        t_shifted = t - gamma_hat
    """
    log_alpha, log_beta = params[0], params[1]
    alpha = anp.exp(log_alpha)
    beta = anp.exp(log_beta)
    t_shifted = anp.maximum(t_shifted, 0.0)

    return anp.exp(-((t_shifted / alpha) ** beta))


def _sf_mixture_autograd(t, params):
    log_a1, log_b1, log_a2, log_b2, p = params
    a1 = anp.exp(log_a1)
    b1 = anp.exp(log_b1)
    a2 = anp.exp(log_a2)
    b2 = anp.exp(log_b2)

    t = anp.maximum(t, 0.0)
    s1 = anp.exp(-((t / a1) ** b1))
    s2 = anp.exp(-((t / a2) ** b2))

    return p * s1 + (1.0 - p) * s2


def _sf_cr_autograd(t, params):
    log_a1, log_b1, log_a2, log_b2 = params
    a1 = anp.exp(log_a1)
    b1 = anp.exp(log_b1)
    a2 = anp.exp(log_a2)
    b2 = anp.exp(log_b2)

    t = anp.maximum(t, 0.0)
    s1 = anp.exp(-((t / a1) ** b1))
    s2 = anp.exp(-((t / a2) ** b2))

    return s1 * s2


# ----------------------------------------------------------------------------------------------------------------------
# Expected number of failures as scalar function of parameter vector
# ----------------------------------------------------------------------------------------------------------------------
def _expected_failures_from_params_2p(params, ages, delta):
    ages = anp.asarray(ages, dtype=float)
    s_now = anp.clip(_sf_2p_autograd(ages, params), 1e-12, 1.0)
    s_future = anp.clip(_sf_2p_autograd(ages + delta, params), 1e-12, 1.0)
    p = 1.0 - s_future / s_now
    p = anp.clip(p, 0.0, 1.0)

    return anp.sum(p)


def _expected_failures_from_params_3p(params, ages_shifted, delta):
    """
    Weibull 3P direct delta method consistent with your existing 3P analytical CI logic:
    gamma is treated as fixed and ages are already shifted by gamma_hat.
    """
    ages_shifted = anp.asarray(ages_shifted, dtype=float)
    s_now = anp.clip(_sf_3p_autograd_shifted(ages_shifted, params), 1e-12, 1.0)
    s_future = anp.clip(_sf_3p_autograd_shifted(ages_shifted + delta, params), 1e-12, 1.0)
    p = 1.0 - s_future / s_now
    p = anp.clip(p, 0.0, 1.0)

    return anp.sum(p)


def _expected_failures_from_params_mixture(params, ages, delta):
    ages = anp.asarray(ages, dtype=float)
    s_now = anp.clip(_sf_mixture_autograd(ages, params), 1e-12, 1.0)
    s_future = anp.clip(_sf_mixture_autograd(ages + delta, params), 1e-12, 1.0)
    p = 1.0 - s_future / s_now
    p = anp.clip(p, 0.0, 1.0)

    return anp.sum(p)


def _expected_failures_from_params_cr(params, ages, delta):
    ages = anp.asarray(ages, dtype=float)
    s_now = anp.clip(_sf_cr_autograd(ages, params), 1e-12, 1.0)
    s_future = anp.clip(_sf_cr_autograd(ages + delta, params), 1e-12, 1.0)
    p = 1.0 - s_future / s_now
    p = anp.clip(p, 0.0, 1.0)

    return anp.sum(p)


# ----------------------------------------------------------------------------------------------------------------------
# Direct delta-method CI for expected number of failures
# ----------------------------------------------------------------------------------------------------------------------
def _expected_failures_direct_delta(fit, installed_running_times, delta, failures, right_censored=None, CI=0.95):
    """
    Compute the point estimate and direct analytical delta-method confidence interval for the expected number of
    failures over a single forecast horizon.

    Re-derives the parameter covariance matrix, differentiates the scalar expected-failures functional E_N(theta, delta)
    with respect to theta via `autograd`, and propagates parameter uncertainty into E_N's variance
    via Var(E_N) ≈ grad(E_N)^T · Cov(theta) · grad(E_N), forming a normal-approximation confidence interval
    around the point estimate.

    Parameters
    ----------
    fit : reliability fitter object
        Already-fitted Weibull model (`Fit_Weibull_2P`, `Fit_Weibull_3P`, `Fit_Weibull_Mixture`, or `Fit_Weibull_CR`).
    installed_running_times : array-like
        Current running times (ages) of installed assets. Non-finite or negative values are dropped.
    delta : float
        Forecast horizon (time units matching the running times, e.g. days).
    failures : array-like
        Observed failure times used to refit the covariance matrix.
    right_censored : array-like, optional
        Right-censored (suspension) times.
    CI : float, optional
        Confidence level for the interval (default: 0.95).

    Returns
    -------
    dict
        {
            'delta': float, the forecast horizon used,
            'n_installed': int, number of valid installed assets used,
            'expected_failures': float, point estimate of E_N,
            'lower_bound': float or None, lower CI bound (clipped to >= 0),
            'upper_bound': float or None, upper CI bound (clipped to <= n_installed),
            'standard_error': float or None, sqrt of the propagated variance,
            'variance': float or None, the propagated variance of E_N
        }
        Bound/SE/variance fields are None if the covariance matrix could not be computed,
        if the gradient computation failed, or if a negative variance was encountered (each case logs a UserWarning
        but still returns the point estimate).

    Raises
    ------
    DataError
        If no installed assets have valid (finite, non-negative) running times, or (for Weibull 3P) if none remain
        with running time >= gamma after shifting.
    ValueError
        If `fit` is not one of the four supported fitter types.

    Notes
    -----
    For Weibull 3P, `installed_running_times` is shifted by the fitted gamma and any resulting negative ages
    are dropped before the forecast is computed; the returned `n_installed` reflects this post-shift count.
    """
    ages = np.asarray(installed_running_times, dtype=float)
    ages = ages[np.isfinite(ages)]
    ages = ages[ages >= 0]

    if len(ages) == 0:
        raise DataError('No installed assets with valid FULL_RUNNING_TIME >= 0.')

    params, cov = _get_params_and_covariance(fit, failures, right_censored=right_censored)

    if cov is None:
        point_estimate = _expected_failures_point_estimate_only(fit, ages, delta)

        return {'delta': float(delta),
                'n_installed': int(len(ages)),
                'expected_failures': float(point_estimate),
                'lower_bound': None,
                'upper_bound': None,
                'standard_error': None,
                'variance': None
        }

    if isinstance(fit, Fit_Weibull_2P):
        scalar_fn = lambda p: _expected_failures_from_params_2p(p, ages, float(delta))
        estimate = float(scalar_fn(params))
        grad_fn = autograd.grad(scalar_fn)

    elif isinstance(fit, Fit_Weibull_3P):
        ages_shifted = ages - fit.gamma
        ages_shifted = ages_shifted[ages_shifted >= 0]

        if len(ages_shifted) == 0:
            raise DataError('No installed assets with FULL_RUNNING_TIME >= gamma for Weibull 3P forecast.')

        scalar_fn = lambda p: _expected_failures_from_params_3p(p, ages_shifted, float(delta))
        estimate = float(scalar_fn(params))
        grad_fn = autograd.grad(scalar_fn)

        ages = ages_shifted

    elif isinstance(fit, Fit_Weibull_Mixture):
        scalar_fn = lambda p: _expected_failures_from_params_mixture(p, ages, float(delta))
        estimate = float(scalar_fn(params))
        grad_fn = autograd.grad(scalar_fn)

    elif isinstance(fit, Fit_Weibull_CR):
        scalar_fn = lambda p: _expected_failures_from_params_cr(p, ages, float(delta))
        estimate = float(scalar_fn(params))
        grad_fn = autograd.grad(scalar_fn)

    else:
        raise ValueError(f'Unsupported fit object type: {type(fit).__name__}')

    try:
        grad = np.asarray(grad_fn(params), dtype=float)
    except Exception as e:
        warnings.warn(f'Gradient computation for expected failures failed: {e}', UserWarning)
        return {'delta': float(delta),
                'n_installed': int(len(ages)),
                'expected_failures': float(estimate),
                'lower_bound': None,
                'upper_bound': None,
                'standard_error': None,
                'variance': None
        }

    variance = float(grad @ cov @ grad)

    if variance < 0:
        warnings.warn(f'Negative variance encountered in direct delta method for delta={delta}: var={variance:.3e}. '
                      'No confidence interval is returned.', UserWarning)
        return {'delta': float(delta),
                'n_installed': int(len(ages)),
                'expected_failures': float(estimate),
                'lower_bound': None,
                'upper_bound': None,
                'standard_error': None,
                'variance': variance
        }

    se = float(np.sqrt(variance))
    z = float(-scipy.stats.norm.ppf((1.0 - CI) / 2.0))

    lower = max(0.0, estimate - z * se)
    upper = min(float(len(ages)), estimate + z * se)

    return {'delta': float(delta),
            'n_installed': int(len(ages)),
            'expected_failures': float(estimate),
            'lower_bound': float(lower),
            'upper_bound': float(upper),
            'standard_error': se,
            'variance': variance
    }


# ----------------------------------------------------------------------------------------------------------------------
# Point estimate helper
# ----------------------------------------------------------------------------------------------------------------------
def _expected_failures_point_estimate_only(fit, ages, delta):
    ages = np.asarray(ages, dtype=float)

    if isinstance(fit, Fit_Weibull_2P):
        alpha = fit.alpha
        beta = fit.beta
        s_now = np.exp(-((np.clip(ages, 0, None) / alpha) ** beta))
        s_future = np.exp(-((np.clip(ages + delta, 0, None) / alpha) ** beta))

    elif isinstance(fit, Fit_Weibull_3P):
        shifted_now = np.clip(ages - fit.gamma, 0, None)
        shifted_future = np.clip(ages + delta - fit.gamma, 0, None)
        s_now = np.exp(-((shifted_now / fit.alpha) ** fit.beta))
        s_future = np.exp(-((shifted_future / fit.alpha) ** fit.beta))

    elif isinstance(fit, Fit_Weibull_Mixture):
        s1_now = np.exp(-((np.clip(ages, 0, None) / fit.alpha_1) ** fit.beta_1))
        s2_now = np.exp(-((np.clip(ages, 0, None) / fit.alpha_2) ** fit.beta_2))
        s1_future = np.exp(-((np.clip(ages + delta, 0, None) / fit.alpha_1) ** fit.beta_1))
        s2_future = np.exp(-((np.clip(ages + delta, 0, None) / fit.alpha_2) ** fit.beta_2))
        s_now = fit.proportion_1 * s1_now + (1.0 - fit.proportion_1) * s2_now
        s_future = fit.proportion_1 * s1_future + (1.0 - fit.proportion_1) * s2_future

    elif isinstance(fit, Fit_Weibull_CR):
        s1_now = np.exp(-((np.clip(ages, 0, None) / fit.alpha_1) ** fit.beta_1))
        s2_now = np.exp(-((np.clip(ages, 0, None) / fit.alpha_2) ** fit.beta_2))
        s1_future = np.exp(-((np.clip(ages + delta, 0, None) / fit.alpha_1) ** fit.beta_1))
        s2_future = np.exp(-((np.clip(ages + delta, 0, None) / fit.alpha_2) ** fit.beta_2))
        s_now = s1_now * s2_now
        s_future = s1_future * s2_future

    else:
        raise ValueError(f'Unsupported fit object type: {type(fit).__name__}')

    s_now = np.clip(s_now, 1e-12, 1.0)
    s_future = np.clip(s_future, 1e-12, 1.0)
    p = np.clip(1.0 - s_future / s_now, 0.0, 1.0)
    return float(np.sum(p))


# ----------------------------------------------------------------------------------------------------------------------
# Data extraction helpers
# ----------------------------------------------------------------------------------------------------------------------
def _extract_installed_running_times(data):
    """
    Extract valid FULL_RUNNING_TIME values for all currently installed assets (CURRENT_STATE == 'I')
    from a part's data dictionary.

    Parameters
    ----------
    data : dict
        Part data dictionary containing an 'installed_assets' key (list of dicts, each expected to include
        a 'FULL_RUNNING_TIME' field).

    Returns
    -------
    list[float]
        Finite, non-negative FULL_RUNNING_TIME values for installed assets.

    Raises
    ------
    DataError
        If `installed_assets` is empty, or if no valid (finite, non-negative, non-None) FULL_RUNNING_TIME values
        remain after filtering.
    KeyError
        If any installed asset record is missing the FULL_RUNNING_TIME key.
    """
    installed_assets = data.get('installed_assets', [])
    if len(installed_assets) == 0:
        raise DataError('No installed assets with CURRENT_STATE == "I" found.')

    full_running_times = []
    for asset in installed_assets:
        if 'FULL_RUNNING_TIME' not in asset:
            raise KeyError('installed_assets must contain FULL_RUNNING_TIME for every asset.')
        value = asset.get('FULL_RUNNING_TIME')
        if value is None:
            continue
        value = float(value)
        if np.isfinite(value) and value >= 0:
            full_running_times.append(value)

    if len(full_running_times) == 0:
        raise DataError('No valid FULL_RUNNING_TIME values found for installed assets.')

    return full_running_times


# ----------------------------------------------------------------------------------------------------------------------
# Main high-level API
# ----------------------------------------------------------------------------------------------------------------------
def forecast_part_direct_delta(part, deltas, fit_table, best_model, data=None, CI=0.95):
    """
    Run the full forecast pipeline for a single part: re-fit the selected best model, compute the direct delta-method
    expected-failures confidence interval for one or more forecast horizons, and package the results together
    with a trimmed fit-quality table.

    Parameters
    ----------
    part : str
        Part identifier.
    deltas : float or list[float]
        Forecast horizon(s) in days. A scalar input is internally wrapped into a single-element list and the output is
        reshaped to reflect the scalar call.
    fit_table : pandas.DataFrame
        Goodness-of-fit results table (as produced by `weibull_fit_best`), containing at least a 'Distribution' column;
        used only to extract and display the row for `best_model`.
    best_model : str
        Name of the previously selected best-fit distribution
         ('Weibull_2P', 'Weibull_3P', 'Weibull_CR', or 'Weibull_Mixture'), used to re-fit the model for the forecast.
    data : dict, optional
        Pre-fetched dataset. If a dict keyed by part name is passed, the part's own sub-dict is extracted
        via `data[part]`; if None, the data is fetched internally via `get_failures_and_suspensions(part)`.
    CI : float, optional
        Confidence level for the forecast intervals (default: 0.95).

    Returns
    -------
    dict
        {
            'part': str,
            'best_model': str,
            'n_installed': int, number of installed assets used in the forecast,
            'results': list[dict], one entry per delta (see
              `_expected_failures_direct_delta` for the per-entry schema),
            'fit_table': pandas.DataFrame, the `best_model` row of
              `fit_table` with irrelevant columns (DS, Mu, Sigma, Lambda,
              Log-likelihood, AICc, BIC, AD) dropped
        }
        If `deltas` was passed as a scalar, 'results' contains exactly one entry (still as a single-item list).

    Raises
    ------
    RuntimeError
        If `part` is falsy.

    Notes
    -----
    Logs an informational message summarizing the forecast run (part, model, number of installed assets, deltas, CI).
    """
    if not part:
        raise RuntimeError('Invalid request: part not specified.')

    if data:
        data = data[part]
    else:
        data = get_failures_and_suspensions(part)

    failures = data['failures']
    suspensions = data.get('suspensions') or []
    installed_running_times = _extract_installed_running_times(data)

    fit = _refit_model(model_name=best_model, failures=failures, suspensions=suspensions if len(suspensions) > 0 else None)

    scalar_input = isinstance(deltas, (int, float))
    if scalar_input:
        deltas = [float(deltas)]

    results = []
    for d in deltas:
        results.append(_expected_failures_direct_delta(fit=fit, installed_running_times=installed_running_times, delta=float(d),
                                                       failures=failures, right_censored=suspensions if len(suspensions) > 0 else None,
                                                       CI=CI)
        )

    logger.info(f'Direct-delta forecast created for part="{part}", model="{best_model}", '
                f'n_installed={len(installed_running_times)}, deltas={deltas}, CI={CI}')

    # Filter fit_table to best model row only, drop unused columns
    cols_to_drop = ['DS', 'Mu', 'Sigma', 'Lambda', 'Log-likelihood', 'AICc', 'BIC', 'AD']
    fit_table_display = (fit_table[fit_table['Distribution'] == best_model].drop(columns=[c for c in cols_to_drop if c in fit_table.columns]).reset_index(drop=True))

    output = {'part': part,
              'best_model': best_model,
              'n_installed': len(installed_running_times),
              'results': results,
              'fit_table': fit_table_display
    }

    return output if not scalar_input else {'part': part,
                                            'best_model': best_model,
                                            'n_installed': len(installed_running_times),
                                            'results': [results[0]],
                                            'fit_table': fit_table_display
                                            }


def forecast_all_parts_direct_delta(deltas, CI=0.95, cached_results=None, skip_errors=True, return_dataframe=False, data_prepared=None):
    """
    Run the direct-delta forecast for every part available in cached Weibull data.

    If no cached model-selection results are supplied, this function first performs model fitting and selection
    (via `weibull_fit_best` and `compare_best_distribution`, using sort_by='CV' with a BIC fallback and delta_ic=0.466)
    for every part in the raw data cache, then runs the forecast for each part using the selected best model.

    Parameters
    ----------
    deltas : float or list[float]
        Forecast horizon(s) in days.
    CI : float, optional
        Confidence level (default: 0.95).
    cached_results : dict, optional
        Pre-computed per-part model-selection results, keyed by part name, each value containing at least 'best_model'
        and 'fit_table' (as produced by `weibull_analysis.refresh_analysis_cache`). If provided, model fitting/selection
        is skipped and `data_prepared` must supply the corresponding raw data. If None, model fitting/selection is
        performed from scratch for every part.
    skip_errors : bool, optional
        If True (default), a part that raises during forecasting is recorded in the 'errors' dict and processing
        continues with the remaining parts. If False, the exception propagates immediately.
    return_dataframe : bool, optional
        If True, return one concatenated `pandas.DataFrame` (via `forecast_to_dataframe`) instead of a dict
        of raw results. Default: False.
    data_prepared : dict, optional
        Pre-fetched raw failures/suspensions/installed-assets data per part (as returned by
        `get_failures_and_suspensions(part=None)`), used together with `cached_results` to avoid redundant data
        fetching. Required (and used) only when `cached_results` is provided.

    Returns
    -------
    dict or pandas.DataFrame
        If `return_dataframe` is False:
            {'results': {part_name: forecast_dict, ...},
             'errors': {part_name: 'error message', ...}}
        If `return_dataframe` is True:
            A single concatenated DataFrame across all successful parts' forecasts (empty DataFrame if none succeeded).

    Notes
    -----
    When computing model selection from scratch (`cached_results is None`), a part that fails model fitting/selection
    is logged and skipped entirely (not included in `part_names`), independent of the `skip_errors` flag.
    """
    sort_by = 'CV'
    delta_ic = 0.466
    new_cache = {}

    if cached_results is None:
        data_all = get_failures_and_suspensions(part=None)

        for part, data in data_all.items():
            try:
                # weibull_fit_best always uses 'BIC' internally; CV is applied only in compare_best_distribution via the sort_by argument
                sort_for_fit = sort_by if sort_by != 'CV' else 'BIC'
                fit_table, _, _, fit_status = weibull_fit_best(part=part, sort_by=sort_for_fit, data=data)

                best_model, cv_used = compare_best_distribution(df=fit_table, sort_by=sort_by, part=part, data=data,
                                                                ic_fallback='BIC', delta=delta_ic, fit_status=fit_status)

                new_cache[part] = {'best_model': best_model,
                                   'fit_table': fit_table,
                                   'data': data,
                                   'cv_used': cv_used
                }

            except Exception as e:
                logger.warning(f'Analysis cache: skipped "{part}": {e}')

        cached_results = new_cache
        part_names = list(cached_results.keys())

    else:
        part_names = list(cached_results.keys())
        data_all = data_prepared

    results = {}
    errors = {}

    for part in part_names:
        try:
            cached_part_results = cached_results[part]
            fit_table = cached_part_results['fit_table']
            best_model = cached_part_results['best_model']
            forecast = forecast_part_direct_delta(part=part, deltas=deltas, fit_table=fit_table, best_model=best_model, data=data_all, CI=CI)
            results[part] = forecast

        except Exception as e:
            if skip_errors:
                errors[part] = str(e)
                logger.warning(f'Forecast failed for part="{part}": {e}')
            else:
                raise

    if return_dataframe:
        frames = [forecast_to_dataframe(single_forecast) for single_forecast in results.values()]
        if len(frames) == 0:
            return pd.DataFrame()

        return pd.concat(frames, ignore_index=True)

    return {'results': results,
            'errors': errors
    }


# ----------------------------------------------------------------------------------------------------------------------
# Output helpers
# ----------------------------------------------------------------------------------------------------------------------
def forecast_to_dataframe(forecast):
    """
    Flatten a single part's forecast output into a tidy pandas DataFrame, one row per forecast horizon (delta).

    Parameters
    ----------
    forecast : dict
        A single-part forecast dictionary as returned by `forecast_part_direct_delta`, containing 'part', 'best_model',
        'n_installed', and a 'results' list of per-delta forecast dicts.

    Returns
    -------
    pandas.DataFrame
        Columns: 'part', 'best_model', 'n_installed', 'delta', 'expected_failures', 'lower_bound', 'upper_bound',
        'standard_error', 'variance' — one row per entry in `forecast['results']`.
    """
    rows = []
    for row in forecast['results']:
        rows.append({
            'part': forecast['part'],
            'best_model': forecast['best_model'],
            'n_installed': forecast['n_installed'],
            'delta': row['delta'],
            'expected_failures': row['expected_failures'],
            'lower_bound': row['lower_bound'],
            'upper_bound': row['upper_bound'],
            'standard_error': row['standard_error'],
            'variance': row['variance'],
        })
    return pd.DataFrame(rows)


def print_forecast(forecast, CI=0.95):
    """
    Print a human-readable, formatted summary table of a single part's forecast results to stdout.

    Parameters
    ----------
    forecast : dict
        A single-part forecast dictionary as returned by `forecast_part_direct_delta`.
    CI : float, optional
        Confidence level to display in the header (default: 0.95).
        Note: this is for display only and does not affect the values already computed in `forecast`.

    Returns
    -------
    None
        Output is printed directly to stdout; nothing is returned.
    """
    print(f"\n{'=' * 96}")
    print(f"Part               : {forecast['part']}")
    print(f"Best model         : {forecast['best_model']}")
    print(f"Installed assets   : {forecast['n_installed']}")
    print(f"Method             : Direct analytical delta method on expected failures")
    print(f"Confidence level   : {CI:.1%}")
    print(f"{'=' * 96}")

    header = (
        f"{'Delta [days]':>14}  "
        f"{'Expected':>12}  "
        f"{'Lower':>12}  "
        f"{'Upper':>12}  "
        f"{'SE':>12}"
    )
    print(header)
    print('-' * len(header))

    for row in forecast['results']:
        lower_txt = f"{row['lower_bound']:.6f}" if row['lower_bound'] is not None else 'N/A'
        upper_txt = f"{row['upper_bound']:.6f}" if row['upper_bound'] is not None else 'N/A'
        se_txt = f"{row['standard_error']:.6f}" if row['standard_error'] is not None else 'N/A'

        print(
            f"{row['delta']:14.3f}  "
            f"{row['expected_failures']:12.6f}  "
            f"{lower_txt:>12}  "
            f"{upper_txt:>12}  "
            f"{se_txt:>12}"
        )

    print(f"{'=' * 96}\n")


# ----------------------------------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    from weibull_user_input import ask_sort_by, ask_ci, ask_deltas
    import os
    from datetime import datetime

    # part = input('Enter part name: ').strip()
    # sort_by = ask_sort_by(default='BIC')
    # ci = ask_ci(default=0.95)
    # deltas = ask_deltas(default=[90.0, 180.0, 365.0])
    #
    # data = get_failures_and_suspensions()
    #
    # sort_for_fit = sort_by if sort_by != 'CV' else 'BIC'
    # fit_table, _, _ = weibull_fit_best(part=part, sort_by=sort_for_fit, data=data[part])
    # best_model, _ = compare_best_distribution(df=fit_table, sort_by=sort_by, part=part, data=data[part], ic_fallback='BIC', delta=0.466)
    #
    # forecast = forecast_part_direct_delta(part=part, deltas=deltas, fit_table=fit_table, best_model=best_model, data=data, CI=ci)
    #
    # print_forecast(forecast, CI=ci)
    # print(forecast_to_dataframe(forecast).to_string(index=False))

    DEFAULT_DELTAS = [90.0, 180.0, 365.0, 1095.0, 1825.0, 3650.0]
    OUTPUT_DIR = r''

    ci = 0.95
    df = forecast_all_parts_direct_delta(deltas=DEFAULT_DELTAS, CI=ci, return_dataframe=True)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = os.path.join(OUTPUT_DIR, f'weibull_forecast_{timestamp}_delta_0-466.csv')

    df.to_csv(output_path, index=False)
    logger.info(f'Forecast saved to "{output_path}" ({len(df)} rows).')