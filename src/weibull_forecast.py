#!/usr/bin/python3
"""
Direct analytical delta-method confidence interval for the expected number of
failures over future time windows.

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

This is an analytical alternative to the conservative envelope bounds based on
pointwise SF intervals.

Important notes
---------------
- For Weibull 3P this implementation follows your current analytical CI logic:
  gamma is treated as a fixed shift in the delta-method variance propagation for
  the forecast functional as well. Therefore only the 2x2 covariance of
  (log_alpha, log_beta) on gamma-shifted data is used.
- For Mixture the proportion parameter remains on linear scale p, while alpha and
  beta parameters are on log scale, matching your weibull_ci.py design.
- Installed assets are expected to have CURRENT_STATE == 'I'.
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
    Returns parameter vector and covariance matrix consistent with weibull_ci.py.

    Parameterizations:
    - Weibull 2P:      [log_alpha, log_beta]
    - Weibull 3P:      [log_alpha, log_beta]   (gamma treated as fixed shift)
    - Weibull Mixture: [log_a1, log_b1, log_a2, log_b2, p]
    - Weibull CR:      [log_a1, log_b1, log_a2, log_b2]
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
    Direct analytical delta-method CI for the scalar expected number of failures.

    Returns
    -------
    dict with:
        delta
        n_installed
        expected_failures
        lower_bound
        upper_bound
        standard_error
        variance
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
    Full forecast pipeline with direct analytical delta-method CI for expected failures.
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

    Parameters
    ----------
    deltas : float | list[float]
        Forecast horizon(s) in days.
    sort_by : str, default 'BIC'
        Criterion passed to model comparison.
    CI : float, default 0.95
        Confidence level.
    delta_ic : float, default 0.466
        Threshold used by compare_best_distribution().
    cached_results : Dict | None
        Used to access the results instead of calculating them again.
    skip_errors : bool, default True
        If True, continue processing other parts when one part fails.
    return_dataframe : bool, default False
        If True, return one concatenated pandas DataFrame instead of raw dicts.

    Returns
    -------
    dict | pandas.DataFrame
        If return_dataframe=False:
            {
                'results': {part_name: forecast_dict, ...},
                'errors': {part_name: 'error message', ...}
            }

        If return_dataframe=True:
            concatenated DataFrame for all successful parts.
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
    OUTPUT_DIR = r'C:\Users\lgroha\cernbox\Documents\Masterthesis\4_Python-Tool\CEM-IN_data_forecast\Normal_data'

    ci = 0.95
    df = forecast_all_parts_direct_delta(deltas=DEFAULT_DELTAS, CI=ci, return_dataframe=True)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = os.path.join(OUTPUT_DIR, f'weibull_forecast_{timestamp}_delta_0-466.csv')

    df.to_csv(output_path, index=False)
    logger.info(f'Forecast saved to "{output_path}" ({len(df)} rows).')