#!/usr/bin/python3
"""
weibull.py
====================

Core Weibull reliability analysis and plotting engine for the HITDB reliability dashboard.

This module fits multiple Weibull-family models (2-Parameter, 3-Parameter, Mixture, and Competing Risks) to asset
failure/suspension data retrieved via `data_weibull`, using the `reliability` package (Fit_Weibull_2P, Fit_Weibull_3P,
Fit_Weibull_Mixture, Fit_Weibull_CR, Fit_Everything). It also compares candidate distributions
(via `weibull_evaluation`), computes analytical/bootstrap/Fisher confidence bounds (via `weibull_ci`),
forecasts expected failures (via `weibull_forecast`), and generates probability / reliability plots
(matplotlib, non-interactive 'Agg' backend) either shown interactively, saved to disk, or returned as an in-memory PNG
buffer for the dashboard.

Three independent, thread-safe in-memory caches support the dashboard:
    _weibull_analysis_cache : dict[str, dict] or None
        Best-fit model + fit table per part (see `refresh_analysis_cache`).
    _weibull_forecast_cache : dict or None
        Pre-computed failure forecasts per part (see `refresh_forecast_cache`).
    (in addition to `data_weibull._weibull_cache` for raw asset data)

Two analysis modes are exposed for interactive/manual use:
    - `automated_weibull`: batch analysis across all qualifying parts,
      prompting for thresholds/CI via `weibull_user_input`.
    - `manual_weibull`: analysis for a single specified part.

`generate_graph` is the dashboard-facing entry point that reuses the
analysis cache where possible and returns a plot as an in-memory buffer.

Author: Lucian Groha
"""
from data_weibull import get_parts
from data_weibull import get_cache_timestamp
from data_weibull import get_failures_and_suspensions
from reliability.Fitters import Fit_Everything
from reliability.Fitters import Fit_Weibull_2P
from reliability.Fitters import Fit_Weibull_3P
from reliability.Fitters import Fit_Weibull_CR
from reliability.Fitters import Fit_Weibull_Mixture
from utils import ThresholdError
from weibull_ci import weibull_cr_fisher_bounds
from weibull_ci import weibull_cr_bootstrap_bounds
from weibull_ci import weibull_cr_analytical_bounds
from weibull_ci import weibull_mixture_fisher_bounds
from weibull_ci import weibull_mixture_bootstrap_bounds
from weibull_ci import weibull_mixture_analytical_bounds
from weibull_evaluation import compare_best_distribution
from weibull_forecast import forecast_all_parts_direct_delta
import io
import os
import datetime
from zoneinfo import ZoneInfo
import warnings
import threading
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, FuncFormatter, MultipleLocator
from utils import get_logger
logger = get_logger(__name__)
# ToDo: Every Confidence Bound with analytical algorithm for consistency



#-----------------------------------------------------------------------------------------------------------------------
# Plot settings
#-----------------------------------------------------------------------------------------------------------------------
def make_minor_label_formatter(decade_span):
    def _minor_label_formatter(x, pos):
        if decade_span > 2.9:
            return ''
        log = np.floor(np.log10(x))
        mantissa = round(x / (10 ** log), 1)
        if mantissa == 2.0:
            return f'$2 \\times 10^{{{int(log)}}}$'
        if mantissa == 5.0:
            return f'$5 \\times 10^{{{int(log)}}}$'
        return ''
    return _minor_label_formatter


def sci_formatter(x, pos):
    if x == 0:
        return '0'
    exp = int(np.floor(np.log10(abs(x))))
    coeff = x / 10**exp
    if abs(coeff - 1.0) < 0.01:
        return f'$10^{{{exp}}}$'
    return f'${coeff:.4g}\\times10^{{{exp}}}$'

DAYS_PER_YEAR = 365.25
def years_formatter(x, pos):
    if x == 0:
        return '0'

    years = x / DAYS_PER_YEAR

    if years >= 1000 or years < 0.1:
        exp = int(np.floor(np.log10(abs(years))))
        coeff = years / 10**exp
        if abs(coeff - 1.0) < 0.01:
            return f'$10^{{{exp}}}$'
        return f'${coeff:.3g}\\times10^{{{exp}}}$'

    if years >= 10:
        return f'{years:.0f}'
    if years >= 1:
        return f'{years:.1f}'
    return f'{years:.2f}'


def make_minor_year_label_formatter(decade_span):
    def _minor_label_formatter(x, pos):
        if decade_span > 2.9 or x <= 0:
            return ''

        years = x / DAYS_PER_YEAR
        log = np.floor(np.log10(years))
        mantissa = round(years / (10 ** log), 1)

        if mantissa == 2.0:
            return f'$2 \\times 10^{{{int(log)}}}$'
        if mantissa == 5.0:
            return f'$5 \\times 10^{{{int(log)}}}$'
        return ''

    return _minor_label_formatter


def plot_settings(fit, upper_quantile=0.99):
    """
    Apply shared axis/figure styling to a Weibull probability plot (log-scaled x-axis in years,
    failure-probability y-axis) and annotate it with the data cache timestamp.

    Extends the x-axis upper limit to cover the given upper quantile of the fitted distribution,
    sets log-scale major/minor tick formatters (in years), resizes the figure, and stamps a "Data as of" text box
    in the bottom-right corner using `get_cache_timestamp()`.

    Parameters
    ----------
    fit : reliability fitter object
        A fitted distribution object (e.g. from Fit_Weibull_2P) exposing `.distribution.quantile()`.
    upper_quantile : float, optional
        Quantile used to determine how far the x-axis should extend (default: 0.99).

    Returns
    -------
    tuple
        (ax, fig, xmin, xmax_new): the current Axes, current Figure, and the (possibly widened) x-axis limits as floats.
    """
    ax = plt.gca()
    ax.set_xlabel('Time in years')
    ax.set_ylabel('Failure probability')
    plt.legend(loc='upper left')
    ax.set_ylim(0.001, 0.999)

    xmin, xmax = ax.get_xlim()
    x_at_upper = fit.distribution.quantile(upper_quantile)
    xmax_new = max(xmax, x_at_upper)
    ax.set_xlim(xmin * 0.8, xmax_new * 1.2)

    decade_span = np.log10((xmax_new * 1.2)) - np.log10((xmin * 0.8))

    ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=6))
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs='auto', numticks=10))
    ax.xaxis.set_major_formatter(FuncFormatter(years_formatter))
    ax.xaxis.set_minor_formatter(FuncFormatter(make_minor_year_label_formatter(decade_span)))
    # ax.xaxis.set_minor_formatter(FuncFormatter(make_minor_label_formatter(decade_span)))

    fig = plt.gcf()
    fig.set_size_inches(9.5, 6)

    # Cache timestamp box
    ts = get_cache_timestamp()
    if ts is not None:
        ts_text = f'Data as of: {ts.strftime("%d.%m.%Y %H:%M")}'
    else:
        ts_text = 'Data as of: unknown'

    ax.text(0.99, 0.01, ts_text, transform=ax.transAxes, fontsize=8, horizontalalignment='right', verticalalignment='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='gray'))

    return ax, fig, float(xmin), float(xmax_new)


# The distribution is not calculated for the full range of xvals by default, this function extends the MLE fit
def plot_extension_mix_cr(fit, fit_data, upper_quantile=0.999):
    """
    Compute extra x-values needed to extend a Mixture/Competing-Risks CDF curve beyond the range
    the `reliability` library plots by default.

    The library only evaluates the fitted curve up to roughly one decade past the maximum observed failure time;
    this function generates a log-spaced extension out to the given upper quantile, matching the point density
    used internally by the library so the extended curve looks visually consistent.

    Parameters
    ----------
    fit : reliability fitter object
        A fitted Mixture or Competing-Risks distribution object exposing `.distribution.quantile()`.
    fit_data : dict
        Data dict containing at least the key 'failures' (list of floats).
    upper_quantile : float, optional
        Quantile defining how far the extension should reach (default: 0.999).

    Returns
    -------
    tuple
        (xvals, n_points): a numpy array of log-spaced x-values to extend the plot with
        (or None if no extension is needed/possible), and the number of points generated (0 if none).
    """
    x_at_upper = fit.distribution.quantile(upper_quantile)

    log_span_lib = (np.log10(max(fit_data['failures'])) + 1) - (np.log10(min(fit_data['failures'])) - 3)
    points_lib = 1000
    density_lib = points_lib / log_span_lib

    if (np.log10(x_at_upper)) > (np.log10(max(fit_data['failures'])) + 1):
        log_span_own = np.log10(x_at_upper) - (np.log10(max(fit_data['failures'])) + 1)
        n_points = int(density_lib * log_span_own)
        if n_points < 2:
            return None, 0
        xvals = np.logspace(np.log10(max(fit_data['failures'])) + 1, np.log10(x_at_upper), n_points)
    else:
        xvals = None
        n_points = 0

    return xvals, n_points


def plot_settings_sf(xmax):
    """
    Apply shared axis/figure styling to a survival-function (reliability) plot (linear x-axis in years,
    reliability y-axis from 0 to 1.05), with gridlines and a data-cache timestamp annotation.

    Parameters
    ----------
    xmax : float
        Upper limit for the x-axis, in the same units as the plotted data
        (days; converted to years for display via `years_formatter`).

    Returns
    -------
    matplotlib.axes.Axes
        The configured current Axes object.
    """
    ax = plt.gca()
    ax.set_xlabel('Time in years')
    ax.set_ylabel('Reliability / survival probability')
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.grid(True, which='major', linestyle='--', linewidth=0.6, alpha=0.7, color='gray')
    ax.grid(True, which='minor', linestyle=':', linewidth=0.3, alpha=0.4, color='gray')
    ax.minorticks_on()
    ax.set_axisbelow(True)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, xmax)

    ax.xaxis.set_major_formatter(FuncFormatter(years_formatter))
    # ax.xaxis.set_major_formatter(FuncFormatter(sci_formatter))

    fig = plt.gcf()
    fig.set_size_inches(9.5, 6)
    fig.tight_layout()

    # Cache timestamp box
    ts = get_cache_timestamp()
    if ts is not None:
        ts_text = f'Data as of: {ts.strftime("%d.%m.%Y %H:%M")}'
    else:
        ts_text = 'Data as of: unknown'

    ax.text(0.99, 0.96, ts_text, transform=ax.transAxes, fontsize=8, horizontalalignment='right',
            verticalalignment='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='gray'))

    return ax


#ToDo: Outsource access to data and only one time for every part --> include it in the automatic weibull and give data as input for the 4 weibull distributions
#-----------------------------------------------------------------------------------------------------------------------
# Function for Weibull 2P
#-----------------------------------------------------------------------------------------------------------------------
def weibull_2p(part, ci=0.95, save_path=None, data=None, return_sf=False, selection_method='CV'):
    """
    Fit a 2-parameter Weibull distribution (MLE, best optimizer) to a part's failure/suspension data and
    produce a probability plot (and optionally a reliability/survival-function plot).

    Parameters
    ----------
    part : str
        Part identifier; used for data lookup (if `data` is None), labeling, and error messages.
    ci : float, optional
        Confidence level for the plotted confidence bounds. A value of 0.0 disables confidence bounds
        (internally substituted with 0.95 to avoid a library error, but bounds are not plotted). Default: 0.95.
    save_path : str, path-like, or file-like, optional
        If given, the probability plot (and SF plot, if requested) is saved here instead of shown interactively.
    data : dict, optional
        Pre-fetched data dict with 'failures'/'suspensions' keys (as returned by `get_failures_and_suspensions`).
        If None, it is fetched internally for `part`.
    return_sf : bool, optional
        If True, also generates and (shows/saves) a survival-function (reliability) plot
        in addition to the probability plot.
    selection_method : str, optional
        Label only — describes how this distribution was selected (e.g. 'CV', 'BIC'),
        shown in the plot title (default: 'CV').

    Returns
    -------
    pandas.DataFrame or similar
        `wb.results`, the fitted-parameter results table from `Fit_Weibull_2P`.

    Raises
    ------
    RuntimeError
        If `part` is falsy, or if the Weibull 2P fit or the survival function computation fails.

    Notes
    -----
    Suspension (right-censored) records with RUNNING_TIME == 0 are dropped with a UserWarning,
    since a zero running time is invalid for censored data.
    """
    if not part:
        raise RuntimeError('Invalid request ("part" not specified)')

    if data is None:
        data = get_failures_and_suspensions(part)

    failure_size = len(data['failures'])
    suspension_size = len(data['suspensions']) if data.get('suspensions') is not None else 0
    sample_size = failure_size + suspension_size

    # Prevent zeros in the right censored data
    if data.get('suspensions') is not None and any(t == 0 for t in data['suspensions']):
        data['suspensions'] = [t for t in data['suspensions'] if t > 0]
        with warnings.catch_warnings():
            warnings.simplefilter('always', UserWarning)
            warnings.warn(f'The suspension data contained zeros as running_time. These assets have been ignored. Data need to be checked for {part}.', UserWarning)

    if data.get('suspensions') is None or len(data['suspensions']) == 0:
        data['suspensions'] = None

    if ci == 0.0:
        ci_type = 'none'
        plot_CI = False
        ci = 0.95  # Standard value for CI in Fit_Weibull_Mixture --> CI=0.0 creates error | only affects the confidence bounds on the variables
    else:
        ci_type = 'reliability'
        plot_CI = True

    plt.figure()
    # see https://reliability.readthedocs.io/en/latest/API/Fitters.html for parameters description
    try:
        wb = Fit_Weibull_2P(failures=data['failures'], right_censored=data['suspensions'],
                            show_probability_plot=True, print_results=False,    # Results can be found in the returned variables as well
                            method='MLE', optimizer='best',                     # Run with all Optimizers: “TNC”, “L-BFGS-B”, “nelder-mead”, and “powell”
                            CI_type=ci_type, CI=ci,
                            label=f'Weibull 2 Parameter fit | MLE \n (n = {sample_size} (f: {failure_size} | s: {suspension_size})')
    except Exception as e:
        raise RuntimeError(f'Weibull 2P fitting failed for "{part}": {e}')

    plt.title(f'Weibull Probability Plot for {part} using {selection_method} as selection method with \n (α={wb.alpha:.4f}, β={wb.beta:.4f}, CI={ci:.3f})')
    ax, fig, xmin, xmax_new = plot_settings(wb)

    if return_sf:
        plt.close()
        plt.figure()

        try:
            wb_sf = wb.distribution.SF(xmin=0, xmax=xmax_new, show_plot=True, plot_CI=plot_CI, CI_type=ci_type, CI=ci)
        except Exception as e:
            raise RuntimeError(f'Creating the survival function failed for "{part}": {e}')

        plt.title(f'Reliability plot for {part} using {selection_method} as selection method with \n (α={wb.alpha:.4f}, β={wb.beta:.4f}, CI={ci:.3f})')
        plot_settings_sf(xmax=xmax_new)

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
    # print(f'Goodness of fit values for the Weibull 2P: \n {wb.goodness_of_fit} \n\n')

    return wb.results


#-----------------------------------------------------------------------------------------------------------------------
# Function for Weibull 3P
#-----------------------------------------------------------------------------------------------------------------------
def weibull_3p(part, ci=0.95, save_path=None, data=None, return_sf=False, selection_method='CV'):
    """
    Fit a 3-parameter Weibull distribution (MLE, best optimizer, with a failure-free time offset γ) to
    a part's failure/suspension data and produce a probability plot (and optionally a reliability plot).

    Parameters
    ----------
    part : str
        Part identifier; used for data lookup (if `data` is None), labeling, and error messages.
    ci : float, optional
        Confidence level for the plotted confidence bounds. A value of 0.0 disables confidence bounds
        (internally substituted with 0.95). Default: 0.95.
    save_path : str, path-like, or file-like, optional
        If given, plot(s) are saved here instead of shown interactively.
    data : dict, optional
        Pre-fetched data dict with 'failures'/'suspensions' keys. If None, fetched internally for `part`.
    return_sf : bool, optional
        If True, also generates a survival-function (reliability) plot.
    selection_method : str, optional
        Label only — shown in the plot title (default: 'CV').

    Returns
    -------
    pandas.DataFrame or similar
        `wb.results`, the fitted-parameter results table from `Fit_Weibull_3P`
        (includes α, β, and the failure-free time γ).

    Raises
    ------
    RuntimeError
        If `part` is falsy, or if the Weibull 3P fit or the survival function computation fails.

    Notes
    -----
    Suspension records with RUNNING_TIME == 0 are dropped with a UserWarning.
    The x-axis label on the probability plot is adjusted to reflect the fitted γ offset (in years).
    """
    if not part:
        raise RuntimeError('Invalid request ("part" not specified)')

    if data is None:
        data = get_failures_and_suspensions(part)

    failure_size = len(data['failures'])
    suspension_size = len(data['suspensions']) if data.get('suspensions') is not None else 0
    sample_size = failure_size + suspension_size

    # Prevent zeros in the right censored data
    if data.get('suspensions') is not None and any(t == 0 for t in data['suspensions']):
        data['suspensions'] = [t for t in data['suspensions'] if t > 0]
        with warnings.catch_warnings():
            warnings.simplefilter('always', UserWarning)
            warnings.warn(f'The suspension data contained zeros as running_time. These assets have been ignored. Data need to be checked for {part}.', UserWarning)

    if data.get('suspensions') is None or len(data['suspensions']) == 0:
        data['suspensions'] = None

    if ci == 0.0:
        ci_type = 'none'
        plot_CI = False
        ci = 0.95   # Standard value for CI in Fit_Weibull_Mixture --> CI=0.0 creates error | only affects the confidence bounds on the variables
    else:
        ci_type = 'reliability'
        plot_CI = True

    plt.figure()
    # see https://reliability.readthedocs.io/en/latest/API/Fitters.html for parameters description
    try:
        wb = Fit_Weibull_3P(failures=data['failures'], right_censored=data['suspensions'],
                            show_probability_plot=True, print_results=False,    # Results can be found in the returned variables as well
                            method='MLE', optimizer='best',                     # Run with all Optimizers: “TNC”, “L-BFGS-B”, “nelder-mead”, and “powell”
                            CI_type=ci_type, CI=ci,
                            label=f'Weibull 3 Parameter fit | MLE \n (n = {sample_size} (f: {failure_size} | s: {suspension_size})')
    except Exception as e:
        raise RuntimeError(f'Weibull 3P fitting failed for "{part}": {e}')

    plt.title(f'Weibull Probability Plot for {part} using {selection_method} as selection method with \n (α={wb.alpha:.4f}, β={wb.beta:.4f}, γ={wb.gamma:.4f}, CI={ci:.3f})')
    ax, fig, xmin, xmax_new = plot_settings(wb)
    ax.set_xlabel(f'Time in years minus failure free time γ={wb.gamma / DAYS_PER_YEAR:.3f}')

    if return_sf:
        plt.close()
        plt.figure()

        try:
            wb_sf = wb.distribution.SF(xmin=0, xmax=xmax_new, show_plot=True, plot_CI=plot_CI, CI_type=ci_type, CI=ci)
        except Exception as e:
            raise RuntimeError(f'Creating the survival function failed for "{part}": {e}')

        plt.title(f'Reliability plot for {part} using {selection_method} as selection method with \n (α={wb.alpha:.4f}, β={wb.beta:.4f}, γ={wb.gamma:.4f}, CI={ci:.3f})')
        plot_settings_sf(xmax=xmax_new)

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
    # print(f'Goodness of fit values for the Weibull 3P: \n {wb.goodness_of_fit} \n\n')

    return wb.results


#-----------------------------------------------------------------------------------------------------------------------
# Function for Weibull Mixture with 2 distributions
#-----------------------------------------------------------------------------------------------------------------------
def weibull_mixture(part, ci=0.95, save_path=None, data=None, return_sf=False, selection_method='CV'):
    """
    Fit a 2-subpopulation Weibull Mixture model to a part's failure/suspension data, plot the probability curve
    (extended beyond the library's default range via `plot_extension_mix_cr`), and optionally overlay
    analytically-computed confidence bounds and a component-colored survival-function plot.

    Parameters
    ----------
    part : str
        Part identifier; used for data lookup (if `data` is None), labeling, and error messages.
    ci : float, optional
        Confidence level for confidence bounds. 0.0 disables bound computation/plotting for the probability plot
        (an internal 0.95 is substituted to satisfy the fitter, but bounds are not drawn/filled). Default: 0.95.
    save_path : str, path-like, or file-like, optional
        If given, the plot is saved here instead of shown interactively.
    data : dict, optional
        Pre-fetched data dict with 'failures'/'suspensions' keys. If None, fetched internally for `part`.
    return_sf : bool, optional
        If True, also generates a survival-function plot with each sub-distribution component colored separately
        (component 1 green, component 2 orange, mixture blue).
    selection_method : str, optional
        Label only — shown in the plot title (default: 'CV').

    Returns
    -------
    pandas.DataFrame or similar
        `wb.results`, the fitted-parameter results table from `Fit_Weibull_Mixture` (α₁, β₁, α₂, β₂, mixing proportion).

    Raises
    ------
    ThresholdError
        If there are fewer than 4 total failures (mixture fitting is not attempted).
    RuntimeError
        If `part` is falsy, or if the Mixture fit or survival function computation fails.

    Warns
    -----
    UserWarning
        If there are fewer than 16 total failures (fit is possible but not recommended),
        or if zero-valued suspensions were dropped.

    Notes
    -----
    When `ci` is nonzero, an analytical confidence band is computed via `weibull_mixture_analytical_bounds` and
    shaded on the plot. Fisher and bootstrap bound calculations exist in the code but are currently commented out.
    """
    if not part:
        raise RuntimeError('Invalid request ("part" not specified)')

    if data is None:
        data = get_failures_and_suspensions(part)

    failure_size = len(data['failures'])
    suspension_size = len(data['suspensions']) if data.get('suspensions') is not None else 0
    sample_size = failure_size + suspension_size

    if failure_size < 4:
        raise ThresholdError('Not enough failures (more than 4) to perform Weibull Mixture in data for "{0}"'.format(part))
    elif failure_size < 16:
        with warnings.catch_warnings():
            warnings.simplefilter('always', UserWarning)
            warnings.warn(f'Less than 16 failures in total for "{part}"! It is highly recommended to use another model if there are less than 16 failures.', UserWarning)

    # Prevent zeros in the right censored data
    if data.get('suspensions') is not None and any(t == 0 for t in data['suspensions']):
        data['suspensions'] = [t for t in data['suspensions'] if t > 0]
        with warnings.catch_warnings():
            warnings.simplefilter('always', UserWarning)
            warnings.warn(f'The suspension data contained zeros as running_time. These assets have been ignored. Data need to be checked for {part}.', UserWarning)

    if data.get('suspensions') is None or len(data['suspensions']) == 0:
        data['suspensions'] = None

    ci_mc = ci

    if ci == 0.0:
        ci = 0.95   # Standard value for CI in Fit_Weibull_Mixture --> CI=0.0 creates error | only affects the confidence bounds on the variables

    plt.figure()

    # see https://reliability.readthedocs.io/en/latest/API/Fitters.html for parameters description
    try:
        wb = Fit_Weibull_Mixture(failures=data['failures'], right_censored=data['suspensions'],
                                show_probability_plot=True, print_results=False,    # Results can be found in the returned variables as well
                                optimizer='best',                                   # Run with all Optimizers: “TNC”, “L-BFGS-B”, “nelder-mead”, and “powell”
                                CI=ci,
                                label=f'Weibull Mixture fit | MLE \n (n = {sample_size} (f: {failure_size} | s: {suspension_size})')
    except Exception as e:
        raise RuntimeError(f'Weibull Mixture fitting failed for "{part}": {e}')

    try:
        xvals_ext, n_points = plot_extension_mix_cr(fit=wb, fit_data=data)
    except Exception as e:
        logger.warning(f'plot_extension_mix_cr failed for "{part}": {e}')
        xvals_ext, n_points = None, 0

    if xvals_ext is not None and len(xvals_ext) > 1:
        wb.distribution.CDF(xvals=xvals_ext, color=plt.gca().get_lines()[-1].get_color(), label='_nolegend_')

    plt.title(f'Weibull Probability Plot for {part} using {selection_method} as selection method with \n (α₁={wb.alpha_1:.4f}, β₁={wb.beta_1:.4f}, α₂={wb.alpha_2:.4f}, β₂={wb.beta_2:.4f}, \n proportion_factor={wb.proportion_1:.3f}, CI={ci:.3f})')
    ax, fig, xmin, xmax = plot_settings(wb)
    xmin_rel, xmax_rel = xmin * 0.8, xmax * 1.2

    xvals = np.logspace(np.log10(xmin_rel), np.log10(xmax_rel), 1000 + n_points)

    if return_sf:
        plt.close()
        plt.figure()

        try:
            wb_sf = wb.distribution.SF(xvals=xvals, show_plot=True, plot_components=True)
        except Exception as e:
            raise RuntimeError(f'Creating the survival function failed for "{part}": {e}')

        plt.title(f'Reliability plot for {part} using {selection_method} as selection method with \n (α₁={wb.alpha_1:.4f}, β₁={wb.beta_1:.4f}, α₂={wb.alpha_2:.4f}, β₂={wb.beta_2:.4f}, \n proportion_factor={wb.proportion_1:.3f}, CI={ci:.3f})')
        ax = plot_settings_sf(xmax=xmax_rel)
        lines = ax.get_lines()
        # lines[-3] = Weibull 1 component, lines[-2] = Weibull 2 component, lines[-1] = Mixture model
        lines[-3].set_color('C2')  # Green for component 1
        lines[-2].set_color('C1')  # Orange for component 2
        lines[-1].set_color('C0')  # Blue for mixture

    if ci_mc != 0.0:
        # Calculation of the Confidence Interval analytically:----------------------------------------------------------
        lower_analytic, upper_analytic, p_lower, p_upper = weibull_mixture_analytical_bounds(fit=wb, xvals=xvals, failures=data['failures'],
                                                                           right_censored=data['suspensions'], CI=ci, return_sf=return_sf)

        if lower_analytic is not None and upper_analytic is not None:
            ax.fill_between(
                xvals,
                lower_analytic,
                upper_analytic,
                alpha=0.3,
                # label=f'{int(ci * 100)}% analytical CI'
            )
        # --------------------------------------------------------------------------------------------------------------

        # Calculation of the Confidence Interval:-----------------------------------------------------------------------
        # lower_mc, upper_mc = weibull_mixture_fisher_bounds(fit=wb, xvals=xvals, failures=data['failures'],
        #                                              right_censored=data['suspensions'], CI=ci)
        #
        # if lower_mc is not None and upper_mc is not None:
        #     ax.fill_between(xvals, lower_mc, upper_mc, alpha=0.3, label=f'{int(ci * 100)}% numerical Fisher CI')
        # --------------------------------------------------------------------------------------------------------------
        # print(f'Starting with the bootstrapping...')
        # Calculation of the Confidence Interval with bootstrap:--------------------------------------------------------
        # lower_bootstrap, upper_bootstrap = weibull_mixture_bootstrap_bounds(xvals=xvals, failures=data['failures'],
        #                                                                    right_censored=data['suspensions'], CI=ci)
        #
        # if lower_bootstrap is not None and upper_bootstrap is not None:
        #     ax.fill_between(xvals, lower_bootstrap, upper_bootstrap, alpha=0.3, facecolor='none', edgecolor='fuchsia', label=f'{int(ci * 100)}% Bootstrapping CI', hatch='oo')
        # --------------------------------------------------------------------------------------------------------------

    plt.legend(loc='best')

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
    # print(f'Goodness of fit values for the Weibull Mixture: \n {wb.goodness_of_fit} \n\n')

    return wb.results


#-----------------------------------------------------------------------------------------------------------------------
# Function for Weibull Competing Risks with 2 distributions
#-----------------------------------------------------------------------------------------------------------------------
def weibull_cr(part, ci=0.95, save_path=None, data=None, return_sf=False, selection_method='CV'):
    """
    Fit a 2-population Weibull Competing Risks model to a part's failure/suspension data, plot the probability curve
    (extended via `plot_extension_mix_cr`), and optionally overlay analytically-computed confidence bounds
    and a component-colored survival-function plot.

    Parameters
    ----------
    part : str
        Part identifier; used for data lookup (if `data` is None), labeling, and error messages.
    ci : float, optional
        Confidence level for confidence bounds. 0.0 disables bound computation/plotting
        (an internal 0.95 is substituted to satisfy the fitter). Default: 0.95.
    save_path : str, path-like, or file-like, optional
        If given, the plot is saved here instead of shown interactively.
    data : dict, optional
        Pre-fetched data dict with 'failures'/'suspensions' keys. If None, fetched internally for `part`.
    return_sf : bool, optional
        If True, also generates a survival-function plot with each risk component colored separately
        (component 1 green, component 2 orange, combined model blue).
    selection_method : str, optional
        Label only — shown in the plot title (default: 'CV').

    Returns
    -------
    pandas.DataFrame or similar
        `wb.results`, the fitted-parameter results table from `Fit_Weibull_CR`
        (α₁, β₁, α₂, β₂ for the two competing risks).

    Raises
    ------
    ThresholdError
        If there are fewer than 4 total failures (competing-risks fitting is not attempted).
    RuntimeError
        If `part` is falsy, or if the Competing Risks fit or survival function computation fails.

    Warns
    -----
    UserWarning
        If there are fewer than 16 total failures, or if zero-valued suspensions were dropped.

    Notes
    -----
    When `ci` is nonzero, an analytical confidence band is computed via `weibull_cr_analytical_bounds` and
    shaded on the plot. Fisher and bootstrap bound calculations exist in the code but are currently commented out.
    """
    if not part:
        raise RuntimeError('Invalid request ("part" not specified)')

    if data is None:
        data = get_failures_and_suspensions(part)

    failure_size = len(data['failures'])
    suspension_size = len(data['suspensions']) if data.get('suspensions') is not None else 0
    sample_size = failure_size + suspension_size

    if failure_size < 4:
        raise ThresholdError('Not enough failures (more than 4) to perform Weibull Competing Risks in data for "{0}"'.format(part))
    elif failure_size < 16:
        with warnings.catch_warnings():
            warnings.simplefilter('always', UserWarning)
            warnings.warn(f'Less than 16 failures in total for "{part}"! It is highly recommended to use another model if there are less than 16 failures.', UserWarning)

    # Prevent zeros in the right censored data
    if data.get('suspensions') is not None and any(t == 0 for t in data['suspensions']):
        data['suspensions'] = [t for t in data['suspensions'] if t > 0]
        with warnings.catch_warnings():
            warnings.simplefilter('always', UserWarning)
            warnings.warn(f'The suspension data contained zeros as running_time. These assets have been ignored. Data need to be checked for {part}.', UserWarning)

    if data.get('suspensions') is None or len(data['suspensions']) == 0:
        data['suspensions'] = None

    ci_mc = ci

    if ci == 0.0:
        ci = 0.95  # Standard value for CI in Fit_Weibull_Mixture --> CI=0.0 creates error | only affects the confidence bounds on the variables

    plt.figure()

    # see https://reliability.readthedocs.io/en/latest/API/Fitters.html for parameters description
    try:
        wb = Fit_Weibull_CR(failures=data['failures'], right_censored=data['suspensions'],
                            show_probability_plot=True, print_results=False,    # Results can be found in the returned variables as well
                            optimizer='best',                                   # Run with all Optimizers: “TNC”, “L-BFGS-B”, “nelder-mead”, and “powell”
                            CI=ci,
                            label=f'Weibull Competing Risk fit | MLE \n (n = {sample_size} (f: {failure_size} | s: {suspension_size})')
    except Exception as e:
        raise RuntimeError(f'Weibull Competing Risk fitting failed for "{part}": {e}')

    try:
        xvals_ext, n_points = plot_extension_mix_cr(fit=wb, fit_data=data)
    except Exception as e:
        logger.warning(f'plot_extension_mix_cr failed for "{part}": {e}')
        xvals_ext, n_points = None, 0

    if xvals_ext is not None and len(xvals_ext) > 1:
        wb.distribution.CDF(xvals=xvals_ext, color=plt.gca().get_lines()[-1].get_color(), label='_nolegend_')

    plt.title(f'Weibull Probability Plot for {part} using {selection_method} as selection method with \n (α₁={wb.alpha_1:.4f}, β₁={wb.beta_1:.4f}, α₂={wb.alpha_2:.4f}, β₂={wb.beta_2:.4f}, CI={ci:.3f})')
    ax, fig, xmin, xmax = plot_settings(wb)
    xmin_rel, xmax_rel = xmin * 0.8, xmax * 1.2

    xvals = np.logspace(np.log10(xmin_rel), np.log10(xmax_rel), 1000 + n_points)

    if return_sf:
        plt.close()
        plt.figure()

        try:
            wb_sf = wb.distribution.SF(xvals=xvals, show_plot=True, plot_components=True)
        except Exception as e:
            raise RuntimeError(f'Creating the survival function failed for "{part}": {e}')

        plt.title(f'Reliability plot for {part} using {selection_method} as selection method with \n (α₁={wb.alpha_1:.4f}, β₁={wb.beta_1:.4f}, α₂={wb.alpha_2:.4f}, β₂={wb.beta_2:.4f}, CI={ci:.3f})')
        ax = plot_settings_sf(xmax=xmax_rel)
        lines = ax.get_lines()
        # lines[-3] = Weibull 1 component, lines[-2] = Weibull 2 component, lines[-1] = Mixture model
        lines[-3].set_color('C2')  # Green for component 1
        lines[-2].set_color('C1')  # Orange for component 2
        lines[-1].set_color('C0')  # Blue for mixture

    if ci_mc != 0.0:
        # Calculation of the Confidence Interval analytically:----------------------------------------------------------
        lower_analytical, upper_analytical, _, _ = weibull_cr_analytical_bounds(fit=wb, xvals=xvals, failures=data['failures'],
                                                                                right_censored=data['suspensions'], CI=ci, return_sf=return_sf)

        if lower_analytical is not None and upper_analytical is not None:
            ax.fill_between(
                xvals,
                lower_analytical,
                upper_analytical,
                alpha=0.3,
                # label=f'{int(ci * 100)}% analytical CI'
            )

        # Calculation of the Confidence Interval:-----------------------------------------------------------------------
        # lower_mc, upper_mc = weibull_cr_fisher_bounds(fit=wb, xvals=xvals, failures=data['failures'],
        #                                               right_censored=data['suspensions'], CI=ci)
        #
        # if lower_mc is not None and upper_mc is not None:
        #     ax.fill_between(xvals, lower_mc, upper_mc, alpha=0.3, label=f'{int(ci * 100)}% numerical Fisher CI')
        # --------------------------------------------------------------------------------------------------------------
        # print(f'Starting with the bootstrapping...')
        # Calculation of the Confidence Interval with bootstrap:--------------------------------------------------------
        # lower_bootstrap, upper_bootstrap = weibull_cr_bootstrap_bounds(xvals=xvals, failures=data['failures'],
        #                                                                     right_censored=data['suspensions'], CI=ci)
        #
        # if lower_bootstrap is not None and upper_bootstrap is not None:
        #     ax.fill_between(xvals, lower_bootstrap, upper_bootstrap, alpha=0.3, facecolor='none', edgecolor='fuchsia', label=f'{int(ci * 100)}% Bootstrapping CI', hatch='oo')
        # --------------------------------------------------------------------------------------------------------------

    plt.legend(loc='best')

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
    # print(f'Goodness of fit values for the Weibull CR: \n {wb.goodness_of_fit} \n\n')

    return wb.results


#-----------------------------------------------------------------------------------------------------------------------
# Function for fitting the data to every available Weibull distribution --> AICc and BIC for every distribution in returned result object
#-----------------------------------------------------------------------------------------------------------------------
def weibull_fit_best(part, sort_by='BIC', data=None):
    """
    Fit every applicable distribution from the `reliability` package's `Fit_Everything` to a part's data and
    return the comparison table, excluding non-Weibull distributions and any Weibull variant that the data is
    statistically too sparse to support.

    The exclusion logic is based on the number of distinct failure times and total failure count:
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
        Pre-fetched data dict with 'failures'/'suspensions' keys. If None, fetched internally for `part`.

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


#-----------------------------------------------------------------------------------------------------------------------
# Perform an automated Weibull Analysis to the HITDB Data by using different Weibull distributions
#-----------------------------------------------------------------------------------------------------------------------
_weibull_analysis_cache = None
_analysis_cache_timestamp = None
_analysis_cache_lock = threading.Lock()

_weibull_forecast_cache = None
_forecast_cache_lock = threading.Lock()


def refresh_analysis_cache(sort_by='CV', delta_ic=0.466):
    """
    Pre-compute Weibull model selection (best-fit distribution + fit table) for every part currently held
    in the raw data cache (`data_weibull._weibull_cache`), storing the results in the module's analysis cache
    (`_weibull_analysis_cache`) for fast reuse by `generate_graph` and other dashboard routes.

    Must be called AFTER `data_weibull.refresh_cache()` so the raw data cache is populated;
    if it is empty, the refresh is skipped with a warning.

    Parameters
    ----------
    sort_by : str, optional
        Distribution-selection criterion passed to `compare_best_distribution` (e.g. 'CV', 'BIC', 'AICc').
        Note that `weibull_fit_best` itself always sorts internally by 'BIC' when `sort_by == 'CV'`;
        the actual CV-based comparison happens in `compare_best_distribution`. Default: 'CV'.
    delta_ic : float, optional
        Delta threshold used by `compare_best_distribution` when falling back to
        an information-criterion-based comparison. Default: 0.466.

    Returns
    -------
    None
        Updates the module-level `_weibull_analysis_cache` and `_analysis_cache_timestamp` in place (thread-safe).

    Notes
    -----
    Parts that fail to fit are skipped and logged (not raised), so a single bad part does not abort
    the whole cache refresh.
    """
    global _weibull_analysis_cache, _analysis_cache_timestamp

    from data_weibull import _weibull_cache

    if _weibull_cache is None:
        logger.warning('Analysis cache refresh skipped — data cache is empty.')
        return

    logger.info('Weibull analysis cache refresh started...')

    new_cache = {}
    errors = {}

    # get_failures_and_suspensions(None) reads from _weibull_cache
    all_data = get_failures_and_suspensions(part=None)

    for part, data in all_data.items():
        try:
            # weibull_fit_best always uses 'BIC' internally; CV is applied only in compare_best_distribution via the sort_by argument
            sort_for_fit = sort_by if sort_by != 'CV' else 'BIC'
            fit_table, _, _, fit_status = weibull_fit_best(part=part, sort_by=sort_for_fit, data=data)

            best_model, cv_used = compare_best_distribution(df=fit_table, sort_by=sort_by, part=part, data=data, ic_fallback='BIC', delta=delta_ic, fit_status=fit_status)

            new_cache[part] = {'best_model': best_model,
                               'fit_table': fit_table,
                               'data': data,
                               'cv_used': cv_used
            }

        except Exception as e:
            errors[part] = str(e)
            logger.warning(f'Analysis cache: skipped "{part}": {e}')

    with _analysis_cache_lock:
        _weibull_analysis_cache = new_cache
        _analysis_cache_timestamp = datetime.datetime.now(tz=ZoneInfo('Europe/Zurich'))

    logger.info(f'Analysis cache refresh completed: {len(new_cache)} parts OK, {len(errors)} parts skipped.')

    if errors:
        logger.debug(f'Analysis cache errors: {errors}')


def refresh_forecast_cache(deltas=None, ci=0.95):
    """
    Pre-compute expected-failure forecasts (over a set of future time horizons) for every part in the analysis cache,
    storing the result in the module's forecast cache (`_weibull_forecast_cache`).

    Must be called AFTER `refresh_analysis_cache()`, since it consumes `_weibull_analysis_cache` as input.
    If the raw data cache is empty, the refresh is skipped with a warning.

    Parameters
    ----------
    deltas : list[float], optional
        Forecast horizons in days. Defaults to [90.0, 180.0, 365.0, 1095.0, 1825.0] (~3mo, 6mo, 1y, 3y, 5y)
        if not provided.
    ci : float, optional
        Confidence level used for the forecast confidence bounds (default: 0.95).

    Returns
    -------
    None
        Updates the module-level `_weibull_forecast_cache` in place (thread-safe).
        If no analysis cache is available, an empty result dict is stored instead.
    """
    global _weibull_forecast_cache

    from data_weibull import _weibull_cache

    if _weibull_cache is None:
        logger.warning('Forecast cache refresh skipped — data cache is empty.')
        return

    if deltas is None:
        deltas = [90.0, 180.0, 365.0, 1095.0, 1825.0]

    logger.info('Forecast cache refresh started...')

    weibull_analysis_cached_results = _weibull_analysis_cache
    data_prepared = get_failures_and_suspensions(part=None)

    if weibull_analysis_cached_results:
        result = forecast_all_parts_direct_delta(deltas=deltas, CI=ci, cached_results=weibull_analysis_cached_results, skip_errors=True, data_prepared=data_prepared)
    else:
        result = {}
        logger.info(f'Calculation of results for the expected number of failures were not possible because there are no weibull_analysis_cached_results.')

    with _forecast_cache_lock:
        _weibull_forecast_cache = result

    n_ok = len(result.get('results', {}))
    n_err = len(result.get('errors', {}))

    logger.info(f'Forecast cache refresh completed: {n_ok} parts OK, {n_err} skipped.')


def get_analysis_cache():
    return _weibull_analysis_cache


def get_analysis_cache_timestamp():
    return _analysis_cache_timestamp


def get_forecast_cache():
    return _weibull_forecast_cache



def automated_weibull(save_path=None, return_sf=False, delta=0.466):
    """
    Run a full, interactive batch Weibull analysis: prompts the user for failure/distinct thresholds, sort criterion,
    and confidence level, finds all qualifying parts, fits every candidate distribution to each, selects
    the best distribution per part, generates and saves a plot for each part's best model,
    and returns the aggregated results.

    Parameters
    ----------
    save_path : str, optional
        Directory in which one plot PNG per part is saved (named '{sort_by}_plot_{part}.png').
        Required implicitly since each plot is saved rather than shown.
    return_sf : bool, optional
        If True, each generated plot also includes a survival-function (reliability) plot
        in addition to the probability plot.
    delta : float, optional
        Delta threshold used by `compare_best_distribution` for the information-criterion fallback comparison
        (default: 0.466).

    Returns
    -------
    tuple
        (parts_fit_results, parts_data_fit_all):
        - parts_fit_results : dict[str, Any], per-part fitted-parameter results table
          from the chosen distribution's fitter function.
        - parts_data_fit_all : dict[str, pandas.DataFrame], per-part goodness-of-fit comparison table
          across all fitted distributions.

        Notes
        -----
        Prompts interactively via `ask_threshold`, `ask_sort_by`, and `ask_ci`
        (imported from `weibull_user_input` in the `__main__` block). Parts
        whose best-fit distribution name is not recognized are skipped with a
        RuntimeWarning.
        """
    failure_threshold = ask_threshold("Failure threshold", default=4)
    distinct_threshold = ask_threshold("Distinct threshold", default=2)
    sort_by = ask_sort_by(default='CV')
    ci = ask_ci(default=0.95)
    ic_fallback = 'BIC'

    print(f"\n→ Starting search for parts with failure_threshold={failure_threshold}, distinct_threshold={distinct_threshold}, sort_by={sort_by} and CI={ci}\n")

    part_names_hit = get_parts(failure_threshold=failure_threshold, distinct_threshold=distinct_threshold)

    print(f"\n→ Starting analysis for {len(part_names_hit)} parts...")

    parts_data_fit_all = []
    parts_best_distribution_names = []

    # Internal sort column for FitEverything if user chose CV
    sort_for_fit = sort_by if sort_by != 'CV' else 'BIC'

    for part in part_names_hit:
        wb_data_fit_all, wb_best_distribution_name, data, fit_status = weibull_fit_best(part=part, sort_by=sort_for_fit)

        wb_data_fit_all['PART'] = part

        compared_best, cv_used = compare_best_distribution(df=wb_data_fit_all, sort_by=sort_by, part=part, data=data, ic_fallback=ic_fallback, delta=delta, fit_status=fit_status)

        selection_used = 'CV' if cv_used else ic_fallback
        wb_data_fit_all['SELECTION_METHOD'] = selection_used

        wb_best_distribution_row = pd.DataFrame({'PART': [part], 'BEST_DISTRIBUTION': [compared_best], 'SELECTION_METHOD': [selection_used]})

        parts_data_fit_all.append(wb_data_fit_all)
        parts_best_distribution_names.append(wb_best_distribution_row)

    parts_data_fit_all = pd.concat(parts_data_fit_all, ignore_index=True)
    parts_data_fit_all = {name: group for name, group in parts_data_fit_all.groupby('PART')}

    parts_best_distribution_names = pd.concat(parts_best_distribution_names, ignore_index=True)

    fitter_map = {'Weibull_2P':         lambda p, sp, sm: weibull_2p(part=p, ci=ci, save_path=sp, return_sf=return_sf, selection_method=sm),
                  'Weibull_3P':         lambda p, sp, sm: weibull_3p(part=p, ci=ci, save_path=sp, return_sf=return_sf, selection_method=sm),
                  'Weibull_Mixture':    lambda p, sp, sm: weibull_mixture(part=p, ci=ci, save_path=sp, return_sf=return_sf, selection_method=sm),
                  'Weibull_CR':         lambda p, sp, sm: weibull_cr(part=p, ci=ci, save_path=sp, return_sf=return_sf, selection_method=sm)}

    parts_fit_results = {}

    print(f"\n→ Found the best distribution for these parts, now calculating the plots...")

    for _, row in parts_best_distribution_names.iterrows():
        part = row['PART']
        best_distribution = row['BEST_DISTRIBUTION']
        selection_used = row['SELECTION_METHOD']

        fit_function = fitter_map.get(best_distribution)

        if fit_function is None:
            with warnings.catch_warnings():
                warnings.simplefilter('always', RuntimeWarning)
                warnings.warn(f'Unknown distribution "{best_distribution}" for "{part}" --> skipped.', RuntimeWarning)
            continue

        part_save_path = os.path.join(save_path, f'{sort_by}_plot_{part}.png')

        parts_fit_results[part] = fit_function(part, part_save_path, selection_used)

    return parts_fit_results, parts_data_fit_all


#-----------------------------------------------------------------------------------------------------------------------
# Perform a manual Weibull Analysis to one specific part by using different Weibull distributions
#-----------------------------------------------------------------------------------------------------------------------
def manual_weibull(part, return_sf=False, delta=0.466):
    """
    Run an interactive Weibull analysis for a single specified part: prompts for sort criterion and confidence level,
    fits every candidate distribution, selects the best one, and generates its plot interactively (shown, not saved).

    Parameters
    ----------
    part : str
        Part identifier to analyze.
    return_sf : bool, optional
        If True, also generates a survival-function (reliability) plot.
    delta : float, optional
        Delta threshold used by `compare_best_distribution` for the information-criterion fallback comparison
        (default: 0.466).

    Returns
    -------
    tuple or None
        (wb_results, wb_data_fit_all, wb_best_distribution_name) where wb_results is the fitted-parameter table
        from the chosen distribution's fitter, wb_data_fit_all is the full goodness-of-fit comparison table,
        and wb_best_distribution_name is the name returned by `weibull_fit_best`. Returns None if the selected best
        distribution name is not recognized (with a RuntimeWarning).

    Raises
    ------
    RuntimeError
        If `part` is falsy.

    Notes
    -----
    Prompts interactively via `ask_sort_by` and `ask_ci` (imported from `weibull_user_input` in the `__main__` block).
    """
    if not part:
        raise RuntimeError('Invalid request ("part" not specified)')

    sort_by = ask_sort_by(default='BIC')
    ci = ask_ci(default=0.95)
    ic_fallback = 'BIC'

    print(f"\n→ Starting Analysis for {part} with sort_by={sort_by} and CI={ci}\n")

    wb_data_fit_all, wb_best_distribution_name, data, fit_status = weibull_fit_best(part=part, sort_by=sort_by if sort_by != 'CV' else 'BIC')

    compared_best, cv_used = compare_best_distribution(df=wb_data_fit_all, sort_by=sort_by, part=part, data=data, ic_fallback=ic_fallback, delta=delta, fit_status=fit_status)

    selection_used = 'CV' if cv_used else ic_fallback

    fitter_map = {'Weibull_2P': lambda p: weibull_2p(part=p, ci=ci, save_path=None, data=data, return_sf=return_sf, selection_method=selection_used),
                  'Weibull_3P': lambda p: weibull_3p(part=p, ci=ci, save_path=None, data=data, return_sf=return_sf, selection_method=selection_used),
                  'Weibull_Mixture': lambda p: weibull_mixture(part=p, ci=ci, save_path=None, data=data, return_sf=return_sf, selection_method=selection_used),
                  'Weibull_CR': lambda p: weibull_cr(part=p, ci=ci, save_path=None, data=data, return_sf=return_sf, selection_method=selection_used)}

    fit_function = fitter_map.get(compared_best)

    if fit_function is None:
        with warnings.catch_warnings():
            warnings.simplefilter('always', RuntimeWarning)
            warnings.warn(f'Unknown distribution "{compared_best}" for "{part}" --> skipped.', RuntimeWarning)
        return None

    wb_results = fit_function(part)

    return wb_results, wb_data_fit_all, wb_best_distribution_name


#-----------------------------------------------------------------------------------------------------------------------
# Perform a Weibull Analysis to one specific part by using different Weibull distributions --> Plot for the HIT Dashboard
#-----------------------------------------------------------------------------------------------------------------------
def generate_graph(part, sort_by='CV', ci=0.95, return_sf=False):
    """
    Generate a Weibull (or reliability) plot for a single part, for use by the HIT dashboard,
    returning the image as an in-memory PNG buffer.

    Reuses the pre-computed analysis cache (`get_analysis_cache()`) when `sort_by == 'CV'` and the part is present,
    avoiding a full re-fit; otherwise recomputes the best-fit distribution on the fly.

    Parameters
    ----------
    part : str
        Part identifier to plot.
    sort_by : str, optional
        Distribution-selection criterion. If 'CV' (default) and cached results exist for `part`,
        the cache is used directly; any other value forces a fresh fit (internally sorted by 'BIC') and a fresh
        comparison via `compare_best_distribution`.
    ci : float, optional
        Confidence level for the plotted confidence bounds (default: 0.95).
    return_sf : bool, optional
        If True, generates a survival-function (reliability) plot instead of/in addition to the probability plot,
        per the underlying fitter function's behavior.

    Returns
    -------
    io.BytesIO or None
        An in-memory buffer positioned at the start, containing the PNG image of the generated plot.
        Returns None if the best-fit distribution name is not recognized (with a RuntimeWarning logged).

    Raises
    ------
    RuntimeError
        If `part` is falsy.
    """
    if not part:
        raise RuntimeError('Invalid request ("part" not specified)')

    buffer = io.BytesIO()   # Save plot in RAM
    
    analysis_cache = get_analysis_cache()

    ic_fallback = 'BIC'
    
    # As long as sort_by=='CV' the cache is valid to use even for the weibull_form
    using_cached_analysis = (sort_by == 'CV')
    
    if using_cached_analysis and analysis_cache and part in analysis_cache:
        cached = analysis_cache[part]
        compared_best = cached['best_model']
        data = cached['data']
        cv_used = cached['cv_used']
        logger.debug(f'generate_graph: cache available for "{part}".')
    else:
        # Only recompute if sort_by differs from default
        logger.debug(f'generate_graph: cache MISS for "{part}" (sort_by={sort_by})')

        sort_for_fit = sort_by if sort_by != 'CV' else 'BIC'

        wb_data_fit_all, _, data, fit_status = weibull_fit_best(part=part, sort_by=sort_for_fit)

        compared_best, cv_used = compare_best_distribution(df=wb_data_fit_all, sort_by=sort_by, part=part, data=data, ic_fallback=ic_fallback, delta=0.466, fit_status=fit_status)

    selection_used = 'CV' if cv_used else ic_fallback

    fitter_map = {'Weibull_2P': lambda p: weibull_2p(part=p, ci=ci, save_path=buffer, data=data, return_sf=return_sf, selection_method=selection_used),
                  'Weibull_3P': lambda p: weibull_3p(part=p, ci=ci, save_path=buffer, data=data, return_sf=return_sf, selection_method=selection_used),
                  'Weibull_Mixture': lambda p: weibull_mixture(part=p, ci=ci, save_path=buffer, data=data, return_sf=return_sf, selection_method=selection_used),
                  'Weibull_CR': lambda p: weibull_cr(part=p, ci=ci, save_path=buffer, data=data, return_sf=return_sf, selection_method=selection_used)}

    fit_function = fitter_map.get(compared_best)

    if fit_function is None:
        with warnings.catch_warnings():
            warnings.simplefilter('always', RuntimeWarning)
            warnings.warn(f'Unknown distribution "{compared_best}" for "{part}" --> skipped.', RuntimeWarning)
        return None

    wb_results = fit_function(part)

    buffer.seek(0)
    return buffer


#***********************************************************************************************************************
# Start the script local
#***********************************************************************************************************************
if __name__ == "__main__":
    from weibull_user_input import ask_threshold, ask_sort_by, ask_ci



    OUTPUT_DIR = r'C:\Users\lgroha\cernbox\Documents\Masterthesis\4_Python-Tool\CEM-IN_data_result-plots\Normal_data'
    parts_data, data_all = automated_weibull(save_path=OUTPUT_DIR)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.datetime.now(tz=ZoneInfo('Europe/Zurich')).strftime('%Y%m%d_%H%M%S')

    full_fit_table = pd.concat(data_all.values(), ignore_index=True)
    csv_path = os.path.join(OUTPUT_DIR, f'automated_weibull_fit_table_{timestamp}.csv')
    full_fit_table.to_csv(csv_path, index=False)

    logger.info(f'Automated Weibull fit table saved to "{csv_path}" ({len(full_fit_table)} rows).')

