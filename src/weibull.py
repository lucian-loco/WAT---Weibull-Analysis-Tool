#!/usr/bin/python3
#Using the predictr library is fine but improvements are done with the Reliability package: https://reliability.readthedocs.io/en/latest/index.html
from predictr import Analysis
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
#from reliability.Other_functions import distribution_explorer
#from reliability.Other_functions import make_right_censored_data
import io
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, FuncFormatter
from utils import get_logger
logger = get_logger(__name__)


# ToDo: Include "if res.optimizer is None:" to check whether the fit was successful
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


def plot_settings(fit, upper_quantile=0.999):
    ax = plt.gca()
    ax.set_xlabel('Time in days')
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
    ax.xaxis.set_minor_formatter(FuncFormatter(make_minor_label_formatter(decade_span)))

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

    return ax, fig, xmin, xmax_new


# The distribution is not calculated for the full range of xvals by default, this function extends the MLE fit
def plot_extension_mix_cr(fit, fit_data, upper_quantile=0.999):
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

#ToDo: Outsource access to data and only one time for every part --> include it in the automatic weibull and give data as input for the 4 weibull distributions
#-----------------------------------------------------------------------------------------------------------------------
# Function for Weibull 2P
#-----------------------------------------------------------------------------------------------------------------------
def weibull_2p(part, ci=0.95, save_path=None, data=None):
    if not part:
        raise RuntimeError('Invalid request ("part" not specified)')

    if data is None:
        data = get_failures_and_suspensions(part)

    failure_size = len(data['failures'])
    suspension_size = len(data['suspensions'])
    sample_size = failure_size + suspension_size

    # Prevent zeros in the right censored data
    if any(t == 0 for t in data['suspensions']):
        data['suspensions'] = [t for t in data['suspensions'] if t > 0]
        with warnings.catch_warnings():
            warnings.simplefilter('always', UserWarning)
            warnings.warn(f'The suspension data contained zeros as running_time. These assets have been ignored. Data need to be checked for {part}.', UserWarning)

    if len(data['suspensions']) == 0:
        data['suspensions'] = None

    if ci == 0.0:
        ci_type = 'none'
        ci = 0.95  # Standard value for CI in Fit_Weibull_Mixture --> CI=0.0 creates error | only affects the confidence bounds on the variables
    else:
        ci_type = 'reliability'

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

    plt.title(f'Weibull Probability Plot for {part} with \n (α={wb.alpha:.4f}, β={wb.beta:.4f}, CI={ci:.3f})')
    ax, fig,_ ,_ = plot_settings(wb)

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
def weibull_3p(part, ci=0.95, save_path=None, data=None):
    if not part:
        raise RuntimeError('Invalid request ("part" not specified)')

    if data is None:
        data = get_failures_and_suspensions(part)

    failure_size = len(data['failures'])
    suspension_size = len(data['suspensions'])
    sample_size = failure_size + suspension_size

    # Prevent zeros in the right censored data
    if any(t == 0 for t in data['suspensions']):
        data['suspensions'] = [t for t in data['suspensions'] if t > 0]
        with warnings.catch_warnings():
            warnings.simplefilter('always', UserWarning)
            warnings.warn(f'The suspension data contained zeros as running_time. These assets have been ignored. Data need to be checked for {part}.', UserWarning)

    if len(data['suspensions']) == 0:
        data['suspensions'] = None

    if ci == 0.0:
        ci_type = 'none'
        ci = 0.95   # Standard value for CI in Fit_Weibull_Mixture --> CI=0.0 creates error | only affects the confidence bounds on the variables
    else:
        ci_type = 'reliability'

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

    plt.title(f'Weibull Probability Plot for {part} with \n (α={wb.alpha:.4f}, β={wb.beta:.4f}, γ={wb.gamma:.4f}, CI={ci:.3f})')
    ax, fig, _, _ = plot_settings(wb)
    ax.set_xlabel(f'Time in days minus failure free time γ={wb.gamma:.4f}')

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
def weibull_mixture(part, ci=0.95, save_path=None, data=None):
    if not part:
        raise RuntimeError('Invalid request ("part" not specified)')

    if data is None:
        data = get_failures_and_suspensions(part)

    failure_size = len(data['failures'])
    suspension_size = len(data['suspensions'])
    sample_size = failure_size + suspension_size

    if failure_size < 4:
        raise ThresholdError('Not enough failures (more than 4) to perform Weibull Mixture in data for "{0}"'.format(part))
    elif failure_size < 20:
        with warnings.catch_warnings():
            warnings.simplefilter('always', UserWarning)
            warnings.warn(f'Less than 20 failures in total for "{part}"! It is highly recommended to use another model if there are less than 20 failures.', UserWarning)

    # Prevent zeros in the right censored data
    if any(t == 0 for t in data['suspensions']):
        data['suspensions'] = [t for t in data['suspensions'] if t > 0]
        with warnings.catch_warnings():
            warnings.simplefilter('always', UserWarning)
            warnings.warn(f'The suspension data contained zeros as running_time. These assets have been ignored. Data need to be checked for {part}.', UserWarning)

    if len(data['suspensions']) == 0:
        data['suspensions'] = None

    ci_mc = ci

    if ci == 0.0:
        ci = 0.95   # Standard value for CI in Fit_Weibull_Mixture --> CI=0.0 creates error | only affects the confidence bounds on the variables

    plt.figure()

    # see https://reliability.readthedocs.io/en/latest/API/Fitters.html for parameters description
    try:
        wb = Fit_Weibull_Mixture(failures=data['failures'], right_censored=data['suspensions'],
                                show_probability_plot=True, print_results=False,    # Results can be found in the returned variables as well
                                optimizer='best',                                  # Run with all Optimizers: “TNC”, “L-BFGS-B”, “nelder-mead”, and “powell”
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

    plt.title(f'Weibull Probability Plot for {part} with \n (α₁={wb.alpha_1:.4f}, β₁={wb.beta_1:.4f}, α₂={wb.alpha_2:.4f}, β₂={wb.beta_2:.4f}, \n proportion_factor={wb.proportion_1:.3f}, CI={ci:.3f})')
    ax, fig, xmin, xmax = plot_settings(wb)
    xmin_rel, xmax_rel = xmin * 0.8, xmax * 1.2

    if ci_mc != 0.0:
        # Calculation of the Confidence Interval:-----------------------------------------------------------------------
        xvals = np.logspace(np.log10(xmin_rel), np.log10(xmax_rel), 1000 + n_points)

        lower_mc, upper_mc = weibull_mixture_fisher_bounds(fit=wb, xvals=xvals, failures=data['failures'],
                                                     right_censored=data['suspensions'], CI=ci)

        if lower_mc is not None and upper_mc is not None:
            ax.fill_between(
                xvals,
                lower_mc,
                upper_mc,
                alpha=0.3,
                label=f'{int(ci * 100)}% numerical Fisher CI'
            )
        # --------------------------------------------------------------------------------------------------------------

        # Calculation of the Confidence Interval analytically:----------------------------------------------------------
        lower_analytic, upper_analytic, p_lower, p_upper = weibull_mixture_analytical_bounds(fit=wb, xvals=xvals, failures=data['failures'],
                                                                           right_censored=data['suspensions'], CI=ci)

        if lower_analytic is not None and upper_analytic is not None:
            ax.fill_between(
                xvals,
                lower_analytic,
                upper_analytic,
                alpha=0.3,
                label=f'{int(ci * 100)}% analytical Fisher CI'
            )
        # --------------------------------------------------------------------------------------------------------------
        # print(f'Starting with the bootstrapping...')
        # # Calculation of the Confidence Interval with bootstrap:--------------------------------------------------------
        # lower_bootstrap, upper_bootstrap = weibull_mixture_bootstrap_bounds(xvals=xvals, failures=data['failures'],
        #                                                                    right_censored=data['suspensions'], CI=ci)
        #
        # if lower_bootstrap is not None and upper_bootstrap is not None:
        #     ax.fill_between(
        #         xvals,
        #         lower_bootstrap,
        #         upper_bootstrap,
        #         alpha=0.3,
        #         facecolor='none',
        #         edgecolor='fuchsia',
        #         label=f'{int(ci * 100)}% Bootstrapping CI',
        #         hatch='oo'
        #     )
        # # --------------------------------------------------------------------------------------------------------------

    plt.legend(loc='upper left')

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
def weibull_cr(part, ci=0.95, save_path=None, data=None):
    if not part:
        raise RuntimeError('Invalid request ("part" not specified)')

    if data is None:
        data = get_failures_and_suspensions(part)

    failure_size = len(data['failures'])
    suspension_size = len(data['suspensions'])
    sample_size = failure_size + suspension_size

    if failure_size < 4:
        raise ThresholdError('Not enough failures (more than 4) to perform Weibull Competing Risks in data for "{0}"'.format(part))
    elif failure_size < 20:
        with warnings.catch_warnings():
            warnings.simplefilter('always', UserWarning)
            warnings.warn(f'Less than 20 failures in total for "{part}"! It is highly recommended to use another model if there are less than 20 failures.', UserWarning)

    # Prevent zeros in the right censored data
    if any(t == 0 for t in data['suspensions']):
        data['suspensions'] = [t for t in data['suspensions'] if t > 0]
        with warnings.catch_warnings():
            warnings.simplefilter('always', UserWarning)
            warnings.warn(f'The suspension data contained zeros as running_time. These assets have been ignored. Data need to be checked for {part}.', UserWarning)

    if len(data['suspensions']) == 0:
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

    plt.title(f'Weibull Probability Plot for {part} with \n (α₁={wb.alpha_1:.4f}, β₁={wb.beta_1:.4f}, α₂={wb.alpha_2:.4f}, β₂={wb.beta_2:.4f}, CI={ci:.3f})')
    ax, fig, xmin, xmax = plot_settings(wb)
    xmin_rel, xmax_rel = xmin * 0.8, xmax * 1.2

    if ci_mc != 0.0:
        # Calculation of the Confidence Interval:-----------------------------------------------------------------------
        xvals = np.logspace(np.log10(xmin_rel), np.log10(xmax_rel), 1000 + n_points)

        lower_mc, upper_mc = weibull_cr_fisher_bounds(fit=wb, xvals=xvals, failures=data['failures'],
                                                      right_censored=data['suspensions'], CI=ci)

        if lower_mc is not None and upper_mc is not None:
            ax.fill_between(
                xvals,
                lower_mc,
                upper_mc,
                alpha=0.3,
                label=f'{int(ci * 100)}% numerical Fisher CI'
            )
        # --------------------------------------------------------------------------------------------------------------

        # Calculation of the Confidence Interval analytically:----------------------------------------------------------
        lower_analytical, upper_analytical, _, _ = weibull_cr_analytical_bounds(fit=wb, xvals=xvals, failures=data['failures'],
                                                                                right_censored=data['suspensions'],CI=ci)

        if lower_analytical is not None and upper_analytical is not None:
            ax.fill_between(
                xvals,
                lower_analytical,
                upper_analytical,
                alpha=0.3,
                label=f'{int(ci * 100)}% analytical Fisher CI'
            )
        # --------------------------------------------------------------------------------------------------------------
        # print(f'Starting with the bootstrapping...')
        # # Calculation of the Confidence Interval with bootstrap:--------------------------------------------------------
        # lower_bootstrap, upper_bootstrap = weibull_cr_bootstrap_bounds(xvals=xvals, failures=data['failures'],
        #                                                                     right_censored=data['suspensions'], CI=ci)
        #
        # if lower_bootstrap is not None and upper_bootstrap is not None:
        #     ax.fill_between(
        #         xvals,
        #         lower_bootstrap,
        #         upper_bootstrap,
        #         alpha=0.3,
        #         facecolor='none',
        #         edgecolor='fuchsia',
        #         label=f'{int(ci * 100)}% Bootstrapping CI',
        #         hatch='oo'
        #     )
        # # --------------------------------------------------------------------------------------------------------------

    plt.legend(loc='upper left')

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
    if any(t == 0 for t in data['suspensions']):
        data['suspensions'] = [t for t in data['suspensions'] if t > 0]
        with warnings.catch_warnings():
            warnings.simplefilter('always', UserWarning)
            warnings.warn(f'The suspension data contained zeros as running_time. These assets have been ignored. Data need to be checked for {part}.', UserWarning)

    if not data['suspensions']:
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
            logger.warning(f'Less than 16 failures in total for "{part}"! It is highly recommended not to use the Weibull Mixture or Weibull CR model. '
                           f'Therefore, these models will not be used for the fitting.')
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

    wb_data_fit_all = wb.results
    wb_best_distribution_name = wb.best_distribution_name
    #print(wb_data_fit_all.to_string())

    return wb_data_fit_all, wb_best_distribution_name, data


#-----------------------------------------------------------------------------------------------------------------------
# Perform an automated Weibull Analysis to the HITDB Data by using different Weibull distributions
#-----------------------------------------------------------------------------------------------------------------------
def validate_sort_by(value_str: str, default: str = 'BIC'):
    valid = ['AICc', 'BIC']
    if value_str.strip() == "":
        return default, None
    if value_str in valid:
        return value_str, None
    return None, f"Invalid input, please enter one of {valid}."


def validate_ci(value_str: str, default: float = 0.95):
    if value_str.strip() == "":
        return default, None
    try:
        v = float(value_str)
        if 0 <= v < 1:
            return v, None
        return None, "Please enter a value strictly between 0 and 1 or 0 for no confidence interval."
    except ValueError:
        return None, "Invalid input, please enter a number (e.g. 0.95)."


def compare_best_distribution(df: pd.DataFrame, sort_by: str, part: str):
    """
    Determines the best distribution by comparing AICc and BIC winners.
    If both agree → take that result.
    If they disagree → fall back to the user's sort_by preference.
    """
    df = df.reset_index(drop=True)

    best_aicc = df.at[df['AICc'].idxmin(), 'Distribution']
    best_bic  = df.at[df['BIC'].idxmin(), 'Distribution']

    if best_aicc == best_bic:
        return best_aicc
    else:
        resolved = df.at[df[sort_by].idxmin(), 'Distribution']
        print(f'For {part}: ⚠ AICc → {best_aicc} vs BIC → {best_bic}: disagreement, using sort_by="{sort_by}" → {resolved}')
        return resolved

# ToDo: In case a Weibull Mixture is made of 1 failure by the first/second distribution and the rest of the failures by the other distribution --> neglect the Weibull Mixture
def automated_weibull():
    failure_threshold = ask_threshold("Failure threshold", default=4)
    distinct_threshold = ask_threshold("Distinct threshold", default=2)
    sort_by = ask_sort_by(default='BIC')
    ci = ask_ci(default=0.95)

    print(f"\n→ Starting search for parts with failure_threshold={failure_threshold}, distinct_threshold={distinct_threshold}, sort_by={sort_by} and CI={ci}\n")

    part_names_hit = get_parts(failure_threshold=failure_threshold, distinct_threshold=distinct_threshold)

    print(f"\n→ Starting analysis for these parts...")

    parts_data_fit_all = []
    parts_best_distribution_names = []

    for part in part_names_hit:
        wb_data_fit_all, wb_best_distribution_name, _ = weibull_fit_best(part=part, sort_by=sort_by)

        wb_data_fit_all['PART'] = part

        compared_best = compare_best_distribution(df=wb_data_fit_all, sort_by=sort_by, part=part)

        wb_best_distribution_row = pd.DataFrame({'PART': [part], 'BEST_DISTRIBUTION': [compared_best]})

        parts_data_fit_all.append(wb_data_fit_all)
        parts_best_distribution_names.append(wb_best_distribution_row)

    parts_data_fit_all = pd.concat(parts_data_fit_all, ignore_index=True)
    parts_data_fit_all = {name: group for name, group in parts_data_fit_all.groupby('PART')}

    parts_best_distribution_names = pd.concat(parts_best_distribution_names, ignore_index=True)

    fitter_map = {'Weibull_2P':         lambda p: weibull_2p(part=p, ci=ci, save_path=None),
                  'Weibull_3P':         lambda p: weibull_3p(part=p, ci=ci, save_path=None),
                  'Weibull_Mixture':    lambda p: weibull_mixture(part=p, ci=ci, save_path=None),
                  'Weibull_CR':         lambda p: weibull_cr(part=p, ci=ci, save_path=None)}

    parts_fit_results = {}

    print(f"\n→ Found the best distribution for these parts, now calculating the plots...")

    for _, row in parts_best_distribution_names.iterrows():
        part = row['PART']
        best_distribution = row['BEST_DISTRIBUTION']

        fit_function = fitter_map.get(best_distribution)

        if fit_function is None:
            with warnings.catch_warnings():
                warnings.simplefilter('always', RuntimeWarning)
                warnings.warn(f'Unknown distribution "{best_distribution}" for "{part}" --> skipped.', RuntimeWarning)
            continue

        parts_fit_results[part] = fit_function(part)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(f"\nThis are the results of the automated Weibull analysis:")
        for part, df in parts_fit_results.items():
            print(f"\n{'=' * 60}")
            print(f"  {part}")
            print(f"{'=' * 60}")
            print(df.to_string(index=False))

    return parts_fit_results, parts_data_fit_all


#-----------------------------------------------------------------------------------------------------------------------
# Perform a manual Weibull Analysis to one specific part by using different Weibull distributions
#-----------------------------------------------------------------------------------------------------------------------
def manual_weibull(part):
    if not part:
        raise RuntimeError('Invalid request ("part" not specified)')

    sort_by = ask_sort_by(default='BIC')
    ci = ask_ci(default=0.95)

    print(f"\n→ Starting Analysis for {part} with sort_by={sort_by} and CI={ci}\n")

    wb_data_fit_all, wb_best_distribution_name, data = weibull_fit_best(part=part, sort_by=sort_by)

    compared_best = compare_best_distribution(df=wb_data_fit_all, sort_by=sort_by, part=part)

    fitter_map = {'Weibull_2P': lambda p: weibull_2p(part=p, ci=ci, save_path=None, data=data),
                  'Weibull_3P': lambda p: weibull_3p(part=p, ci=ci, save_path=None, data=data),
                  'Weibull_Mixture': lambda p: weibull_mixture(part=p, ci=ci, save_path=None, data=data),
                  'Weibull_CR': lambda p: weibull_cr(part=p, ci=ci, save_path=None, data=data)}

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
def generate_graph_new(part, sort_by='BIC', ci=0.95):
    if not part:
        raise RuntimeError('Invalid request ("part" not specified)')

    buffer = io.BytesIO()   # Save plot in RAM

    wb_data_fit_all, wb_best_distribution_name, data = weibull_fit_best(part=part, sort_by=sort_by)

    compared_best = compare_best_distribution(df=wb_data_fit_all, sort_by=sort_by, part=part)

    fitter_map = {'Weibull_2P': lambda p: weibull_2p(part=p, ci=ci, save_path=buffer, data=data),
                  'Weibull_3P': lambda p: weibull_3p(part=p, ci=ci, save_path=buffer, data=data),
                  'Weibull_Mixture': lambda p: weibull_mixture(part=p, ci=ci, save_path=buffer, data=data),
                  'Weibull_CR': lambda p: weibull_cr(part=p, ci=ci, save_path=buffer, data=data)}

    fit_function = fitter_map.get(compared_best)

    if fit_function is None:
        with warnings.catch_warnings():
            warnings.simplefilter('always', RuntimeWarning)
            warnings.warn(f'Unknown distribution "{compared_best}" for "{part}" --> skipped.', RuntimeWarning)
        return None

    wb_results = fit_function(part)

    buffer.seek(0)
    return buffer


def generate_graph(part):
    if not part:
        raise RuntimeError('Invalid request ("part" not specified)')

    data = get_failures_and_suspensions(part)

    # Prepare the response
    buffer = io.BytesIO()   # buffer to keep the plot in RAM (instead of a file)

    # Weibull Analysis
    # see https://tvtoglu.github.io/predictr/classes/#default-arguments-and-values for more parameters
    x = Analysis(df=data['failures'], ds=data['suspensions'],
            show=False, save=True,
            fig_size=(9.5, 6),    # (8, 6) -> 800x600
            unit='days',
            plot_title='Weibull Probability Plot for {0}'.format(part),
            path=buffer)
    x.mle()

    # Send the plot image
    buffer.seek(0)
    return buffer



#***********************************************************************************************************************
# Start the script local
#***********************************************************************************************************************
if __name__ == "__main__":
    from weibull_user_input import ask_threshold, ask_sort_by, ask_ci
    from Synthetic_Data import load_datasets_from_csv

    data_csv = load_datasets_from_csv(csv_path=r'C:\Users\lgroha\cernbox\Documents\Masterthesis\4_Python-Tool\Synthetic-Data\synth_Mix_a1000_b0_5_n1000_cr0_1_a215000_b20_5_p0_2.csv', seed=2)

    # print(f'Der Datensatz sieht wie folgt aus: {data_csv[0]}')

    weibull_mixture(part='Synthetic data', data=data_csv[0])



    # manual_weibull(part='HCCVSWB')
    # weibull_cr(part='HCCVSEA')
    # weibull_mixture(part='HCCVSWB')
#     # data, _, name = manual_weibull('HCCFISA')
#     parts_data, data_all = automated_weibull()
#
#     with pd.option_context('display.max_rows', None, 'display.max_columns', None):
#         # print(f'\nResult data of the {name} fit:\n ', data)
#         print(f'\nFull result data of every part for every distribution:')
#         for part, df in data_all.items():
#             print(f"\n{'=' * 60}")
#             print(f"  {part}")
#             print(f"{'=' * 60}")
#             print(df.to_string(index=False))





