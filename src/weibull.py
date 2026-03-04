#!/usr/bin/python3
#Using the predictr library is fine but improvements are done with the Reliability package: https://reliability.readthedocs.io/en/latest/index.html
import data_weibull
from predictr import Analysis
from data_weibull import get_data
from data_weibull import get_parts
from data_weibull import get_all_data
from data_weibull import get_csv_data
from reliability.Utils import colorprint
from reliability.Fitters import Fit_Everything
from reliability.Fitters import Fit_Weibull_2P
from reliability.Fitters import Fit_Weibull_3P
from reliability.Fitters import Fit_Weibull_CR
from reliability.Fitters import Fit_Weibull_Mixture
from weibull_ci import weibull_cr_fisher_bounds
from weibull_ci import weibull_mixture_fisher_bounds
#from reliability.Other_functions import distribution_explorer
#from reliability.Other_functions import make_right_censored_data
import io
import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



#-----------------------------------------------------------------------------------------------------------------------
# Plot settings
#-----------------------------------------------------------------------------------------------------------------------
def plot_settings():
    ax = plt.gca()
    ax.set_xlabel('Time in days')
    ax.set_ylabel('Failure probability')
    plt.legend(loc='upper left')
    ax.set_ylim(0.001, 0.999)
    labels = ax.get_xticklabels()
    for i, label in enumerate(labels):  # Prevents overlapping of x-axis' ticks
        label.set_visible(i < 3 or (i - 3) % 2 == 0)

    xmin, xmax = ax.get_xlim()
    ax.set_xlim(xmin * 0.8, xmax * 1.2)

    fig = plt.gcf()
    width = 9.5/3
    height = 6/3
    fig.set_size_inches(width, height)

    return ax, fig, xmin, xmax


#-----------------------------------------------------------------------------------------------------------------------
# Function for Weibull 2P
#-----------------------------------------------------------------------------------------------------------------------
def weibull_2p(part, ci=0.95, save_path=None):
    if not part:
        raise RuntimeError('Invalid request ("part" not specified)')

    data = get_data(part)

    failure_size = len(data['failures'])
    suspension_size = len(data['suspensions'])
    sample_size = failure_size + suspension_size

    # Prevent zeros in the right censored data
    if any(t == 0 for t in data['suspensions']):
        data['suspensions'] = [t for t in data['suspensions'] if t > 0]
        with warnings.catch_warnings():
            warnings.simplefilter('always', UserWarning)
            warnings.warn(f'The suspension data contained zeros as running_time. These assets have been ignored. Data need to be checked for {part}.', UserWarning)

    if not data['suspensions']:
        data['suspensions'] = None

    plt.figure(figsize=(9.5, 6), dpi=100)
# ToDo: Edit the CI_type and CI in the way that if CI=0 then CI_type='None'
    # see https://reliability.readthedocs.io/en/latest/API/Fitters.html for parameters description
    wb = Fit_Weibull_2P(failures=data['failures'], right_censored=data['suspensions'],
                        show_probability_plot=True, print_results=False,    # Results can be found in the returned variables as well
                        method='MLE', optimizer='best',                     # Run with all Optimizers: “TNC”, “L-BFGS-B”, “nelder-mead”, and “powell”
                        CI_type='reliability', CI=ci,
                        label=f'Weibull 2 Parameter fit | MLE \n (n = {sample_size} (f: {failure_size} | s: {suspension_size})')

    plt.title(f'Weibull Probability Plot for {part} with \n (α={wb.alpha:.4f}, β={wb.beta:.4f}, CI={ci:.3f})')
    ax, fig,_ ,_ = plot_settings()

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
def weibull_3p(part, ci=0.95, save_path=None):
    if not part:
        raise RuntimeError('Invalid request ("part" not specified)')

    data = get_data(part)

    failure_size = len(data['failures'])
    suspension_size = len(data['suspensions'])
    sample_size = failure_size + suspension_size

    # Prevent zeros in the right censored data
    if any(t == 0 for t in data['suspensions']):
        data['suspensions'] = [t for t in data['suspensions'] if t > 0]
        with warnings.catch_warnings():
            warnings.simplefilter('always', UserWarning)
            warnings.warn(f'The suspension data contained zeros as running_time. These assets have been ignored. Data need to be checked for {part}.', UserWarning)

    if not data['suspensions']:
        data['suspensions'] = None

    plt.figure()
# ToDo: Edit the CI_type and CI in the way that if CI=0 then CI_type='None'
    # see https://reliability.readthedocs.io/en/latest/API/Fitters.html for parameters description
    wb = Fit_Weibull_3P(failures=data['failures'], right_censored=data['suspensions'],
                        show_probability_plot=True, print_results=False,    # Results can be found in the returned variables as well
                        method='MLE', optimizer='best',                     # Run with all Optimizers: “TNC”, “L-BFGS-B”, “nelder-mead”, and “powell”
                        CI_type='reliability', CI=ci,
                        label=f'Weibull 3 Parameter fit | MLE \n (n = {sample_size} (f: {failure_size} | s: {suspension_size})')

    plt.title(f'Weibull Probability Plot for {part} with \n (α={wb.alpha:.4f}, β={wb.beta:.4f}, γ={wb.gamma:.4f}, CI={ci:.3f})')
    ax, fig, _, _ = plot_settings()
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
def weibull_mixture(part, ci=0.95, save_path=None):
    if not part:
        raise RuntimeError('Invalid request ("part" not specified)')

    data = get_data(part)

    failure_size = len(data['failures'])
    suspension_size = len(data['suspensions'])
    sample_size = failure_size + suspension_size

    if failure_size < 4:
        raise RuntimeError('Not enough failures (more than 4) to perform Weibull CR in data for "{0}"'.format(part))
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

    if not data['suspensions']:
        data['suspensions'] = None

    plt.figure()
# ToDo: Edit the CI_type and CI in the way that if CI=0 then CI_type='None'
    # see https://reliability.readthedocs.io/en/latest/API/Fitters.html for parameters description
    wb = Fit_Weibull_Mixture(failures=data['failures'], right_censored=data['suspensions'],
                        show_probability_plot=True, print_results=False,    # Results can be found in the returned variables as well
                        optimizer='best',                                  # Run with all Optimizers: “TNC”, “L-BFGS-B”, “nelder-mead”, and “powell”
                        CI=ci,
                        label=f'Weibull Mixture fit | MLE \n (n = {sample_size} (f: {failure_size} | s: {suspension_size})')

    plt.title(f'Weibull Probability Plot for {part} with \n (α_1={wb.alpha_1:.4f}, β_1={wb.beta_1:.4f}, α_2={wb.alpha_2:.4f}, β_2={wb.beta_2:.4f}, \n proportion_factor={wb.proportion_1:.3f}, CI={ci:.3f})')
    ax, fig, xmin, xmax = plot_settings()
    xmin_rel, xmax_rel = xmin * 0.8, xmax * 1.2

    # Calculation of the Confidence Interval:---------------------------------------------------------------------------
    xvals = np.logspace(np.log10(xmin_rel), np.log10(xmax_rel), 800)

    lower, upper = weibull_mixture_fisher_bounds(fit=wb, xvals=xvals, failures=data['failures'], right_censored=data['suspensions'], CI=ci)

    if lower is not None and upper is not None:
        ax.fill_between(
            xvals,
            lower,
            upper,
            alpha=0.3,
            # label=f"{int(ci * 100)}% Fisher CI"
        )
    #-------------------------------------------------------------------------------------------------------------------

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
def weibull_cr(part, ci=0.95, save_path=None):
    if not part:
        raise RuntimeError('Invalid request ("part" not specified)')

    data = get_data(part)

    failure_size = len(data['failures'])
    suspension_size = len(data['suspensions'])
    sample_size = failure_size + suspension_size

    if failure_size < 4:
        raise RuntimeError('Not enough failures (more than 4) to perform Weibull CR in data for "{0}"'.format(part))
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

    if not data['suspensions']:
        data['suspensions'] = None

    plt.figure()

    # see https://reliability.readthedocs.io/en/latest/API/Fitters.html for parameters description
    wb = Fit_Weibull_CR(failures=data['failures'], right_censored=data['suspensions'],
                        show_probability_plot=True, print_results=False,    # Results can be found in the returned variables as well
                        optimizer='best',                                  # Run with all Optimizers: “TNC”, “L-BFGS-B”, “nelder-mead”, and “powell”
                        CI=ci,
                        label=f'Weibull Mixture fit | MLE \n (n = {sample_size} (f: {failure_size} | s: {suspension_size})')

    plt.title(f'Weibull Probability Plot for {part} with \n (α_1={wb.alpha_1:.4f}, β_1={wb.beta_1:.4f}, α_2={wb.alpha_2:.4f}, β_2={wb.beta_2:.4f}, CI={ci:.3f})')
    ax, fig, xmin, xmax = plot_settings()
    xmin_rel, xmax_rel = xmin * 0.8, xmax * 1.2

    # Calculation of the Confidence Interval:---------------------------------------------------------------------------
    xvals = np.logspace(np.log10(xmin_rel), np.log10(xmax_rel), 800)

    lower, upper = weibull_cr_fisher_bounds(fit=wb, xvals=xvals, failures=data['failures'], right_censored=data['suspensions'], CI=ci)

    if lower is not None and upper is not None:
        ax.fill_between(
            xvals,
            lower,
            upper,
            alpha=0.3,
            # label=f"{int(ci * 100)}% Fisher CI"
        )
    # -------------------------------------------------------------------------------------------------------------------

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
def weibull_fit_best(part, sort_by='BIC'):
    if not part:
        raise RuntimeError('Invalid request ("part" not specified)')

    warnings.filterwarnings("ignore", category=FutureWarning, message="The behavior of DataFrame concatenation with empty or all-NA entries is deprecated")

    data = get_data(part)

    failure_size = len(data['failures'])
    distinct_failure_count = len(set(data['failures']))
    # suspension_size = len(data['suspensions'])
    # sample_size = failure_size + suspension_size

    if failure_size < 2:
        raise RuntimeError('Not enough failures (more than 2) to perform Weibull in data for "{0}"'.format(part))

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

    if distinct_failure_count < 3:
        colorprint(f'Less than 3 distinct failure times for "{part}"! It is not possible to fit the Weibull_3P, Weibull_Mixture and Weibull_CR. '
                'Therefore, these models will not be used for the fitting.', text_color='red')

        exclude = ['Weibull_3P', 'Weibull_CR', 'Weibull_Mixture', 'Weibull_DS', 'Normal_2P', 'Gamma_2P', 'Loglogistic_2P',
                   'Gamma_3P', 'Lognormal_2P', 'Lognormal_3P', 'Loglogistic_3P',
                   'Gumbel_2P', 'Exponential_2P', 'Exponential_1P', 'Beta_2P']
    elif failure_size < 20:
        colorprint(f'Less than 20 failures in total for "{part}"! It is highly recommended not to use the Weibull Mixture or Weibull CR model. '
                'Therefore, these models will not be used for the fitting.', text_color='red')

        exclude = ['Weibull_CR', 'Weibull_Mixture', 'Weibull_DS', 'Normal_2P', 'Gamma_2P', 'Loglogistic_2P',
                   'Gamma_3P', 'Lognormal_2P', 'Lognormal_3P', 'Loglogistic_3P',
                   'Gumbel_2P', 'Exponential_2P', 'Exponential_1P', 'Beta_2P']
    else:
        exclude = ['Weibull_DS', 'Normal_2P', 'Gamma_2P', 'Loglogistic_2P',
                   'Gamma_3P', 'Lognormal_2P', 'Lognormal_3P', 'Loglogistic_3P',
                   'Gumbel_2P', 'Exponential_2P', 'Exponential_1P', 'Beta_2P']

    wb = Fit_Everything(failures=data['failures'], right_censored=data['suspensions'],
                        sort_by=sort_by,
                        show_probability_plot=False,
                        show_histogram_plot=False, show_PP_plot=False,
                        show_best_distribution_probability_plot=False,
                        exclude=exclude,
                        print_results=False,
                        method='MLE', optimizer='Best')

    wb_data_fit_all = wb.results
    wb_best_distribution_name = wb.best_distribution_name
    #print(wb_data_fit_all.to_string())

    return wb_data_fit_all, wb_best_distribution_name


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
        if 0 < v < 1:
            return v, None
        return None, "Please enter a value strictly between 0 and 1."
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
        wb_data_fit_all, wb_best_distribution_name = weibull_fit_best(part=part, sort_by=sort_by)

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

    fitter_map = {'Weibull_2P':         lambda p: weibull_2p(part=p, ci=ci, save_path=None),
                  'Weibull_3P':         lambda p: weibull_3p(part=p, ci=ci, save_path=None),
                  'Weibull_Mixture':    lambda p: weibull_mixture(part=p, ci=ci, save_path=None),
                  'Weibull_CR':         lambda p: weibull_cr(part=p, ci=ci, save_path=None)}

    wb_data_fit_all, wb_best_distribution_name = weibull_fit_best(part=part, sort_by=sort_by)

    compared_best = compare_best_distribution(df=wb_data_fit_all, sort_by=sort_by, part=part)

    fit_function = fitter_map.get(compared_best)

    if fit_function is None:
        with warnings.catch_warnings():
            warnings.simplefilter('always', RuntimeWarning)
            warnings.warn(f'Unknown distribution "{compared_best}" for "{part}" --> skipped.', RuntimeWarning)
        return None

    wb_results = fit_function(part)

    return wb_results, wb_data_fit_all, wb_best_distribution_name


#-----------------------------------------------------------------------------------------------------------------------
# Perform a manual Weibull Analysis to one specific part by using different Weibull distributions --> Plot for the HITDB Dashboard
#-----------------------------------------------------------------------------------------------------------------------
def generate_graph_new(part, sort_by='BIC', ci=0.95):
    if not part:
        raise RuntimeError('Invalid request ("part" not specified)')

    buffer = io.BytesIO()   # Save plot in RAM

    fitter_map = {'Weibull_2P':         lambda p: weibull_2p(part=p, ci=ci, save_path=buffer),
                  'Weibull_3P':         lambda p: weibull_3p(part=p, ci=ci, save_path=buffer),
                  'Weibull_Mixture':    lambda p: weibull_mixture(part=p, ci=ci, save_path=buffer),
                  'Weibull_CR':         lambda p: weibull_cr(part=p, ci=ci, save_path=buffer)}

    wb_data_fit_all, wb_best_distribution_name = weibull_fit_best(part=part, sort_by=sort_by)

    compared_best = compare_best_distribution(df=wb_data_fit_all, sort_by=sort_by, part=part)

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

    data = get_data(part)

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
# Start the script
#***********************************************************************************************************************
if __name__ == "__main__":
    from weibull_user_input import ask_threshold, ask_sort_by, ask_ci

    weibull_2p('HCCTRV')
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





