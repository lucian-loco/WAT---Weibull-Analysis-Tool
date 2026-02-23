#!/usr/bin/python3
#Using the predictr library is fine but improvements are done with the Reliability package: https://reliability.readthedocs.io/en/latest/index.html
import data_weibull
from predictr import Analysis
from data_weibull import get_data
from data_weibull import get_parts
from data_weibull import get_all_data
from data_weibull import get_csv_data
from reliability.Fitters import Fit_Everything
from reliability.Fitters import Fit_Weibull_2P
from reliability.Fitters import Fit_Weibull_3P
from reliability.Fitters import Fit_Weibull_CR
from reliability.Fitters import Fit_Weibull_Mixture
#from reliability.Other_functions import distribution_explorer
#from reliability.Other_functions import make_right_censored_data
import io
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



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
            warnings.simplefilter('always', RuntimeWarning)
            warnings.warn('The suspension data contained zeros as running_time. These assets have been ignored. Data need to be checked.', RuntimeWarning)

    if not data['suspensions']:
        data['suspensions'] = None

    plt.figure()
# ToDo Edit the CI_type and CI in the way that if CI=0 then CI_type='None'
    # see https://reliability.readthedocs.io/en/latest/API/Fitters.html for parameters description
    wb = Fit_Weibull_2P(failures=data['failures'], right_censored=data['suspensions'],
                        show_probability_plot=True, print_results=False,    # Results can be found in the returned variables as well
                        method='MLE', optimizer='best',                     # Run with all Optimizers: “TNC”, “L-BFGS-B”, “nelder-mead”, and “powell”
                        CI_type='reliability', CI=ci,
                        label=f'Weibull 2 Parameter fit | MLE \n (n = {sample_size} (f: {failure_size} | s: {suspension_size})')

    plt.title(f'Weibull Probability Plot for {part} with \n (α={wb.alpha:.4f}, β={wb.beta:.4f})')
    plt.legend(loc='upper left')
    ax = plt.gca()
    ax.set_xlabel('Time in days')
    ax.set_ylabel('Failure probability')
    ax.set_ylim(0.001, 0.999)
    xmin, xmax = ax.get_xlim()
    ax.set_xlim(xmin * 0.8, xmax * 1.2)
    labels = ax.get_xticklabels()
    for i, label in enumerate(labels):  # Prevents overlapping of x-axis' ticks
        label.set_visible(i < 3 or (i - 3) % 2 == 0)
    fig = plt.gcf()
    fig.set_size_inches(9.5, 6)
    if save_path:
        plt.savefig(save_path, dpi=300)
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
            warnings.simplefilter('always', RuntimeWarning)
            warnings.warn('The suspension data contained zeros as running_time. These assets have been ignored. Data need to be checked.', RuntimeWarning)

    if not data['suspensions']:
        data['suspensions'] = None

    plt.figure()

    # see https://reliability.readthedocs.io/en/latest/API/Fitters.html for parameters description
    wb = Fit_Weibull_3P(failures=data['failures'], right_censored=data['suspensions'],
                        show_probability_plot=True, print_results=False,    # Results can be found in the returned variables as well
                        method='MLE', optimizer='best',                     # Run with all Optimizers: “TNC”, “L-BFGS-B”, “nelder-mead”, and “powell”
                        CI_type='reliability', CI=ci,
                        label=f'Weibull 3 Parameter fit | MLE \n (n = {sample_size} (f: {failure_size} | s: {suspension_size})')

    plt.title(rf'Weibull Probability Plot for {part} with {'\n'} ($\alpha$={wb.alpha:.4f}, $\beta$={wb.beta:.4f}, $\gamma$={wb.gamma:.4f})')
    plt.legend(loc='upper left')
    ax = plt.gca()
    ax.set_xlabel(rf'Time in days minus failure free time $\gamma$={wb.gamma:.4f}')
    ax.set_ylabel('Failure probability')
    ax.set_ylim(0.001, 0.999)
    xmin, xmax = ax.get_xlim()
    ax.set_xlim(xmin * 0.8, xmax * 1.2)
    labels = ax.get_xticklabels()
    for i, label in enumerate(labels):  # Prevents overlapping of x-axis' ticks
        label.set_visible(i < 3 or (i - 3) % 2 == 0)
    fig = plt.gcf()
    fig.set_size_inches(9.5, 6)
    if save_path:
        plt.savefig(save_path, dpi=300)
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
            warnings.simplefilter('always', RuntimeWarning)
            warnings.warn('Less than 20 failures in total! It is highly recommended to use another model if there are less than 20 failures.', RuntimeWarning)

    # Prevent zeros in the right censored data
    if any(t == 0 for t in data['suspensions']):
        data['suspensions'] = [t for t in data['suspensions'] if t > 0]
        with warnings.catch_warnings():
            warnings.simplefilter('always', RuntimeWarning)
            warnings.warn('The suspension data contained zeros as running_time. These assets have been ignored. Data need to be checked.', RuntimeWarning)

    if not data['suspensions']:
        data['suspensions'] = None

    plt.figure()

    # see https://reliability.readthedocs.io/en/latest/API/Fitters.html for parameters description
    wb = Fit_Weibull_Mixture(failures=data['failures'], right_censored=data['suspensions'],
                        show_probability_plot=True, print_results=False,    # Results can be found in the returned variables as well
                        optimizer='best',                                  # Run with all Optimizers: “TNC”, “L-BFGS-B”, “nelder-mead”, and “powell”
                        CI=ci,
                        label=f'Weibull Mixture fit | MLE \n (n = {sample_size} (f: {failure_size} | s: {suspension_size})')

# ToDo Confidence Interval for Weibull Mixture

    plt.title(rf'Weibull Probability Plot for {part} with {'\n'} ($\alpha_1$={wb.alpha_1:.4f}, $\beta_1$={wb.beta_1:.4f}, $\alpha_2$={wb.alpha_2:.4f}, $\beta_2$={wb.beta_2:.4f}, proportion_factor={wb.proportion_1:.3f})')
    plt.legend(loc='upper left')
    ax = plt.gca()
    ax.set_xlabel('Time in days')
    ax.set_ylabel('Failure probability')
    ax.set_ylim(0.001, 0.999)
    xmin, xmax = ax.get_xlim()
    ax.set_xlim(xmin * 0.8, xmax * 1.2)
    labels = ax.get_xticklabels()
    for i, label in enumerate(labels):  # Prevents overlapping of x-axis' ticks
        label.set_visible(i < 3 or (i - 3) % 2 == 0)
    fig = plt.gcf()
    fig.set_size_inches(9.5, 6)
    if save_path:
        plt.savefig(save_path, dpi=300)
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
            warnings.simplefilter('always', RuntimeWarning)
            warnings.warn('Less than 20 failures in total! It is highly recommended to use another model if there are less than 20 failures.', RuntimeWarning)

    # Prevent zeros in the right censored data
    if any(t == 0 for t in data['suspensions']):
        data['suspensions'] = [t for t in data['suspensions'] if t > 0]
        with warnings.catch_warnings():
            warnings.simplefilter('always', RuntimeWarning)
            warnings.warn('The suspension data contained zeros as running_time. These assets have been ignored. Data need to be checked.', RuntimeWarning)

    if not data['suspensions']:
        data['suspensions'] = None

    plt.figure()

    # see https://reliability.readthedocs.io/en/latest/API/Fitters.html for parameters description
    wb = Fit_Weibull_CR(failures=data['failures'], right_censored=data['suspensions'],
                        show_probability_plot=True, print_results=False,    # Results can be found in the returned variables as well
                        optimizer='best',                                  # Run with all Optimizers: “TNC”, “L-BFGS-B”, “nelder-mead”, and “powell”
                        CI=ci,
                        label=f'Weibull Mixture fit | MLE \n (n = {sample_size} (f: {failure_size} | s: {suspension_size})')

# ToDo Confidence Interval for Weibull Competing Risks

    plt.title(rf'Weibull Probability Plot for {part} with {'\n'} ($\alpha_1$={wb.alpha_1:.4f}, $\beta_1$={wb.beta_1:.4f}, $\alpha_2$={wb.alpha_2:.4f}, $\beta_2$={wb.beta_2:.4f})')
    plt.legend(loc='upper left')
    ax = plt.gca()
    ax.set_xlabel('Time in days')
    ax.set_ylabel('Failure probability')
    ax.set_ylim(0.001, 0.999)
    xmin, xmax = ax.get_xlim()
    ax.set_xlim(xmin * 0.8, xmax * 1.2)
    labels = ax.get_xticklabels()
    for i, label in enumerate(labels):  # Prevents overlapping of x-axis' ticks
        label.set_visible(i < 3 or (i - 3) % 2 == 0)
    fig = plt.gcf()
    fig.set_size_inches(9.5, 6)
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()
    # print(f'Goodness of fit values for the Weibull CR: \n {wb.goodness_of_fit} \n\n')

    return wb.results


#-----------------------------------------------------------------------------------------------------------------------
# Function for fitting the data to every available Weibull distribution --> AICc and BIC for every distribution in returned result object
#-----------------------------------------------------------------------------------------------------------------------
def weibull_fit_best(part, sort_by='AICc'):
    if not part:
        raise RuntimeError('Invalid request ("part" not specified)')

    warnings.filterwarnings("ignore", category=FutureWarning, message="The behavior of DataFrame concatenation with empty or all-NA entries is deprecated")

    data = get_data(part)

    failure_size = len(data['failures'])
    # suspension_size = len(data['suspensions'])
    # sample_size = failure_size + suspension_size

    if failure_size < 4:
        raise RuntimeError('Not enough failures (more than 4) to perform Weibull in data for "{0}"'.format(part))

    # Prevent zeros in the right censored data
    if any(t == 0 for t in data['suspensions']):
        data['suspensions'] = [t for t in data['suspensions'] if t > 0]
        with warnings.catch_warnings():
            warnings.simplefilter('always', RuntimeWarning)
            warnings.warn('The suspension data contained zeros as running_time. These assets have been ignored. Data need to be checked.', RuntimeWarning)

    if not data['suspensions']:
        data['suspensions'] = None

    # Weibull Analysis
    # see https://reliability.readthedocs.io/en/latest/API/Fitters.html for parameters description

    if failure_size < 20:
        with warnings.catch_warnings():
            warnings.simplefilter('always', RuntimeWarning)
            warnings.warn(f'Less than 20 failures in total for "{part}"! It is highly recommended not to use the Weibull Mixture or Weibull CR model. '
                'Therefore, these models will not be used for the fitting.', RuntimeWarning)

        exclude = ['Weibull_Mixture', 'Weibull_CR', 'Weibull_DS', 'Normal_2P', 'Gamma_2P', 'Loglogistic_2P',
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
def ask_threshold(name: str, default: int):
    while True:
        user_input = input(f"Enter {name} and press enter (Default value: {default}): ").strip()
        if user_input == "":
            return default
        try:
            value = int(user_input)
            if value > 0:
                return value
            else:
                print("  → Please enter a positive number.")
        except ValueError:
            print("  → Invalid input, please enter an integer..")


def ask_sort_by(default: str = 'AICc'):
    valid_options = ['AICc', 'BIC']
    while True:
        user_input = input(f"Enter sort method ({valid_options}) and press enter (Default value: {default}): ").strip()
        if user_input == "":
            return default
        if user_input in valid_options:
            return user_input
        else:
            print(f"  → Invalid input, please enter one of {valid_options}.")


def ask_ci(default: float = 0.95):
    while True:
        user_input = input(f"Enter confidence interval (0-1) and press enter (Default value: {default}): ").strip()
        if user_input == "":
            return default
        try:
            value = float(user_input)
            if 0 < value < 1:
                return value
            else:
                print("  → Please enter a value strictly between 0 and 1.")
        except ValueError:
            print("  → Invalid input, please enter a number (e.g. 0.95).")


def automated_weibull():
    failure_threshold = ask_threshold("Failure threshold", default=4)
    distinct_threshold = ask_threshold("Distinct threshold", default=2)
    sort_by = ask_sort_by(default='AICc')
    ci = ask_ci(default=0.95)

    print(f"\n→ Starting search for parts with failure_threshold={failure_threshold}, distinct_threshold={distinct_threshold}, sort_by={sort_by} and CI={ci}\n")

    part_names_hit = get_parts(failure_threshold=failure_threshold, distinct_threshold=distinct_threshold)

    print(f"\n→ Starting analysis for these parts...")

    parts_data_fit_all = []
    parts_best_distribution_names = []

    for part in part_names_hit:
        wb_data_fit_all, wb_best_distribution_name = weibull_fit_best(part=part, sort_by=sort_by)

        wb_data_fit_all['PART'] = part

        wb_best_distribution_row = pd.DataFrame({'PART': [part], 'BEST_DISTRIBUTION': [wb_best_distribution_name]})

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

    print(f"\nThis are the results of the automated Weibull analysis: ", parts_fit_results)

    return parts_fit_results


#-----------------------------------------------------------------------------------------------------------------------
# Perform a manual Weibull Analysis to one specific part by using different Weibull distributions
#-----------------------------------------------------------------------------------------------------------------------
def manual_weibull(part):
    if not part:
        raise RuntimeError('Invalid request ("part" not specified)')

    sort_by = ask_sort_by(default='AICc')
    ci = ask_ci(default=0.95)

    fitter_map = {'Weibull_2P':         lambda p: weibull_2p(part=p, ci=ci, save_path=None),
                  'Weibull_3P':         lambda p: weibull_3p(part=p, ci=ci, save_path=None),
                  'Weibull_Mixture':    lambda p: weibull_mixture(part=p, ci=ci, save_path=None),
                  'Weibull_CR':         lambda p: weibull_cr(part=p, ci=ci, save_path=None)}

    wb_data_fit_all, wb_best_distribution_name = weibull_fit_best(part=part, sort_by=sort_by)

    fit_function = fitter_map.get(wb_best_distribution_name)

    if fit_function is None:
        with warnings.catch_warnings():
            warnings.simplefilter('always', RuntimeWarning)
            warnings.warn(f'Unknown distribution "{wb_best_distribution_name}" for "{part}" --> skipped.', RuntimeWarning)
        return None

    wb_results = fit_function(part)

    return wb_results, wb_data_fit_all


#-----------------------------------------------------------------------------------------------------------------------
# Perform a manual Weibull Analysis to one specific part by using different Weibull distributions --> Plot for the HITDB Dashboard
#-----------------------------------------------------------------------------------------------------------------------
def generate_graph(part):
    if not part:
        raise RuntimeError('Invalid request ("part" not specified)')

    buffer = io.BytesIO()   # Save plot in RAM

    fitter_map = {'Weibull_2P':         lambda p: weibull_2p(part=p, ci=0.95, save_path=buffer),
                  'Weibull_3P':         lambda p: weibull_3p(part=p, ci=0.95, save_path=buffer),
                  'Weibull_Mixture':    lambda p: weibull_mixture(part=p, ci=0.95, save_path=buffer),
                  'Weibull_CR':         lambda p: weibull_cr(part=p, ci=0.95, save_path=buffer)}

    wb_data_fit_all, wb_best_distribution_name = weibull_fit_best(part=part, sort_by='AICc')

    fit_function = fitter_map.get(wb_best_distribution_name)

    if fit_function is None:
        with warnings.catch_warnings():
            warnings.simplefilter('always', RuntimeWarning)
            warnings.warn(f'Unknown distribution "{wb_best_distribution_name}" for "{part}" --> skipped.', RuntimeWarning)
        return None

    wb_results = fit_function(part)

    buffer.seek(0)
    return buffer


automated_weibull()




#ToDo implementation of different libraries

#ToDo implement own function for plotting the data out of the several distribution functions

weibull_distributions = [weibull_2p, weibull_3p, weibull_mixture, weibull_cr, weibull_fit_best]

# Definition of the required part
part_name = 'HCCVSEA'

# Create the Weibull plot with 2 different ways
#weibull_2p(part_name)
#generate_graph_local(part_name)
#weibull_fit_best(part_name)
#weibull_mixture(part=part_name)

# for weibull_distribution in weibull_distributions:
#     weibull_distribution(part=part_name)




#base_dir = r"C:\Users\lgroha\cernbox\Documents\Masterthesis\3_Data-Preparation\Weibull_Plots"

#part_groups = [data_weibull.parts_to_be_edited_or_changed, data_weibull.parts_with_sus_dates, data_weibull.parts_failed_selection]

#for group_name, parts in zip(["parts_to_be_edited_or_changed", "parts_with_sus_dates", "parts_failed_selection"], part_groups):
#    group_dir = os.path.join(base_dir, group_name)
#    os.makedirs(group_dir, exist_ok=True)

#    for part in parts:
#        save_path = os.path.join(group_dir, f"weibull_plot_{part}.png")
#        weibull_2p(part, save_path=save_path)
