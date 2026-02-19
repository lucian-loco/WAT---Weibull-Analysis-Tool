#!/usr/bin/python3
#Using the predictr library is fine but improvements are done with the Reliability package: https://reliability.readthedocs.io/en/latest/index.html
from predictr import Analysis
from reliability.Fitters import Fit_Weibull_2P
from reliability.Fitters import Fit_Weibull_3P
from reliability.Fitters import Fit_Weibull_Mixture
from reliability.Fitters import Fit_Weibull_CR
#from reliability.Other_functions import distribution_explorer
#from reliability.Other_functions import make_right_censored_data
#from reliability.Probability_plotting import Weibull_probability_plot
from reliability.Fitters import Fit_Everything
import matplotlib.pyplot as plt
import io
import numpy as np
import os
import warnings
import data_weibull
from data_weibull import get_data
from data_weibull import get_all_data
from data_weibull import get_parts
from data_weibull import get_csv_data


#old library and current state in the HIT Dashboard
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


#-----------------------------------------------------------------------------------------------------------------------
# Function for Weibull 2P
#-----------------------------------------------------------------------------------------------------------------------
def weibull_2p(part, save_path=None):
    if not part:
        raise RuntimeError('Invalid request ("part" not specified)')

    data = get_data(part)

    failure_size = len(data['failures'])
    suspension_size = len(data['suspensions'])
    sample_size = failure_size + suspension_size

    # Prevent zeros in the right censored data
    if any(t == 0 for t in data['suspensions']):
        data['suspensions'] = [t for t in data['suspensions'] if t > 0]
        warnings.warn('The suspension data contained zeros as running_time. These assets have been ignored. Data need to be checked.', RuntimeWarning)

    if not data['suspensions']:
        data['suspensions'] = None

    plt.figure()

    # see https://reliability.readthedocs.io/en/latest/API/Fitters.html for parameters description
    wb = Fit_Weibull_2P(failures=data['failures'], right_censored=data['suspensions'],
                        show_probability_plot=True, print_results=False,    # Results can be found in the returned variables as well
                        method='MLE', optimizer='best',                     # Run with all Optimizers: “TNC”, “L-BFGS-B”, “nelder-mead”, and “powell”
                        CI_type='reliability', CI=0.95,
                        label=f'Weibull 2 Parameter fit | MLE \n (n = {sample_size} (f: {failure_size} | s: {suspension_size})'
                        )

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
    plt.show()
    print(f'Goodness of fit values for the Weibull 2P: \n {wb.goodness_of_fit} \n\n')


#-----------------------------------------------------------------------------------------------------------------------
# Function for Weibull 3P
#-----------------------------------------------------------------------------------------------------------------------
def weibull_3p(part, save_path=None):
    if not part:
        raise RuntimeError('Invalid request ("part" not specified)')

    data = get_data(part)

    failure_size = len(data['failures'])
    suspension_size = len(data['suspensions'])
    sample_size = failure_size + suspension_size

    # Prevent zeros in the right censored data
    if any(t == 0 for t in data['suspensions']):
        data['suspensions'] = [t for t in data['suspensions'] if t > 0]
        warnings.warn('The suspension data contained zeros as running_time. These assets have been ignored. Data need to be checked.', RuntimeWarning)

    if not data['suspensions']:
        data['suspensions'] = None

    plt.figure()

    # see https://reliability.readthedocs.io/en/latest/API/Fitters.html for parameters description
    wb = Fit_Weibull_3P(failures=data['failures'], right_censored=data['suspensions'],
                        show_probability_plot=True, print_results=False,    # Results can be found in the returned variables as well
                        method='MLE', optimizer='best',                     # Run with all Optimizers: “TNC”, “L-BFGS-B”, “nelder-mead”, and “powell”
                        CI_type='reliability', CI=0.95,
                        label=f'Weibull 3 Parameter fit | MLE \n (n = {sample_size} (f: {failure_size} | s: {suspension_size})'
                        )

    plt.title(rf'Weibull Probability Plot for {part} with {'\n'} ($\alpha$={wb.alpha:.4f}, $\beta$={wb.beta:.4f}, $\gamma$={wb.gamma:.4f})')
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
    plt.show()
    print(f'Goodness of fit values for the Weibull 3P: \n {wb.goodness_of_fit} \n\n')


#-----------------------------------------------------------------------------------------------------------------------
# Function for Weibull Mixture with 2 distributions
#-----------------------------------------------------------------------------------------------------------------------
def weibull_mixture(part, save_path=None):
    if not part:
        raise RuntimeError('Invalid request ("part" not specified)')

    data = get_data(part)

    failure_size = len(data['failures'])
    suspension_size = len(data['suspensions'])
    sample_size = failure_size + suspension_size

    if failure_size < 4:
        raise RuntimeError('Not enough failures (more than 4) to perform Weibull CR in data for "{0}"'.format(part))
    elif failure_size < 20:
        warnings.warn('Less than 20 failures in total! It is highly recommended to use another model if there are less than 20 failures.', RuntimeWarning)

    # Prevent zeros in the right censored data
    if any(t == 0 for t in data['suspensions']):
        data['suspensions'] = [t for t in data['suspensions'] if t > 0]
        warnings.warn('The suspension data contained zeros as running_time. These assets have been ignored. Data need to be checked.', RuntimeWarning)

    if not data['suspensions']:
        data['suspensions'] = None

    plt.figure()

    # see https://reliability.readthedocs.io/en/latest/API/Fitters.html for parameters description
    wb = Fit_Weibull_Mixture(failures=data['failures'], right_censored=data['suspensions'],
                        show_probability_plot=True, print_results=False,    # Results can be found in the returned variables as well
                        optimizer='best',                                  # Run with all Optimizers: “TNC”, “L-BFGS-B”, “nelder-mead”, and “powell”
                        CI=0.95,
                        label=f'Weibull Mixture fit | MLE \n (n = {sample_size} (f: {failure_size} | s: {suspension_size})'
                        )

    # # Bootstrap parameter
    # b = 500
    #
    # cdf_boot = []
    #
    # for b in range(b):
    #     simulated = wb.distribution.random_samples(sample_size)
    #
    #     try:
    #         fit_b = Fit_Weibull_Mixture(failures=simulated, show_probability_plot=False)
    #
    #         cdf_boot.append(
    #             fit_b.distribution.CDF(t)
    #         )
    #
    #     except:
    #         continue  # falls Fit numerisch nicht konvergiert
    #
    # cdf_boot = np.array(cdf_boot)
    #
    # # 4. Quantile berechnen
    # lower = np.percentile(cdf_boot, 2.5, axis=0)
    # upper = np.percentile(cdf_boot, 97.5, axis=0)

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
    plt.show()
    print(f'Goodness of fit values for the Weibull Mixture: \n {wb.goodness_of_fit} \n\n')


#-----------------------------------------------------------------------------------------------------------------------
# Function for Weibull Competing Risks with 2 distributions
#-----------------------------------------------------------------------------------------------------------------------
def weibull_cr(part, save_path=None):
    if not part:
        raise RuntimeError('Invalid request ("part" not specified)')

    data = get_data(part)

    failure_size = len(data['failures'])
    suspension_size = len(data['suspensions'])
    sample_size = failure_size + suspension_size

    if failure_size < 4:
        raise RuntimeError('Not enough failures (more than 4) to perform Weibull CR in data for "{0}"'.format(part))
    elif failure_size < 20:
        warnings.warn('Less than 20 failures in total! It is highly recommended to use another model if there are less than 20 failures.', RuntimeWarning)

    # Prevent zeros in the right censored data
    if any(t == 0 for t in data['suspensions']):
        data['suspensions'] = [t for t in data['suspensions'] if t > 0]
        warnings.warn('The suspension data contained zeros as running_time. These assets have been ignored. Data need to be checked.', RuntimeWarning)

    if not data['suspensions']:
        data['suspensions'] = None

    plt.figure()

    # see https://reliability.readthedocs.io/en/latest/API/Fitters.html for parameters description
    wb = Fit_Weibull_CR(failures=data['failures'], right_censored=data['suspensions'],
                        show_probability_plot=True, print_results=False,    # Results can be found in the returned variables as well
                        optimizer='best',                                  # Run with all Optimizers: “TNC”, “L-BFGS-B”, “nelder-mead”, and “powell”
                        CI=0.95,
                        label=f'Weibull Mixture fit | MLE \n (n = {sample_size} (f: {failure_size} | s: {suspension_size})'
                        )

    # # Bootstrap parameter
    # b = 500
    #
    # cdf_boot = []
    #
    # for b in range(b):
    #     simulated = wb.distribution.random_samples(sample_size)
    #
    #     try:
    #         fit_b = Fit_Weibull_Mixture(failures=simulated, show_probability_plot=False)
    #
    #         cdf_boot.append(
    #             fit_b.distribution.CDF(t)
    #         )
    #
    #     except:
    #         continue  # falls Fit numerisch nicht konvergiert
    #
    # cdf_boot = np.array(cdf_boot)
    #
    # # 4. Quantile berechnen
    # lower = np.percentile(cdf_boot, 2.5, axis=0)
    # upper = np.percentile(cdf_boot, 97.5, axis=0)

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
    plt.show()
    print(f'Goodness of fit values for the Weibull CR: \n {wb.goodness_of_fit} \n\n')


#-----------------------------------------------------------------------------------------------------------------------
# Function for fitting the data to every available Weibull distribution --> AICc and BIC as result
#-----------------------------------------------------------------------------------------------------------------------
def weibull_fit_best(part):
    if not part:
        raise RuntimeError('Invalid request ("part" not specified)')

    data = get_data(part)

    failure_size = len(data['failures'])
    # suspension_size = len(data['suspensions'])
    # sample_size = failure_size + suspension_size

    if failure_size < 4:
        raise RuntimeError('Not enough failures (more than 4) to perform Weibull in data for "{0}"'.format(part))
    elif failure_size < 20:
        warnings.warn('Less than 20 failures in total! It is highly recommended not to use the Weibull Mixture or Weibull CR model.', RuntimeWarning)

    # Prevent zeros in the right censored data
    if any(t == 0 for t in data['suspensions']):
        data['suspensions'] = [t for t in data['suspensions'] if t > 0]
        warnings.warn('The suspension data contained zeros as running_time. These assets have been ignored. Data need to be checked.', RuntimeWarning)

    if not data['suspensions']:
        data['suspensions'] = None

    # Weibull Analysis
    # see https://reliability.readthedocs.io/en/latest/API/Fitters.html for parameters description

    wb = Fit_Everything(failures=data['failures'], right_censored=data['suspensions'],
                        sort_by='AICc',
                        show_probability_plot=False,
                        show_histogram_plot=False, show_PP_plot=False, show_best_distribution_probability_plot=False,
                        exclude=['Weibull_DS', 'Normal_2P', 'Gamma_2P', 'Loglogistic_2P', 'Gamma_3P', 'Lognormal_2P', 'Lognormal_3P',
                                 'Loglogistic_3P', 'Gumbel_2P', 'Exponential_2P', 'Exponential_1P', 'Beta_2P'],
                        print_results=True,
                        method='MLE', optimizer='Best',
                        )




#ToDo implement of different libraries

#ToDo implement own function for plotting the data out of the several distribution functions

weibull_distributions = [weibull_2p, weibull_3p, weibull_mixture, weibull_cr, weibull_fit_best]

# Definition of the required part
part_name = 'HCCVSEA'

# Create the Weibull plot with 2 different ways
#weibull_2p(part_name)
#generate_graph_local(part_name)
#weibull_fit_best(part_name)
#weibull_mixture(part=part_name)

for weibull_distribution in weibull_distributions:
    weibull_distribution(part=part_name)


#base_dir = r"C:\Users\lgroha\cernbox\Documents\Masterthesis\3_Data-Preparation\Weibull_Plots"

#part_groups = [data_weibull.parts_to_be_edited_or_changed, data_weibull.parts_with_sus_dates, data_weibull.parts_failed_selection]

#for group_name, parts in zip(["parts_to_be_edited_or_changed", "parts_with_sus_dates", "parts_failed_selection"], part_groups):
#    group_dir = os.path.join(base_dir, group_name)
#    os.makedirs(group_dir, exist_ok=True)

#    for part in parts:
#        save_path = os.path.join(group_dir, f"weibull_plot_{part}.png")
#        weibull_2p(part, save_path=save_path)
