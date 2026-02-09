#!/usr/bin/python3
#Using the predictr library is fine but improvements are done with the Reliability package: https://reliability.readthedocs.io/en/latest/index.html
from predictr import Analysis
from reliability.Fitters import Fit_Weibull_2P
#from reliability.Fitters import Fit_Weibull_3P
#from reliability.Fitters import Fit_Weibull_Mixture
#from reliability.Fitters import Fit_Weibull_CR
#from reliability.Other_functions import distribution_explorer
#from reliability.Probability_plotting import Weibull_probability_plot
from reliability.Fitters import Fit_Everything
import matplotlib.pyplot as plt
import io
import os
import warnings
import data_weibull
from data_weibull import get_data
#from data_weibull import get_all_data
#from data_weibull import get_parts
#from data_weibull import get_csv_data


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


#old library and current state in the HIT Dashboard but as local version
def generate_graph_local(part):
    if not part:
        raise RuntimeError('Invalid request ("part" not specified)')

    data = get_data(part)

    # Weibull Analysis
    # see https://tvtoglu.github.io/predictr/classes/#default-arguments-and-values for more parameters
    x = Analysis(df=data['failures'], ds=data['suspensions'],
            show=True, save=False,
            fig_size=(9.5, 6),    # (8, 6) -> 800x600
            unit='days',
            plot_title='Weibull Probability Plot for {0}'.format(part)
            )
    x.mle()


def weibull_2p(part, save_path=None):
    if not part:
        raise RuntimeError('Invalid request ("part" not specified)')

    data = get_data(part)

    failure_size = len(data['failures'])
    suspension_size = len(data['suspensions'])
    sample_size = failure_size + suspension_size

    # Prevent zeros in the right censored data --> HCCVAED
    if data['suspensions']:
        data['suspensions'] = [t for t in data['suspensions'] if t > 0]
        warnings.warn('The suspension data contained zeros as running_time. These assets have been ignored. Data need to be checked.', RuntimeWarning)

    if not data['suspensions']:
        data['suspensions'] = None

    plt.figure()
    # Weibull Analysis
    # see https://reliability.readthedocs.io/en/latest/API/Fitters.html for parameters description
    wb = Fit_Weibull_2P(failures=data['failures'], right_censored=data['suspensions'],
                        show_probability_plot=True, print_results=False, # Results can be found in the returned variables as well
                        method='MLE', optimizer='best', # Run with all Optimizers: “TNC”, “L-BFGS-B”, “nelder-mead”, and “powell”
                        CI_type='none', # In case of CI --> CI='float between 0 and 1'
                        label=f'Weibull 2 Parameter fit | MLE \n (n = {sample_size} (f: {failure_size} | s: {suspension_size})'
                        )

    plt.title(f'Weibull Probability Plot for {part} with \n (α={wb.alpha:.3f}, β={wb.beta:.3f})')
    ax = plt.gca()
    ax.set_xlabel('Time in days')
    ax.set_ylabel('Unreliability')
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


# Folgende Funktion kann auch nur zum "Finde die beste Verteilung" genutzt werden
# und dann wird diese Verteilung nochmal manuell erstellt (CI kann dann weg)
def weibull_fit_best(part):
    if not part:
        raise RuntimeError('Invalid request ("part" not specified)')

    data = get_data(part)

    failure_size = len(data['failures'])
    suspension_size = len(data['suspensions'])
    sample_size = failure_size + suspension_size

    # Weibull Analysis
    # see https://reliability.readthedocs.io/en/latest/API/Fitters.html for parameters description

    wb = Fit_Everything(failures=data['failures'], right_censored=data['suspensions'],
                        show_probability_plot=False, print_results=True,
                        method='MLE',
                        show_histogram_plot=False, show_PP_plot=False, show_best_distribution_probability_plot=False,
                        exclude=['Normal_2P', 'Gamma_2P', 'Loglogistic_2P', 'Gamma_3P', 'Lognormal_2P', 'Lognormal_3P',
                                 'Loglogistic_3P', 'Gumbel_2P', 'Exponential_2P', 'Exponential_1P', 'Beta_2P']
                        )

    plt.show()


#ToDo implement of different libraries and weibull distributions

#ToDo implement own function for plotting the data out of the several distribution functions

# Definition of the required part
part_name = 'HCCVREC'    #'HCCVAED'

# Create the Weibull plot with 2 different ways
#weibull_2p(part_name)
#generate_graph_local(part_name)
#weibull_fit_best(part_name)


base_dir = r"C:\Users\lgroha\cernbox\Documents\Masterthesis\3_Data-Preparation\Weibull_Plots"

part_groups = [data_weibull.parts_to_be_edited_or_changed, data_weibull.parts_with_sus_dates,
               data_weibull.parts_failed_selection]

for group_name, parts in zip(["parts_to_be_edited_or_changed", "parts_with_sus_dates", "parts_failed_selection"], part_groups):
    group_dir = os.path.join(base_dir, group_name)
    os.makedirs(group_dir, exist_ok=True)

    for part in parts:
        save_path = os.path.join(group_dir, f"weibull_plot_{part}.png")
        weibull_2p(part)    #, save_path=save_path)
