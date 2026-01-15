#!/usr/bin/python3
#Using the predictr library is fine but improvements are done with the Reliability package: https://reliability.readthedocs.io/en/latest/index.html
from predictr import Analysis
from reliability.Fitters import Fit_Weibull_2P
from reliability.Fitters import Fit_Weibull_3P
from reliability.Fitters import Fit_Weibull_Mixture
from reliability.Fitters import Fit_Everything
import matplotlib.pyplot as plt
import db_hitdata
import io


def get_data(part):
    failures = []
    suspensions = []

    # Columns available in Weibull_data view: PART,ASSET_ID,RUNNING_TIME,STATUS
    with db_hitdata.get_cursor() as cursor:
        sql_query = "SELECT RUNNING_TIME, STATUS FROM Weibull_data WHERE PART = :part_id"
        result = cursor.execute(sql_query, (part,))

        for row in result:
            if row[1] == 'S':
                suspensions.append(int(row[0]))
            elif row[1] == 'F':
                failures.append(int(row[0]))
            else:
                raise RuntimeError('Unknown status "{0}"'.format(row[1]))

    if len(failures) < 2:
        raise RuntimeError('Not enough failure data for "{0}"'.format(part))

    return {'failures': failures, 'suspensions': suspensions}


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


#ToDo for schleife um das ganze bauen fuer alle failure parts (daten aus sql query)
def weibull_2p(part):
    if not part:
        raise RuntimeError('Invalid request ("part" not specified)')

    data = get_data(part)

    failure_size = len(data['failures'])
    suspension_size = len(data['suspensions'])
    sample_size = failure_size + suspension_size

    # Weibull Analysis
    # see https://reliability.readthedocs.io/en/latest/API/Fitters.html for parameters description

    wb = Fit_Weibull_2P(failures=data['failures'], right_censored=data['suspensions'],
                        show_probability_plot=True, print_results=True,
                        method='MLE',
                        CI_type='none', # In case of CI --> CI='float between 0 and 1'
                        label=f'Weibull fit (n = {sample_size} (f: {failure_size} | s: {suspension_size})'
                        )

    plt.title(f'Weibull Probability Plot for {part} with (α={wb.alpha:.3f}, β={wb.beta:.3f})')
    ax = plt.gca()
    ax.set_xlabel('Time in days')
    ax.set_ylabel('Unreliability')
    #Todo size of the plot and axis limits still to be adjusted
    #plt.ylim([0.5, 99])
    #plt.figure(figsize=(9.5, 6))
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
                        show_probability_plot=True, print_results=True,
                        method='MLE',
                        show_histogram_plot=False, show_PP_plot=False, show_best_distribution_probability_plot=True,
                        exclude=['Normal_2P', 'Gamma_2P', 'Loglogistic_2P', 'Gamma_3P', 'Lognormal_2P', 'Lognormal_3P', 'Loglogistic_3P', 'Gumbel_2P', 'Exponential_2P', 'Exponential_1P', 'Beta_2P']
                        #label=f'Weibull fit (n = {sample_size} (f: {failure_size} | s: {suspension_size})'
                        )

    #plt.title(f'Weibull Probability Plot for {part} with (α={wb.alpha:.3f}, β={wb.beta:.3f})')
    #Todo size of the plot and axis limits still to be adjusted (maybe inside library directly)
    #plt.ylim([0.5, 99])
    #plt.figure(figsize=(9.5, 6))
    plt.show()


#ToDo implement of different libraries and weibull distributions

# Test the Weibull plot directly
part_name = 'HCCTDWA' #'HCCFCRA'
#weibull_2p(part_name)
weibull_fit_best(part_name)