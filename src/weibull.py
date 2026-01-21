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

#Todo limit of the minimum failures and minimum distinct failures need to be adjusted

    if len(failures) < 4:
        raise RuntimeError('Not enough failures (more than 4) in data for "{0}"'.format(part))
    elif len(set(failures)) < 2:
        raise RuntimeError('Not enough distinct failures in data for "{0}"'.format(part))

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


#old library and current state but as local version
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


def weibull_2p(part):
    if not part:
        raise RuntimeError('Invalid request ("part" not specified)')

    data = get_data(part)

    failure_size = len(data['failures'])
    suspension_size = len(data['suspensions'])
    sample_size = failure_size + suspension_size

    if not data['suspensions']:
        data['suspensions'] = None

    # Weibull Analysis
    # see https://reliability.readthedocs.io/en/latest/API/Fitters.html for parameters description

    wb = Fit_Weibull_2P(failures=data['failures'], right_censored=data['suspensions'],
                        show_probability_plot=True, print_results=False, # Results can be found in the returned variables as well
                        method='MLE',
                        CI_type='none', # In case of CI --> CI='float between 0 and 1'
                        label=f'Weibull 2 Parameter fit | MLE \n (n = {sample_size} (f: {failure_size} | s: {suspension_size})'
                        )

    plt.title(f'Weibull Probability Plot for {part} with \n (α={wb.alpha:.3f}, β={wb.beta:.3f})')
    ax = plt.gca()
    ax.set_xlabel('Time in days')
    ax.set_ylabel('Unreliability')
    ax.set_ylim(0.001, 0.999)
# Todo x-axis limit needs to be adjusted
    xmin, xmax = ax.get_xlim()
    ax.set_xlim(xmin * 0.8, xmax * 1.2)
    labels = ax.get_xticklabels()
    for i, label in enumerate(labels):  # Prevents overlapping of x-axis' ticks
        label.set_visible(i < 3 or (i - 3) % 2 == 0)
    fig = plt.gcf()
    fig.set_size_inches(9.5, 6)
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
                        exclude=['Normal_2P', 'Gamma_2P', 'Loglogistic_2P', 'Gamma_3P', 'Lognormal_2P', 'Lognormal_3P', 'Loglogistic_3P', 'Gumbel_2P', 'Exponential_2P', 'Exponential_1P', 'Beta_2P']
                        )

    plt.show()


#ToDo implement of different libraries and weibull distributions

#ToDo implement own function for plotting the data out of the several distribution functions

# Definition of the required part
part_name = 'HCCTDAR'            #'HCCTGXA'        #'HCCFIUB'            #'HCCVREC'


#ToDo parts_failed automatically out of data base with sql query (daten aus sql query)

# Every part-name with failures ≥ 4 that are distinct more than 2 times of the weibull_data
parts_failed = ["HCCFIRD","HCCVOJI","HCCFIOH","HCCVOJD","HCCBWMB","HCCVFEC","HCCFCIH","HCCTARA",
            "HCCBWRB","HCCVORA","HCCFIDH","HCCVOPF","HCCFIUF","HCCTGXA","HCCTRVD","HCCFCIV",
            "HCCIBBB","HCCFIUB","HCCFIUC","HCCFCIY","HCCTDAB","HCCVOPC","HCCTDST",
            "HCCBEGU","HCCAPAC","HCCBWMF","HCCCVAB","HCCVOPA","HCCTDLT","HCCFEII","HCCFCIA",
            "HCCBMIA","HCCFCRJ","HCCBWDC","HCCFFIC","HCCVORB","HCCVOJB","HCCVOIA","HCCVOAA",
            "HCCFIDB","HCCTDWA","HCCFIDE","HCCVORD","HCCVOGE","HCCTDAR","HCCBWDB","HCCTDPR",
            "HCCFCRC","HCCFIUI","HCCVRED","HCCVREC","HCCTDAG","HCCCTMA","HCCFFIE","HCCVFEA",
            "HCCTDET","HCCFCRG","HCCVUNC","HCCVSWB","HCCTDAH","HCCVFEB","HCCTRVA","HCCVSEB",
            "HCCFEIA","HCCFISA","HCCVSEA","HCCFCRI","HCCVAED","HCCFFIB","HCCFCRB","HCCVUEB",
#start from here completely new
            "HCCVUEA","HCCVSWA","HCCVOTB","HCCVFWA","HCCTRP","HCCTRV",
            "HCCBWRE","HCCTRI","HCCVUNB","HCCFCRA","HCCFFIA"]

# Refined selection of failed parts where it's noticeable that many asset failed at the same time --> look into it whether it was the same date --> building up criteria
parts_failed_at_once = []

# Refined selection of failed parts where it's noticeable in the Weibull plot that there are probably multiple failure modes hidden in the data
parts_mult_failure_mode = []


# Create the Weibull plot with 2 different ways
#weibull_2p(part_name)
#generate_graph_local(part_name)
#weibull_fit_best(part_name)



#ToDo sort out the not functional ones for Weibull (!)
for part in parts_failed:
    weibull_2p(part)
