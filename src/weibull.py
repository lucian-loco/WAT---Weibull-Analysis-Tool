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
import pandas as pd
import matplotlib.pyplot as plt
import db_hitdata
import io
import os


# Number from 1 as threshold for the number of failures and the count of distinct failure times

# ToDo Define default thresholds
def get_parts(failure_threshold, distinct_threshold):
    # Only parts with failures more than the failure_threshold will be considered
    if not (isinstance(failure_threshold, int) and failure_threshold >= 1):
        raise RuntimeError('Invalid failure threshold "{0}"'.format(failure_threshold))

    part_names = []

    # Columns available in Weibull_data view: PART,ASSET_ID,RUNNING_TIME,STATUS,FAILURE_DATE
    with db_hitdata.get_cursor() as cursor:
        sql_query = """SELECT * FROM (
                            SELECT COUNT(*) AS FAILURES_COUNT, PART, COUNT(DISTINCT RUNNING_TIME) as DISTINCT_FAILURES_COUNT FROM weibull_data
                                    WHERE STATUS = 'F'
                                    GROUP BY PART) T
                        WHERE T.FAILURES_COUNT >= :threshold
                        AND T.DISTINCT_FAILURES_COUNT >= :distinct
                        ORDER BY T.FAILURES_COUNT"""
        result = cursor.execute(sql_query, {"threshold": failure_threshold, "distinct": distinct_threshold})

        for row in result:
            part_names.append(str(row[1]))

    if len(part_names) == 0:
        raise RuntimeError('No parts found with more than "{0}" failures'.format(failure_threshold))

    print("Number of parts found: {0}".format(len(part_names)))

    return part_names


# Number from 1 as threshold for the number of failures and the count of distinct failure times
def get_all_data(failure_threshold, distinct_threshold):
    # Only parts with failures more than the failure_threshold will be considered
    if not (isinstance(failure_threshold, int) and failure_threshold >= 1):
        raise RuntimeError('Invalid failure threshold "{0}"'.format(failure_threshold))

    if not (isinstance(distinct_threshold, int) and distinct_threshold >= 1):
        raise RuntimeError('Invalid distinct threshold "{0}"'.format(distinct_threshold))

    data = []

    # Columns available in Weibull_data view: PART,ASSET_ID,RUNNING_TIME,STATUS,FAILURE_DATE
    with db_hitdata.get_cursor() as cursor:
        sql_query = """SELECT w.PART, w.ASSET_ID, w.RUNNING_TIME, w.STATUS, w.FAILURE_DATE FROM weibull_data w
                            JOIN (
                                SELECT PART FROM weibull_data 
                                WHERE STATUS = 'F'
                                GROUP BY PART
                                HAVING COUNT(*) >= :threshold
                                AND COUNT(DISTINCT RUNNING_TIME) >= :distinct) f
                            ON f.PART = w.PART
                            ORDER BY w.PART, w.STATUS, w.FAILURE_DATE, w.ASSET_ID"""
        result = cursor.execute(sql_query, {"threshold": failure_threshold, "distinct": distinct_threshold})

        for row in result:
            data.append(row)

    weibull_data = pd.DataFrame(data, columns=['PART', 'ASSET_ID', 'RUNNING_TIME', 'STATUS', 'FAILURE_DATE'])

    if weibull_data.shape[0] == 0:
        raise RuntimeError('No parts found with more than "{0}" failures'.format(failure_threshold))

    print("Number of assets found: {0}".format(weibull_data.shape[0]))

    return weibull_data


def get_data(part):
    failures = []
    suspensions = []
    irp_dates = []

    # Columns available in Weibull_data view: PART,ASSET_ID,RUNNING_TIME,STATUS,FAILURE_DATE
    # FAILURE_DATE has the format "yyyy-mm-dd hh:mm:ss" with 24 hours format
    with db_hitdata.get_cursor() as cursor:
        sql_query = "SELECT RUNNING_TIME, STATUS, FAILURE_DATE FROM Weibull_data WHERE PART = :part_id"
        result = cursor.execute(sql_query, {"part_id": part})

        for row in result:
            if row[1] == 'S':
                suspensions.append(int(row[0]))
            elif row[1] == 'F':
                failures.append(int(row[0]))
                irp_dates.append(str(row[2]))
            else:
                raise RuntimeError('Unknown status "{0}"'.format(row[1]))

#Todo limit of the minimum failures and minimum distinct failures need to be adjusted | insert rule whether even with 2 distinct failures but many failure times at this times --> no good data

    if len(failures) < 4:
        raise RuntimeError('Not enough failures (more than 4) in data for "{0}"'.format(part))
    elif len(set(failures)) < 2: #For Weibull Mixture at least 5 distinct failure times for 2 subdistributions
        raise RuntimeError('Not enough distinct failures in data for "{0}"'.format(part))

    return {'failures': failures, 'suspensions': suspensions, 'IRP_dates': irp_dates}


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
    #if data['suspensions']:
    #    data['suspensions'] = [t for t in data['suspensions'] if t > 0]

    if not data['suspensions']:
        data['suspensions'] = None

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


# Every part-name with failures ≥ 4 that are distinct more than 2 times of the weibull_data
parts_failed = ["HCCFIRD","HCCVOJI","HCCFIOH","HCCVOJD","HCCBWMB","HCCVFEC","HCCFCIH","HCCTARA",
                "HCCBWRB","HCCVORA","HCCFIDH","HCCVOPF","HCCFIUF","HCCTGXA","HCCTRVD","HCCFCIV",
                'HCCIBBB',"HCCFIUB","HCCFIUC","HCCFCIY","HCCTDAB","HCCVOPC","HCCTDST",
                "HCCBEGU","HCCAPAC","HCCBWMF",'HCCCVAB',"HCCVOPA","HCCTDLT","HCCFEII","HCCFCIA",
                "HCCBMIA","HCCFCRJ","HCCBWDC","HCCFFIC","HCCVORB","HCCVOJB","HCCVOIA","HCCVOAA",
                "HCCFIDB","HCCTDWA","HCCFIDE","HCCVORD","HCCVOGE","HCCTDAR","HCCBWDB","HCCTDPR",
                "HCCFCRC",'HCCFIUI',"HCCVRED","HCCVREC","HCCTDAG","HCCCTMA","HCCFFIE","HCCVFEA",
                "HCCTDET","HCCFCRG","HCCVUNC","HCCVSWB","HCCTDAH","HCCVFEB","HCCTRVA","HCCVSEB",
                "HCCFEIA","HCCFISA","HCCVSEA","HCCFCRI","HCCVAED","HCCFFIB","HCCFCRB",'HCCVUEB',
                'HCCVUEA',"HCCVSWA","HCCVOTB","HCCVFWA","HCCTRP","HCCTRV",
                "HCCBWRE","HCCTRI","HCCVUNB",'HCCFCRA',"HCCFFIA"]
# parts with only '...' are not findable in the Catalogue --> out of order

# Excluded because Weibull plot not possible: "HCCVRSA", "HCCBWRF", "HCCVBRB", "HCCFIUB", "HCCTRVA", "HCCBWRE", "HCCFCRA", "HCCVOPA", "HCCVUEB", "HCCTGXA", "HCCFIUC"

# ToDo Include the .csv file to double check these parts and make it still accessible


# Refined selection of failed parts that should presumably be edited or the failures should be changed to suspended:
parts_to_be_edited_or_changed = ["HCCVSWB", "HCCFFIC", "HCCTDET", "HCCFISA",
                                 "HCCVUEA", "HCCVSWA", "HCCVFWA", "HCCFFIA",
                                 "HCCVOGE"]

# Refined selection of failed parts that contains failures with interesting dates that needs to be checked:
parts_with_sus_dates = ["HCCVOJI", "HCCVFEC", "HCCFCIH", "HCCVOPF",
                        "HCCFCIV", "HCCTDAB", "HCCTDLT", "HCCBWDC",
                        "HCCVOJB", "HCCTDAR", "HCCBWDB", "HCCTDPR",
                        "HCCCTMA", "HCCFFIB", "HCCFCRB", "HCCVOTB",
                        "HCCTRP", "HCCTRI", "HCCVOPC", "HCCVORD"]

# Refined selection of failed parts that are not sorted out yet (may contain the good data at some point):
parts_failed_selection = ["HCCFIRD", "HCCFIOH", "HCCVOJD", "HCCBWMB",
                          "HCCTARA", "HCCBWRB", "HCCVORA", "HCCFIDH",
                          "HCCFIUF", "HCCTRVD", "HCCIBBB", "HCCFCIY",
                          "HCCTDST", "HCCBEGU", "HCCAPAC",
                          "HCCBWMF", "HCCCVAB", "HCCFEII", "HCCFCIA",
                          "HCCBMIA", "HCCFCRJ", "HCCVORB", "HCCVOIA",
                          "HCCVOAA", "HCCFIDB", "HCCTDWA", "HCCFIDE",
                          "HCCFCRC", "HCCFIUI", "HCCVRED",
                          "HCCVREC", "HCCTDAG", "HCCFFIE", "HCCVFEA",
                          "HCCFCRG", "HCCVUNC", "HCCTDAH", "HCCVFEB",
                          "HCCVSEB", "HCCFEIA", "HCCVSEA", "HCCFCRI",
                          "HCCVAED", "HCCTRV", "HCCVUNB"]


# Create the Weibull plot with 2 different ways
#weibull_2p(part_name)
#generate_graph_local(part_name)
#weibull_fit_best(part_name)


base_dir = r"C:\Users\lgroha\cernbox\Documents\Masterthesis\3_Data-Preparation\Weibull_Plots"

part_groups = [parts_to_be_edited_or_changed, parts_with_sus_dates,
               parts_failed_selection]

for group_name, parts in zip(["parts_to_be_edited_or_changed", "parts_with_sus_dates", "parts_failed_selection"], part_groups):
    group_dir = os.path.join(base_dir, group_name)
    os.makedirs(group_dir, exist_ok=True)

    for part in parts:
        save_path = os.path.join(group_dir, f"weibull_plot_{part}.png")
        weibull_2p(part, save_path=save_path)


#for parts in part_groups:
#    for part in parts:
#        weibull_2p(part)

    # Weißer Platzhalter-Plot zur Trennung
#    fig, ax = plt.subplots()
#    ax.set_facecolor('white')
#    ax.set_xticks([])
#    ax.set_yticks([])
#    plt.show()