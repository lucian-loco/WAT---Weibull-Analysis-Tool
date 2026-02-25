#!/usr/bin/python3
import db_hitdata
import pandas as pd
import os
import warnings


# Number >= 1 as threshold for the number of failures and the count of distinct failure times
def get_parts(failure_threshold=4, distinct_threshold=2):
    # Only parts with failures more than the failure_threshold will be considered
    if not (isinstance(failure_threshold, int) and failure_threshold >= 1):
        raise RuntimeError('Invalid failure threshold "{0}", needs to be an integer and greater than or equal as 1'.format(failure_threshold))

    if not (isinstance(distinct_threshold, int) and distinct_threshold >= 1):
        raise RuntimeError('Invalid distinct failure threshold "{0}", needs to be an integer and greater than or equal as 1'.format(distinct_threshold))

    part_names = []

    # Columns available in Weibull_data view: PART,ASSET_ID,RUNNING_TIME,STATUS,FAILURE_DATE
    with db_hitdata.get_cursor() as cursor:
        sql_query = """SELECT * FROM (
                            SELECT COUNT(*) AS FAILURES_COUNT, PART, COUNT(DISTINCT RUNNING_TIME) as DISTINCT_FAILURES_COUNT FROM weibull_data
                                    WHERE STATUS = 'F'
                                    GROUP BY PART) T
                        WHERE T.FAILURES_COUNT >= :threshold_failures
                        AND T.DISTINCT_FAILURES_COUNT >= :threshold_distinct
                        ORDER BY T.FAILURES_COUNT"""
        result = cursor.execute(sql_query, {"threshold_failures": failure_threshold, "threshold_distinct": distinct_threshold})

        for row in result:
            part_names.append(str(row[1]))

    if len(part_names) == 0:
        raise RuntimeError(f'No parts found with more than {failure_threshold} failures and {distinct_threshold} distinct failures')

    print("Number of parts found: {0}".format(len(part_names)))

    return part_names


# Number >= 1 as threshold for the number of failures and the count of distinct failure times
def get_all_data(failure_threshold=4, distinct_threshold=2):
    # Only parts with failures more than the failure_threshold will be considered
    if not (isinstance(failure_threshold, int) and failure_threshold >= 1):
        raise RuntimeError('Invalid failure threshold "{0}", needs to be an integer and greater than or equal as 1'.format(failure_threshold))

    if not (isinstance(distinct_threshold, int) and distinct_threshold >= 1):
        raise RuntimeError('Invalid distinct failure threshold "{0}", needs to be an integer and greater than or equal as 1'.format(distinct_threshold))

    if distinct_threshold < 2:  # For Weibull Mixture at least 5 distinct failure times for 2 subdistributions
        raise RuntimeError(f'Requested less than 2 distinct failure times. Weibull Analysis not possible.')
    elif failure_threshold < 4:
        with warnings.catch_warnings():
            warnings.simplefilter('always', UserWarning)
            warnings.warn('Requested less than 4 failures in total! The results should not be trusted.', UserWarning)

    data = []

    # Columns available in Weibull_data view: PART,ASSET_ID,RUNNING_TIME,STATUS,FAILURE_DATE
    with db_hitdata.get_cursor() as cursor:
        sql_query = """SELECT w.PART, w.ASSET_ID, w.RUNNING_TIME, w.STATUS, w.FAILURE_DATE FROM weibull_data w
                            JOIN (
                                SELECT PART FROM weibull_data 
                                WHERE STATUS = 'F'
                                GROUP BY PART
                                HAVING COUNT(*) >= :threshold_failures
                                AND COUNT(DISTINCT RUNNING_TIME) >= :threshold_distinct) f
                            ON f.PART = w.PART
                            ORDER BY w.PART, w.STATUS, w.FAILURE_DATE, w.ASSET_ID"""
        result = cursor.execute(sql_query, {"threshold_failures": failure_threshold, "threshold_distinct": distinct_threshold})

        for row in result:
            data.append(row)

    weibull_data = pd.DataFrame(data, columns=['PART', 'ASSET_ID', 'RUNNING_TIME', 'STATUS', 'FAILURE_DATE'])

    if weibull_data.shape[0] == 0:
        raise RuntimeError(f'No parts found with more than {failure_threshold} failures and {distinct_threshold} distinct failures')

    print("Number of assets found: {0}".format(weibull_data.shape[0]))

    weibull_data = {name: group for name, group in weibull_data.groupby('PART')}

    return weibull_data


def get_data(part):
    failures = []
    suspensions = []
    irp_dates = []

    # Columns available in Weibull_data view: PART,ASSET_ID,RUNNING_TIME,STATUS,FAILURE_DATE
    # FAILURE_DATE has the format "yyyy-mm-dd hh:mm:ss" with 24 hours format
    with db_hitdata.get_cursor() as cursor:
        sql_query = "SELECT RUNNING_TIME, STATUS, FAILURE_DATE FROM weibull_data WHERE PART = :part_id"
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

    if len(failures) < 2:
        raise RuntimeError('Not enough failures (more than 2) in data for "{0}"'.format(part))
    elif len(set(failures)) < 2: #For Weibull Mixture at least 5 distinct failure times for 2 subdistributions
        raise RuntimeError('Not enough distinct failures (more than 2) in data for "{0}"'.format(part))
    elif len(failures) < 4:
        with warnings.catch_warnings():
            warnings.simplefilter('always', UserWarning)
            warnings.warn(f'Less than 4 failures in total for "{part}"! The results should not be trusted.', UserWarning)

    return {'failures': failures, 'suspensions': suspensions, 'IRP_dates': irp_dates}


# Get data for the Weibull Analysis out of local .csv files
def get_csv_data(query='above 3'):
    base_dir = r"C:\Users\lgroha\cernbox\Documents\Masterthesis\3_Data-Preparation\Weibull-Data_before-cleaning-Christine"

    if query == 'above 3':
        weibull_data = pd.read_csv(os.path.join(base_dir, 'Weibull_Data_parts-with-failures-above-3_2026-02-04.csv'))
        print("Function returned data for parts with failures above 3.")

    elif query == 'failures':
        weibull_data = pd.read_csv(os.path.join(base_dir, 'Weibull_Data_parts-with-failures_2026-02-04.csv'))
        print("Function returned data for parts with failures.")

    elif query == 'all':
        weibull_data = pd.read_csv(os.path.join(base_dir, 'Weibull_Data_2026-02-04.csv'))
        print("Function returned every data for every part.")

    else:
        raise RuntimeError('Unknown query "{0}"'.format(query))

    if weibull_data.shape[0] == 0:
        raise RuntimeError('The requested .csv file does not contain any data')

    print("Number of assets found: {0}".format(weibull_data.shape[0]))

    weibull_data = {name: group for name, group in weibull_data.groupby('PART')}

    return weibull_data



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
#                                             "HCCTDET", "HCCFFIC", "HCCVOGE", "HCCVOJI", "HCCFCIH", "HCCVOPF", "HCCTDLT", "HCCTDPR", "HCCVOPC",

# Refined selection of failed parts that should presumably be edited or the failures should be changed to suspended:
parts_to_be_edited_or_changed = ["HCCVSWB", "HCCFISA",
                                 "HCCVUEA", "HCCVSWA", "HCCVFWA", "HCCFFIA"]

# Refined selection of failed parts that contains failures with interesting dates that needs to be checked:
parts_with_sus_dates = ["HCCVFEC",
                        "HCCFCIV", "HCCTDAB", "HCCBWDC",
                        "HCCVOJB", "HCCTDAR", "HCCBWDB",
                        "HCCCTMA", "HCCFFIB", "HCCFCRB", "HCCVOTB",
                        "HCCTRP", "HCCTRI", "HCCVORD"]

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
