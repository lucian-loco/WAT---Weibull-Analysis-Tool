import db_hitdata
import pandas as pd
import os


# Number >= 1 as threshold for the number of failures and the count of distinct failure times
def get_parts(failure_threshold=4, distinct_threshold=2):
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


# Number >= 1 as threshold for the number of failures and the count of distinct failure times
def get_all_data(failure_threshold=4, distinct_threshold=2):
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


# Get data for Weibull Analysis out of local .csv files
def get_csv_data(query='above 3'):
    base_dir = r"C:\Users\lgroha\cernbox\Documents\Masterthesis\3_Data-Preparation\Weibull-Data_before-cleaning-Christine"

    if query == 'above 3':
        weibull_data = pd.read_csv(os.path.join(base_dir, 'Weibull_Data_parts-with-failures-above-3_2026-02-04.csv'))

    elif query == 'failures':
        weibull_data = pd.read_csv(os.path.join(base_dir, 'Weibull_Data_parts-with-failures_2026-02-04.csv'))

    elif query == 'all':
        weibull_data = pd.read_csv(os.path.join(base_dir, 'Weibull_Data_2026-02-04.csv'))

    else:
        raise RuntimeError('Unknown query "{0}"'.format(query))

    if weibull_data.shape[0] == 0:
        raise RuntimeError('The requested .csv file does not contain any data')

    print("Number of assets found: {0}".format(weibull_data.shape[0]))

    return weibull_data