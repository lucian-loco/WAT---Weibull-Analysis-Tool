#!/usr/bin/python3
import db_hitdata
import pandas as pd
import threading
import logging
import warnings
import os
import time



_weibull_cache = None
_cache_lock = threading.Lock()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = False

if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)


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


def refresh_cache():
    global _weibull_cache
    logger.info('Cache refresh for Weibull data started...')
    try:
        new_data = get_all_data()
        with _cache_lock:
            _weibull_cache = new_data
        logger.info('Cache refresh for Weibull data completed...')
    except Exception as e:
        logger.error(f'Cache refresh failed: {e}')


def get_cached_data(part=None, ):
    with _cache_lock:
        if _weibull_cache is None:
            raise RuntimeError('Cache empty and not loaded yet.')

        return _weibull_cache


def get_failures_and_suspensions(part=None, data=None):
    if data is None:
        data = get_cached_data()

    if part is None:
        weibull_lists = {}
        for part, df in data.items():
            status_grouped = df.groupby('STATUS')['RUNNING_TIME']
            weibull_lists[part] = {'failures': status_grouped.get_group('F').tolist() if 'F' in df['STATUS'].values else [],
                                   'suspensions': status_grouped.get_group('S').tolist() if 'S' in df['STATUS'].values else []}

        return weibull_lists
    else:
        part_df = data[part]

        return {'failures': part_df[part_df['STATUS'] == 'F']['RUNNING_TIME'].tolist(),
                'suspensions': part_df[part_df['STATUS'] == 'S']['RUNNING_TIME'].tolist()}


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


# ToDo: Process the data further to capture failures and suspensions
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


if __name__ == '__main__':
    start = time.time()
    print(get_all_data())
    print(f'{time.time() - start} seconds to complete get_all_data()')