#!/usr/bin/python3
from utils import DataError, ThresholdError, NoCacheError
import pandas as pd
import db_hitdata
import threading
import warnings
import os
import time
from utils import get_logger
logger = get_logger(__name__)



weibull_cache_enabled = os.environ.get('WEIBULL_CACHE_ENABLED', 'true').lower() not in ('false', 'off', '0', 'no')

_weibull_cache = None
_cache_lock = threading.Lock()
_cache_was_loaded = False



def get_all_data(failure_threshold=4, distinct_threshold=2):
    # Only parts with failures more than the failure_threshold will be considered
    if not (isinstance(failure_threshold, int) and failure_threshold >= 1):
        raise ThresholdError(f'Invalid failure threshold "{failure_threshold}", needs to be an integer and greater than or equal as 1')

    if not (isinstance(distinct_threshold, int) and distinct_threshold >= 1):
        raise ThresholdError(f'Invalid distinct failure threshold "{distinct_threshold}", needs to be an integer and greater than or equal as 1')

    if distinct_threshold < 2:  # For Weibull Mixture at least 5 distinct failure times for 2 subdistributions
        raise ThresholdError(f'Requested less than 2 distinct failure times. Weibull Analysis not possible.')
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
        raise DataError(f'No parts found with more than {failure_threshold} failures and {distinct_threshold} distinct failures')

    print(f'Number of assets found: {weibull_data.shape[0]}')

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
                raise DataError('Unknown status "{0}"'.format(row[1]))

#Todo limit of the minimum failures and minimum distinct failures need to be adjusted | insert rule whether even with 2 distinct failures but many failure times at this times --> no good data

    if len(failures) < 2:
        raise DataError('Not enough failures (more than 2) in data for "{0}"'.format(part))
    elif len(set(failures)) < 2: #For Weibull Mixture at least 5 distinct failure times for 2 subdistributions
        raise DataError('Not enough distinct failures (more than 2) in data for "{0}"'.format(part))
    elif len(failures) < 4:
        with warnings.catch_warnings():
            warnings.simplefilter('always', UserWarning)
            warnings.warn(f'Less than 4 failures in total for "{part}"! The results should not be trusted.', UserWarning)

    return {'failures': failures, 'suspensions': suspensions, 'IRP_dates': irp_dates}


def get_parts(failure_threshold=4, distinct_threshold=2):
    # Only parts with failures more than the failure_threshold will be considered
    if not (isinstance(failure_threshold, int) and failure_threshold >= 1):
        raise ThresholdError('Invalid failure threshold "{0}", needs to be an integer and greater than or equal as 1'.format(failure_threshold))

    if not (isinstance(distinct_threshold, int) and distinct_threshold >= 1):
        raise ThresholdError('Invalid distinct failure threshold "{0}", needs to be an integer and greater than or equal as 1'.format(distinct_threshold))

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
        raise DataError(f'No parts found with more than {failure_threshold} failures and {distinct_threshold} distinct failures')

    print("Number of parts found: {0}".format(len(part_names)))

    return part_names


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
        raise DataError('The requested .csv file does not contain any data')

    print("Number of assets found: {0}".format(weibull_data.shape[0]))

    weibull_data = {name: group for name, group in weibull_data.groupby('PART')}

    return weibull_data


def refresh_cache():
    global _weibull_cache, _cache_was_loaded
    logger.info('Cache refresh for Weibull data started...')
    try:
        new_data = get_all_data()
        with _cache_lock:
            _weibull_cache = new_data
            _cache_was_loaded = True
        logger.info('Cache refresh for Weibull data completed...')
    except ThresholdError as e:
        logger.error(f'Cache refresh failed due to invalid thresholds: {e}')
    except DataError as e:
        logger.warning(f'Cache refresh found no data: {e}')
    except Exception as e:
        logger.error(f'Unexpected error during cache refresh: {e}')


def _get_cached_or_direct_data(part=None):
    if weibull_cache_enabled:
        with _cache_lock:
            cache_loaded = _cache_was_loaded
            cache_data = _weibull_cache

        if not cache_loaded:
            logger.warning(f'Cache was never loaded - triggering refresh. This may take a moment...')
            refresh_cache()
            with _cache_lock:
                cache_data = _weibull_cache

            if cache_data is None:
                raise NoCacheError('Cache refresh failed. No data available.')

        return cache_data

    else:
        if part is not None:
            logger.warning(f'Accessing the data base directly for "{part}". This may take a moment...')
            return get_data(part)
        else:
            logger.warning(f'Accessing the data base directly for every part. This may take a moment...')
            return get_all_data()


def get_failures_and_suspensions(part=None):
    if not weibull_cache_enabled and part is not None:
        return _get_cached_or_direct_data(part=part)

    data = _get_cached_or_direct_data(part=part)

    if part is not None:
        if part not in data:
            raise DataError(f'Part {part} not found in data.')

        part_df = data[part]
        return {'failures': part_df[part_df['STATUS'] == 'F']['RUNNING_TIME'].tolist(),
                'suspensions': part_df[part_df['STATUS'] == 'S']['RUNNING_TIME'].tolist(),
                'IRP_dates': part_df[part_df['STATUS'] == 'F']['FAILURE_DATE'].tolist()}

    else:
        weibull_lists = {}
        for p, df in data.items():
            failure_df = df[df['STATUS'] == 'F']
            suspension_df = df[df['STATUS'] == 'S']
            weibull_lists[p] = {'failures': failure_df['RUNNING_TIME'].tolist(),
                                'suspensions': suspension_df['RUNNING_TIME'].tolist(),
                                'IRP_dates': failure_df['FAILURE_DATE'].tolist()}

        return weibull_lists



if __name__ == '__main__':
    start = time.time()
    print(get_all_data())
    print(f'{time.time() - start} seconds to complete get_all_data()')