#!/usr/bin/python3
from utils import DataError, ThresholdError, NoCacheError
import pandas as pd
import db_hitdata
import threading
import warnings
import os
import datetime
from zoneinfo import ZoneInfo
from utils import get_logger
logger = get_logger(__name__)


# os.environ['WEIBULL_CACHE_ENABLED'] = 'false'

# The env variable WEIBULL_CACHE_ENABLED is true by default
weibull_cache_enabled = os.environ.get('WEIBULL_CACHE_ENABLED', 'true').lower() not in ('false', 'off', '0', 'no')

_weibull_cache = None
_cache_timestamp = None
_cache_lock = threading.Lock()
# Variable to set for the data that will be used in the automated Weibull analysis.
failure_threshold_global = 4



REQUIRED_COLUMNS = ['PART', 'ASSET_ID', 'RUNNING_TIME', 'STATUS', 'FAILURE_DATE', 'CURRENT_STATE', 'FULL_RUNNING_TIME']


def _validate_thresholds(failure_threshold, distinct_threshold):
    if not (isinstance(failure_threshold, int) and failure_threshold >= 1):
        raise ThresholdError(f'Invalid failure threshold "{failure_threshold}", needs to be an integer and >= 1')

    if not (isinstance(distinct_threshold, int) and distinct_threshold >= 1):
        raise ThresholdError(f'Invalid distinct failure threshold "{distinct_threshold}", needs to be an integer and >= 1')

    if distinct_threshold < 2:
        raise ThresholdError('Requested less than 2 distinct failure times. Weibull Analysis not possible.')

    if failure_threshold < 4:
        with warnings.catch_warnings():
            warnings.simplefilter('always', UserWarning)
            warnings.warn('Requested less than 4 failures in total! The results should not be trusted.', UserWarning)


def _normalize_weibull_df(df):
    if df.empty:
        return df

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataError(f'Missing required columns in Weibull data: {missing}')

    df = df.copy()
    df['PART'] = df['PART'].astype(str).str.strip()
    df['ASSET_ID'] = df['ASSET_ID'].astype(str).str.strip()
    df['STATUS'] = df['STATUS'].astype(str).str.strip().str.upper()
    df['CURRENT_STATE'] = df['CURRENT_STATE'].astype(str).str.strip().str.upper()

    df['RUNNING_TIME'] = pd.to_numeric(df['RUNNING_TIME'], errors='coerce')
    df['FULL_RUNNING_TIME'] = pd.to_numeric(df['FULL_RUNNING_TIME'], errors='coerce')
    df['FAILURE_DATE'] = pd.to_datetime(df['FAILURE_DATE'], errors='coerce')

    invalid_status = set(df['STATUS'].dropna().unique()) - {'F', 'S'}
    if invalid_status:
        raise DataError(f'Unknown STATUS values found: {sorted(invalid_status)}')

    df = df.dropna(subset=['RUNNING_TIME', 'FULL_RUNNING_TIME'])
    df = df[(df['RUNNING_TIME'] >= 0) & (df['FULL_RUNNING_TIME'] >= 0)].copy()

    return df


def get_all_data(failure_threshold=failure_threshold_global, distinct_threshold=2):
    # Only parts with failures more than the failure_threshold and the distinct_threshold will be considered
    _validate_thresholds(failure_threshold=failure_threshold, distinct_threshold=distinct_threshold)

    data = []

    # Columns available in Weibull_data view: PART,ASSET_ID,RUNNING_TIME,STATUS,FAILURE_DATE
    with db_hitdata.get_cursor() as cursor:
        sql_query = """SELECT w.PART, w.ASSET_ID, w.RUNNING_TIME, w.STATUS, w.FAILURE_DATE, w.CURRENT_STATE, w.FULL_RUNNING_TIME FROM weibull_data w
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

    weibull_data = _normalize_weibull_df(pd.DataFrame(data, columns=REQUIRED_COLUMNS))

    if weibull_data.empty:
        raise DataError(f'No parts found with more than {failure_threshold} failures and {distinct_threshold} distinct failure times.')

    print(f'Number of assets found: {weibull_data.shape[0]}')

    weibull_data = {name: group for name, group in weibull_data.groupby('PART')}

    return weibull_data


def get_data(part):
    # Columns available in Weibull_data view: PART,ASSET_ID,RUNNING_TIME,STATUS,FAILURE_DATE,CURRENT_STATE
    # FAILURE_DATE has the format "yyyy-mm-dd hh:mm:ss" with 24 hours format
    with db_hitdata.get_cursor() as cursor:
        sql_query = ("""SELECT PART, ASSET_ID, RUNNING_TIME, STATUS, FAILURE_DATE, CURRENT_STATE, FULL_RUNNING_TIME FROM weibull_data
                        WHERE PART = :part_id
                        ORDER BY ASSET_ID, STATUS, FAILURE_DATE""")
        result = cursor.execute(sql_query, {"part_id": part})
        rows = list(result)

    df = _normalize_weibull_df(pd.DataFrame(rows, columns=REQUIRED_COLUMNS))

    if df.empty:
        raise DataError(f'No data found for "{part}"')

    failure_df = df[df['STATUS'] == 'F'].copy()
    suspension_df = df[df['STATUS'] == 'S'].copy()

    if len(failure_df) < 2:
        raise DataError(f'Not enough failures (more than 2) in data for "{part}".')

    if failure_df['RUNNING_TIME'].nunique() < 2: #For Weibull Mixture at least 5 distinct failure times for 2 subdistributions
        raise DataError(f'Not enough distinct failures (more than 2) in data for "{part}"')

    if len(failure_df) < 4:
        with warnings.catch_warnings():
            warnings.simplefilter('always', UserWarning)
            warnings.warn(f'Less than 4 failures in total for "{part}"! The results should not be trusted.', UserWarning)

    installed_df = df[df['CURRENT_STATE'] == 'I'].copy()

    return {'part': part,
            'all_assets': df.reset_index(drop=True),
            'installed_assets': installed_df.reset_index(drop=True),
            'failures': failure_df['RUNNING_TIME'].tolist(),
            'suspensions': suspension_df['RUNNING_TIME'].tolist(),
            'IRP_dates': failure_df['FAILURE_DATE'].tolist()
            }


def get_parts(failure_threshold=failure_threshold_global, distinct_threshold=2):
    # Only parts with failures more than the failure_threshold will be considered
    _validate_thresholds(failure_threshold=failure_threshold, distinct_threshold=distinct_threshold)

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


# ToDo: Process the data further to capture failures and suspensions and current states of the new data as well
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
    global _weibull_cache, _cache_timestamp
    logger.info('Cache refresh for Weibull data started...')
    try:
        new_data = get_all_data()
        with _cache_lock:
            _weibull_cache = new_data
            _cache_timestamp = datetime.datetime.now(tz=ZoneInfo('Europe/Zurich'))
        logger.info('Cache refresh for Weibull data completed...')
    except ThresholdError as e:
        logger.error(f'Cache refresh failed due to invalid thresholds: {e}')
    except DataError as e:
        logger.warning(f'Cache refresh found no data: {e}')
    except Exception as e:
        logger.error(f'Unexpected error during cache refresh: {e}')


def get_cache_timestamp():
    return _cache_timestamp


def _get_cached_or_direct_data(part=None):
    if weibull_cache_enabled:
        if _weibull_cache is None:
            logger.warning(f'Cache was never loaded - triggering refresh. This may take a moment...')
            refresh_cache()

        if _weibull_cache is None:
            raise NoCacheError('Cache refresh failed. No data available.')

        return _weibull_cache

    else:
        if part is not None:
            logger.warning(f'Accessing the data base directly for "{part}". This may take a moment...')
            return get_data(part)
        else:
            logger.warning(f'Accessing the data base directly for every part. This may take a moment...')
            return get_all_data()


def get_failures_and_suspensions(part=None, failure_threshold=failure_threshold_global):
    # This exception is needed since the return of data_weibull.get_parts() is already formatted correctly
    if not weibull_cache_enabled and part is not None:
        return _get_cached_or_direct_data(part=part)

    data = _get_cached_or_direct_data(part=part)

    if part is None:
        weibull_lists = {}
        for p, df in data.items():
            failure_df = df[df['STATUS'] == 'F'].copy()
            suspension_df = df[df['STATUS'] == 'S'].copy()
            installed_df = df[df['CURRENT_STATE'] == 'I'].copy()

            weibull_lists[p] = {'failures': failure_df['RUNNING_TIME'].tolist(),
                                'suspensions': suspension_df['RUNNING_TIME'].tolist(),
                                'IRP_dates': failure_df['FAILURE_DATE'].tolist(),
                                'installed_assets': installed_df[['ASSET_ID', 'RUNNING_TIME', 'STATUS', 'CURRENT_STATE', 'FULL_RUNNING_TIME']].to_dict('records'),
                                'all_assets': df[['ASSET_ID', 'RUNNING_TIME', 'STATUS', 'FAILURE_DATE', 'CURRENT_STATE', 'FULL_RUNNING_TIME']].to_dict('records')}

        return weibull_lists

    else:
        if part not in data:
            raise DataError(f'Part {part} not found in data. Probably there were too few failures. '
                            f'Data contains only parts with more than {failure_threshold} failures.')

        part_df = data[part].copy()
        failure_df = part_df[part_df['STATUS'] == 'F'].copy()
        suspension_df = part_df[part_df['STATUS'] == 'S'].copy()
        installed_df = part_df[part_df['CURRENT_STATE'] == 'I'].copy()

        return {'failures': failure_df['RUNNING_TIME'].tolist(),
                'suspensions': suspension_df['RUNNING_TIME'].tolist(),
                'IRP_dates': failure_df['FAILURE_DATE'].tolist(),
                'installed_assets': installed_df[['ASSET_ID', 'RUNNING_TIME', 'STATUS', 'CURRENT_STATE', 'FULL_RUNNING_TIME']].to_dict('records'),
                'all_assets': part_df[['ASSET_ID', 'RUNNING_TIME', 'STATUS', 'FAILURE_DATE', 'CURRENT_STATE', 'FULL_RUNNING_TIME']].to_dict('records')}
