#!/usr/bin/python3
import os
import warnings
import db_hitdata
import pandas as pd
from weibull import weibull_fit_best
from data_weibull import get_all_data
from weibull_evaluation import compare_best_distribution
from weibull_forecast import forecast_all_parts_direct_delta
from utils import DataError, ThresholdError
from utils import get_logger
logger = get_logger(__name__)



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


def get_all_snapshot_data(failure_threshold=4, distinct_threshold=2, view_name='weibull_data'):
    # Only parts with failures more than the failure_threshold and the distinct_threshold will be considered
    _validate_thresholds(failure_threshold=failure_threshold, distinct_threshold=distinct_threshold)

    data = []

    # Columns available in Weibull_data view: PART,ASSET_ID,RUNNING_TIME,STATUS,FAILURE_DATE
    with db_hitdata.get_cursor() as cursor:
        sql_query = f"""SELECT w.PART, w.ASSET_ID, w.RUNNING_TIME, w.STATUS, w.FAILURE_DATE, w.CURRENT_STATE, w.FULL_RUNNING_TIME FROM "{view_name}" w
                            JOIN (
                                SELECT PART FROM "{view_name}" 
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

    logger.info(f'Number of assets found: {weibull_data.shape[0]}')

    weibull_data = {name: group for name, group in weibull_data.groupby('PART')}

    weibull_lists = {}

    for p, df in weibull_data.items():
        failure_df = df[df['STATUS'] == 'F'].copy()
        suspension_df = df[df['STATUS'] == 'S'].copy()
        installed_df = df[df['CURRENT_STATE'] == 'I'].copy()

        weibull_lists[p] = {'failures': failure_df['RUNNING_TIME'].tolist(),
                            'suspensions': suspension_df['RUNNING_TIME'].tolist(),
                            'IRP_dates': failure_df['FAILURE_DATE'].tolist(),
                            'installed_assets': installed_df[
                                ['ASSET_ID', 'RUNNING_TIME', 'STATUS', 'CURRENT_STATE', 'FULL_RUNNING_TIME']].to_dict(
                                'records'),
                            'all_assets': df[['ASSET_ID', 'RUNNING_TIME', 'STATUS', 'FAILURE_DATE', 'CURRENT_STATE',
                                              'FULL_RUNNING_TIME']].to_dict('records')}

    return weibull_lists


def weibull_analysis_snapshot(snapshot_data, snapshot, sort_by):
    results_snapshots = {}
    errors = {}

    for part, data in snapshot_data.items():
        try:
            # weibull_fit_best always uses 'BIC' internally; CV is applied only in compare_best_distribution via the sort_by argument
            sort_for_fit = sort_by if sort_by != 'CV' else 'BIC'
            fit_table, _, _, fit_status = weibull_fit_best(part=part, sort_by=sort_for_fit, data=data)

            best_model, cv_used = compare_best_distribution(df=fit_table, sort_by=sort_by, part=part, data=data,
                                                            ic_fallback='BIC', delta=delta_ic, fit_status=fit_status)

            results_snapshots[part] = {'best_model': best_model,
                                       'fit_table': fit_table,
                                       'data': data,
                                       'cv_used': cv_used
                                       }
        except Exception as e:
            errors[part] = str(e)
            logger.warning(f'Analysis for snapshot {snapshot}: skipped "{part}": {e}')

    return results_snapshots



if __name__ == '__main__':
    weibull_snapshots = ['WEIBULL_DATA_snapshot_2018-01-01', 'WEIBULL_DATA_snapshot_2020-01-01',
                         'WEIBULL_DATA_snapshot_2022-01-01', 'WEIBULL_DATA_snapshot_2024-01-01']

    sort_by = 'AIC'
    # Defined with delta tuning script
    delta_ic = 0.466
    ci = 0.95
    # To be defined
    output_dir = r''

    dates = ['2020-01-01', '2022-01-01', '2024-01-01', '2026-01-01', '2026-07-06']
    forecast_dates = pd.to_datetime(dates)

    logger.info('Pre-fetching real failure data for all parts from weibull_data...')
    real_data_cache = get_all_data()
    # Build a per-part lookup: part -> sorted array of real IRP_dates (failures only)
    real_irp_cache = {}
    for part, df in real_data_cache.items():
        real_irp_cache[part] = df.loc[df['STATUS'] == 'F', 'FAILURE_DATE'].dropna().values
    logger.info(f'Real IRP dates cached for {len(real_irp_cache)} parts.')

    for snapshot in weibull_snapshots:
        snapshot_date_str = snapshot.split('_')[-1]
        snapshot_date = pd.to_datetime(snapshot_date_str)

        # Filter forecast dates: exclude dates that are <= snapshot_date (past or equal)
        valid_forecast_dates = forecast_dates[forecast_dates > snapshot_date]

        deltas = (valid_forecast_dates - snapshot_date).days.tolist()

        # Fetching all the data of the weibull_data snapshot
        snapshot_data_all = get_all_snapshot_data(view_name=snapshot)

        logger.info(f'Start now to fit all the data for {snapshot}...')

        # Calculating all the weibull analysis results of based on the current snapshot
        snapshot_results = weibull_analysis_snapshot(snapshot_data=snapshot_data_all, snapshot=snapshot, sort_by=sort_by)

        logger.info(f'Start now to calculate the forecast for {snapshot}...')

        # Calculate the expected number of failures forecast based on the current snapshot
        result_forecast = forecast_all_parts_direct_delta(deltas=deltas, CI=ci, cached_results=snapshot_results, skip_errors=True, data_prepared=snapshot_data_all)

        # Convert result_forecast to DataFrame rows for this snapshot
        snapshot_rows = []
        for part_name, forecast_dict in result_forecast['results'].items():
            # forecast_dict: {'part', 'best_model', 'n_installed', 'results', 'fit_table'}
            # results is a list of dicts with: delta, n_installed, expected_failures, lower_bound, upper_bound, standard_error, variance

            irp_dates = real_irp_cache.get(part_name, None)

            for idx, forecast_result in enumerate(forecast_dict['results']):
                forecast_date = valid_forecast_dates[idx]
                # Count actual failures in the interval (snapshot_date, forecast_date]
                if irp_dates is not None and len(irp_dates) > 0:
                    actual_failures = int(((irp_dates > snapshot_date) & (irp_dates <= forecast_date)).sum())
                else:
                    actual_failures = None

                row = {
                    'snapshot': snapshot,
                    'snapshot_date': snapshot_date_str,
                    'delta_days': int(deltas[idx]),
                    'forecast_date': valid_forecast_dates[idx].strftime('%Y-%m-%d'),
                    'part': forecast_dict['part'],
                    'best_model': forecast_dict['best_model'],
                    'n_installed': forecast_dict['n_installed'],
                    'expected_failures': forecast_result['expected_failures'],
                    'ci_lower': forecast_result['lower_bound'],
                    'ci_upper': forecast_result['upper_bound'],
                    'standard_error': forecast_result['standard_error'],
                    'variance': forecast_result['variance'],
                    'actual_failures': actual_failures
                }
                snapshot_rows.append(row)

        # Log errors for this snapshot
        if result_forecast['errors']:
            logger.warning(f'Errors for {snapshot}: {result_forecast["errors"]}')

        # Save DataFrame for this snapshot to CSV
        if snapshot_rows:
            output_path = os.path.join(output_dir, f'weibull_forecast_{snapshot_date_str}_AIC.csv')
            df_snapshot = pd.DataFrame(snapshot_rows)
            df_snapshot.to_csv(output_path, index=False)
            logger.info(f'Saved forecast for {snapshot} to {output_path} ({len(df_snapshot)} rows)')
        else:
            logger.warning(f'No forecast results for {snapshot} to save.')