#!/usr/bin/python3
import os
import itertools
import warnings
import numpy as np
import pandas as pd
from reliability.Distributions import Weibull_Distribution



"""
Script for synthetic Weibull data generation.
Generates 2P, 3P, Competing Risk, and Mixture datasets with right-censoring
and optional holdout set for predictability evaluation.

Output: dict (in-memory) and optionally .csv files per dataset.

Usage in Validate_Weibull_CI.py:
    from generate_synthetic_data import generate_all_datasets
    datasets = generate_all_datasets()
"""
#-----------------------------------------------------------------------------------------------------------------------
# CONFIGURATION — adjust here
#-----------------------------------------------------------------------------------------------------------------------
SIMULATE_2P  = False
SIMULATE_3P  = False
SIMULATE_CR  = False
SIMULATE_MIX = False

N_SAMPLES_VALUES          = [10, 50, 100, 500, 1000]            # Sample size without censoring
CENSOR_RATE_VALUES        = [0.2, 0.8, 0.9, 0.95]               # Censor rates
CENSOR_RATE_VALUES_CR_MIX = [0.1, 0.3, 0.7, 0.9]                # Censor rates for Mixture and CR

ALPHA_VALUES = [1000, 2500, 8000, 15000]                        # Scale
BETA_VALUES  = [0.5, 1.5, 3.0, 12.0]                            # Shape
GAMMA_VALUES = [100, 365, 730, 3000, 6000]                      # Location for Weibull 3P

CR_ALPHA2_VALUES  = [1000, 2500, 8000, 15000]
CR_BETA2_VALUES   = [0.5, 1.5, 3.0, 12.0]
MIN_PCT_VALUES    = [0.2, 0.4]                                  # CR specific value
PROPORTION_VALUES = [0.2, 0.4]                                  # Proportion for Weibull Mixture

N_LOOPS = 10                                                    # Count of different seeds

SAVE_CSV = True
# To be defined
CSV_OUTPUT_DIR = r""
CSV_OUTPUT_RELIASOFT_DIR = r""

# Lookup-Dict: which censor rate is used for the distributions
CENSOR_RATES_BY_TYPE = {'2P' : CENSOR_RATE_VALUES,
                        '3P' : CENSOR_RATE_VALUES,
                        'CR' : CENSOR_RATE_VALUES_CR_MIX,
                        'Mix': CENSOR_RATE_VALUES_CR_MIX}


#-----------------------------------------------------------------------------------------------------------------------
# CR SEED VALIDATION
#-----------------------------------------------------------------------------------------------------------------------
def validate_cr_scenario(alpha1, beta1, alpha2, beta2, n_samples, min_pct, needed_count=10, max_scan=1000):
    """
    Pre-scans random seeds to find ones that produce a balanced split
    between the two CR failure mechanisms.

    A seed is accepted when both mechanisms cause at least
    (n_samples * min_pct) failures in the simulated dataset.
    This ensures both components are estimable by the MLE fitter.
    """
    valid_seeds  = []
    min_required = int(n_samples * min_pct)

    for seed in range(max_scan):
        np.random.seed(seed)
        t1     = Weibull_Distribution(alpha=alpha1, beta=beta1).random_samples(n_samples)
        t2     = Weibull_Distribution(alpha=alpha2, beta=beta2).random_samples(n_samples)
        count1 = np.sum(t1 < t2)

        if count1 >= min_required and (n_samples - count1) >= min_required:
            valid_seeds.append(seed)
            if len(valid_seeds) >= needed_count:
                return valid_seeds

    return valid_seeds


#-----------------------------------------------------------------------------------------------------------------------
# TEST PLAN GENERATION
#-----------------------------------------------------------------------------------------------------------------------
def generate_test_plans():
    """
    Generates a list of scenario dicts, one per parameter combination.
    Each plan fully describes one simulation scenario including data type,
    distribution parameters, sample size, and censoring rate.
    """
    plans = []

    # --- 2P: min_pct not applicable ---
    if SIMULATE_2P:
        for n_s, z_r, alpha, beta in itertools.product(N_SAMPLES_VALUES, CENSOR_RATES_BY_TYPE['2P'], ALPHA_VALUES, BETA_VALUES):
            expected_failures = int(n_s * (1 - z_r))
            if expected_failures <= 3:
                continue
            plans.append({
                'data_type': '2P',
                'alpha': alpha, 'beta': beta,
                'n_samples': n_s, 'censor_rate': z_r,
                'min_pct': np.nan  # not used for 2P
            })

    # --- 3P: min_pct not applicable ---
    if SIMULATE_3P:
        for n_s, z_r, alpha, beta, gamma in itertools.product(N_SAMPLES_VALUES, CENSOR_RATES_BY_TYPE['3P'], ALPHA_VALUES, BETA_VALUES, GAMMA_VALUES):
            expected_failures = int(n_s * (1 - z_r))
            if expected_failures <= 3:
                continue
            plans.append({
                'data_type': '3P',
                'alpha': alpha, 'beta': beta, 'gamma': gamma,
                'n_samples': n_s, 'censor_rate': z_r,
                'min_pct': np.nan  # not used for 3P
            })

    # --- CR: min_pct actively used in validate_cr_scenario ---
    if SIMULATE_CR:
        for n_s, z_r, m_p, a1, b1, a2, b2 in itertools.product(N_SAMPLES_VALUES, CENSOR_RATES_BY_TYPE['CR'], MIN_PCT_VALUES, ALPHA_VALUES, BETA_VALUES, CR_ALPHA2_VALUES, CR_BETA2_VALUES):
            if a1 == a2 and b1 == b2:
                continue
            # Both distributions need to have enough failures after censoring, first check
            limit_idx = int(n_s * (1 - z_r))
            if limit_idx < 16:
                continue
            good_seeds = validate_cr_scenario(a1, b1, a2, b2, n_s, m_p, needed_count=N_LOOPS)
            if len(good_seeds) >= N_LOOPS:
                plans.append({
                    'data_type': 'CR',
                    'alpha1': a1, 'beta1': b1,
                    'alpha2': a2, 'beta2': b2,
                    'n_samples': n_s, 'censor_rate': z_r,
                    'min_pct': m_p,  # actively used
                    'good_seeds': good_seeds
                })
            else:
                warnings.warn(f"CR scenario (a1={a1}, b1={b1}, a2={a2}, b2={b2}) "
                              f"found only {len(good_seeds)} valid seeds — skipped.", UserWarning)

    # --- Mix: min_pct not applicable ---
    if SIMULATE_MIX:
        for n_s, z_r, a1, b1, a2, b2, prop in itertools.product(N_SAMPLES_VALUES, CENSOR_RATES_BY_TYPE['Mix'], ALPHA_VALUES, BETA_VALUES, CR_ALPHA2_VALUES, CR_BETA2_VALUES, PROPORTION_VALUES):
            if a1 == a2 and b1 == b2:
                continue
            # Both distributions need to have enough failures after censoring, first check
            limit_idx = int(n_s * (1 - z_r))
            if limit_idx < 16:
                continue
            plans.append({
                'data_type': 'Mix',
                'alpha1': a1, 'beta1': b1,
                'alpha2': a2, 'beta2': b2,
                'n_samples': n_s, 'censor_rate': z_r,
                'min_pct': np.nan,  # not used for Mix
                'proportion': prop
            })

    return plans


#-----------------------------------------------------------------------------------------------------------------------
# SINGLE DATASET GENERATION
#-----------------------------------------------------------------------------------------------------------------------
def generate_single_dataset(seed, plan):
    """
    Generates one synthetic dataset for a given seed and scenario plan.

    Applies Type-I right-censoring: the last observed failure in the
    training window defines the censoring time. All units beyond that
    time are censored at the SAME fixed time point (train_censored).
    Units beyond that time are retained as holdout failures, which can
    be used for:
        - Holdout RMSE (predictability evaluation)
        - Predictive CB coverage (do future failures fall inside the CB band?)

    Parameters
    ----------
    seed : int
    plan : dict  — one entry from generate_test_plans()

    Returns
    -------
    dataset : dict with keys:
        seed, data_type, n_samples, censor_rate,
        true_alpha_1, true_beta_1, true_gamma, true_alpha_2, true_beta_2,
        cr_risk1_share  : float — fraction of failures from mechanism 1 (CR only)
        failures        : np.ndarray — training failures
        censored        : np.ndarray — training censored (all equal for Type-I)
        holdout         : np.ndarray — true future failures (evaluation only)
        raw_data        : np.ndarray — full unsplit sample (for reference)
    Returns None if generation fails.
    """
    # warnings.simplefilter('always')
    warnings.filterwarnings('ignore')
    np.random.seed(seed)

    data_type   = plan['data_type']
    n_samples   = plan['n_samples']
    censor_rate = plan['censor_rate']

    dataset = {
        'seed'          : seed,
        'data_type'     : data_type,
        'n_samples'     : n_samples,
        'censor_rate'   : censor_rate,
        'true_alpha_1'  : plan.get('alpha',  plan.get('alpha1', np.nan)),
        'true_beta_1'   : plan.get('beta',   plan.get('beta1',  np.nan)),
        'true_gamma'    : plan.get('gamma',  np.nan),
        'true_alpha_2'  : plan.get('alpha2', np.nan),
        'true_beta_2'   : plan.get('beta2',  np.nan),
        'cr_risk1_share': np.nan,
        'failures'      : None,
        'censored'      : None,
        'holdout'       : None,
        'raw_data'      : None,
    }

    try:
        # ------------------------------------------------------------------
        # Raw data generation
        # ------------------------------------------------------------------
        if data_type == '2P':
            raw_data = Weibull_Distribution(alpha=plan['alpha'], beta=plan['beta']).random_samples(n_samples)

        elif data_type == '3P':
            raw_data = Weibull_Distribution(alpha=plan['alpha'], beta=plan['beta'], gamma=plan['gamma']).random_samples(n_samples)

        elif data_type == 'CR':
            # Explicit seed reset ensures the same generator state as
            # validate_cr_scenario — guaranteeing the same balance.
            np.random.seed(seed)
            t1       = Weibull_Distribution(alpha=plan['alpha1'], beta=plan['beta1']).random_samples(n_samples)
            t2       = Weibull_Distribution(alpha=plan['alpha2'], beta=plan['beta2']).random_samples(n_samples)
            raw_data = np.minimum(t1, t2)
            dataset['cr_risk1_share'] = float(np.mean(t1 < t2))

        elif data_type == 'Mix':
            # Seed reset for reproducibility of the Bernoulli mask
            np.random.seed(seed)
            p_mix               = plan['proportion']
            mask                = np.random.rand(n_samples) < p_mix
            samples_1           = Weibull_Distribution(alpha=plan['alpha1'], beta=plan['beta1']).random_samples(n_samples)
            samples_2           = Weibull_Distribution(alpha=plan['alpha2'], beta=plan['beta2']).random_samples(n_samples)
            raw_data_unsorted   = np.where(mask, samples_1, samples_2)
            raw_data            = raw_data_unsorted.copy()

        else:
            warnings.warn(f"Unknown data_type '{data_type}' — skipping.", UserWarning)
            return None

        # ------------------------------------------------------------------
        # Type-I right-censoring
        # The censoring time equals the runtime of the last training failure.
        # All units beyond that time are censored at the SAME fixed value.
        # These censored units' TRUE runtimes form the holdout set.
        # ------------------------------------------------------------------
        raw_data    = np.sort(raw_data)
        limit_idx   = int(n_samples * (1 - censor_rate))
        censor_time = raw_data[limit_idx - 1]

        # --- Dataset-Level-Filter for CR ---
        if data_type == 'CR':
            # t1 and t2 were generated earlier (np.random.seed(seed) was set)
            # Sorting mapping: which of the n_samples units belong to train_failures?
            sort_idx = np.argsort(np.minimum(t1, t2))
            mech1_mask = (t1 < t2)[sort_idx[:limit_idx]]
            n_mech1 = int(np.sum(mech1_mask))
            n_mech2 = limit_idx - n_mech1

            # Each component must account for at least 10% of the failures or minimum 4 failures
            if n_mech1 < max(4, int(0.10 * limit_idx)) or n_mech2 < max(4, int(0.10 * limit_idx)):
                warnings.warn(f"CR seed={seed}: component split {n_mech1}/{n_mech2} — "
                              f"one component < 10% of {limit_idx} failures. Skipping.", UserWarning)
                return None

        # --- Dataset-Level-Filter for Mix ---
        elif data_type == 'Mix':
            # In Mix, component membership is determined by ‘mask’
            # mask was generated earlier: mask = np.random.rand(n_samples) < p_mix
            # Training mask: which of the sorted units belong to train_failures?
            sort_idx = np.argsort(raw_data_unsorted)  # before sorting → original indices
            train_mask = mask[sort_idx[:limit_idx]]
            n_comp1 = int(np.sum(train_mask))
            n_comp2 = limit_idx - n_comp1

            # Each component must account for at least 10% of the failures or minimum 4 failures
            if n_comp1 < max(4, int(0.10 * limit_idx)) or n_comp2 < max(4, int(0.10 * limit_idx)):
                warnings.warn(f"Mix seed={seed}: component split {n_comp1}/{n_comp2} — "
                              f"one component < 10% of {limit_idx} failures. Skipping.", UserWarning)
                return None

        dataset.update({
            'failures' : raw_data[:limit_idx],
            'censored' : np.full(n_samples - limit_idx, censor_time),
            'holdout'  : raw_data[limit_idx:],
            'raw_data' : raw_data,
        })

    except Exception as e:
        warnings.warn(f"Data generation failed for seed={seed}, type={data_type}: {e}", UserWarning)
        return None

    return dataset


#-----------------------------------------------------------------------------------------------------------------------
# MAIN FUNCTION: generate all datasets
#-----------------------------------------------------------------------------------------------------------------------
def generate_all_datasets(save_csv=SAVE_CSV, output_dir=CSV_OUTPUT_DIR, output_reliasoft_dir=CSV_OUTPUT_RELIASOFT_DIR):
    """
    Generates all synthetic datasets across all scenarios and seeds.

    Parameters
    ----------
    save_csv                : bool — if True, saves one CSV per scenario
    output_dir              : str  — output directory for CSV files
    output_reliasoft_dir    : str  — output directory for CSV files for reliasoft's formatting

    Returns
    -------
    all_datasets : list of dict
        Ready for direct use in Validate_Weibull_CI.py:
            for ds in datasets:
                failures  = ds['failures']
                censored  = ds['censored']
                true_alpha = ds['true_alpha_1']
    """
    test_plans = generate_test_plans()

    print(f"\nScenarios: {len(test_plans)} total")
    print(f"   2P:  {sum(1 for p in test_plans if p['data_type'] == '2P')}")
    print(f"   3P:  {sum(1 for p in test_plans if p['data_type'] == '3P')}")
    print(f"   CR:  {sum(1 for p in test_plans if p['data_type'] == 'CR')}")
    print(f"   Mix: {sum(1 for p in test_plans if p['data_type'] == 'Mix')}")

    all_datasets = []

    skipped_plans = 0

    # This can be edited depending on the desired amount of valid seed for each plan
    min_valid_seeds = N_LOOPS

    for plan in test_plans:
        data_type    = plan['data_type']
        seeds        = (plan['good_seeds'][:N_LOOPS]
                        if data_type == 'CR' and 'good_seeds' in plan
                        else range(N_LOOPS))
        plan_datasets = []

        for seed in seeds:
            ds = generate_single_dataset(seed, plan)
            if ds is not None:
                all_datasets.append(ds)
                plan_datasets.append(ds)

        if len(plan_datasets) < min_valid_seeds:
            if plan_datasets:   # at least on seed was OK
                warnings.warn(f"{plan['data_type']} plan (n={plan['n_samples']}, cr={plan['censor_rate']}) "
                              f"only {len(plan_datasets)}/{N_LOOPS} valid seeds — plan skipped.", UserWarning)
            plan_ids = {id(ds) for ds in plan_datasets}
            all_datasets[:] = [d for d in all_datasets if id(d) not in plan_ids]
            plan_datasets = []  # makes sure there is no .csv export

        if not plan_datasets:
            skipped_plans += 1

        # ------------------------------------------------------------------
        # Optional CSV export — one file per scenario
        # Arrays are stored as semicolon-joined strings for human readability.
        # Use load_datasets_from_csv() to reconstruct numpy arrays.
        # ------------------------------------------------------------------
        if save_csv and plan_datasets and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            rows = []
            for ds in plan_datasets:
                row = {k: v for k, v in ds.items() if not isinstance(v, np.ndarray)}
                for key in ['failures', 'censored', 'holdout', 'raw_data']:
                    arr = ds.get(key)
                    row[key] = ';'.join(f'{x:.6f}' for x in arr) if arr is not None else ''
                rows.append(row)

            a_str = str(plan.get('alpha', plan.get('alpha1', 'x'))).replace('.', '_')
            b_str = str(plan.get('beta', plan.get('beta1', 'x'))).replace('.', '_')
            n_str = str(plan['n_samples'])
            cr_str = str(plan['censor_rate']).replace('.', '_')

            fname = f"synth_{data_type}_a{a_str}_b{b_str}_n{n_str}_cr{cr_str}"

            if data_type == '3P':
                g_str = str(plan.get('gamma', 'x')).replace('.', '_')
                fname += f"_g{g_str}"

            if data_type in ('CR', 'Mix'):
                a2_str = str(plan.get('alpha2', 'x')).replace('.', '_')
                b2_str = str(plan.get('beta2', 'x')).replace('.', '_')
                fname += f"_a2{a2_str}_b2{b2_str}"

            if data_type == 'Mix':
                p_str = str(plan.get('proportion', 'x')).replace('.', '_')
                fname += f"_p{p_str}"

            if data_type == 'CR':
                mp_str = str(plan.get('min_pct', 'x')).replace('.', '_')
                fname += f"_mp{mp_str}"

            fname += ".csv"
            fpath = os.path.join(output_dir, fname)
            pd.DataFrame(rows).to_csv(fpath, index=False, sep=',', decimal='.')
            # print(f"  Saved: {fpath}")

        # .csv export for reliasoft's convention and more convenient import
        if save_csv and plan_datasets and output_reliasoft_dir:
            os.makedirs(output_reliasoft_dir, exist_ok=True)
            # --- Build filename (same logic as above) ---
            a_str = str(plan.get('alpha', plan.get('alpha1', 'x'))).replace('.', '_')
            b_str = str(plan.get('beta', plan.get('beta1', 'x'))).replace('.', '_')
            n_str = str(plan['n_samples'])
            cr_str = str(plan['censor_rate']).replace('.', '_')

            fname_rs = f"synth_{data_type}_a{a_str}_b{b_str}_n{n_str}_cr{cr_str}"

            if data_type == '3P':
                g_str = str(plan.get('gamma', 'x')).replace('.', '_')
                fname_rs += f"_g{g_str}"

            if data_type in ('CR', 'Mix'):
                a2_str = str(plan.get('alpha2', 'x')).replace('.', '_')
                b2_str = str(plan.get('beta2', 'x')).replace('.', '_')
                fname_rs += f"_a2{a2_str}_b2{b2_str}"

            if data_type == 'Mix':
                p_str = str(plan.get('proportion', 'x')).replace('.', '_')
                fname_rs += f"_p{p_str}"

            if data_type == 'CR':
                mp_str = str(plan.get('min_pct', 'x')).replace('.', '_')
                fname_rs += f"_mp{mp_str}"

            fname_rs += "_reliasoft.csv"
            fpath_rs = os.path.join(output_reliasoft_dir, fname_rs)

            # --- Collect true parameters as header comment lines ---
            param_lines = []
            param_lines.append(f"# data_type={data_type}")
            param_lines.append(f"# n_samples={plan['n_samples']}")
            param_lines.append(f"# censor_rate={plan['censor_rate']}")
            param_lines.append(f"# true_alpha_1={plan.get('alpha', plan.get('alpha1', 'NA'))}")
            param_lines.append(f"# true_beta_1={plan.get('beta', plan.get('beta1', 'NA'))}")
            if data_type == '3P':
                param_lines.append(f"# true_gamma={plan.get('gamma', 'NA')}")
            if data_type in ('CR', 'Mix'):
                param_lines.append(f"# true_alpha_2={plan.get('alpha2', 'NA')}")
                param_lines.append(f"# true_beta_2={plan.get('beta2', 'NA')}")
            if data_type == 'Mix':
                param_lines.append(f"# proportion={plan.get('proportion', 'NA')}")
            if data_type == 'CR':
                param_lines.append(f"# min_pct={plan.get('min_pct', 'NA')}")

            # --- Write file: param header + one F/S block per seed ---
            with open(fpath_rs, 'w', newline='') as f:
                for line in param_lines:
                    f.write(line + '\n')

                for ds_rs in plan_datasets:
                    f.write(f'\n# seed={ds_rs["seed"]}\n')
                    f.write('State,Time\n')
                    failures = ds_rs['failures'] if ds_rs['failures'] is not None else np.array([])
                    censored = ds_rs['censored'] if ds_rs['censored'] is not None else np.array([])
                    for t in failures:
                        f.write(f'F,{t:.6f}\n')
                    for t in censored:
                        f.write(f'S,{t:.6f}\n')

    print(f"\nScenario plans: {len(test_plans)} total")
    print(f"Plans with at least {min_valid_seeds} valid dataset: {len(test_plans) - skipped_plans}")
    print(f"Plans skipped (not enough valid seeds):  {skipped_plans}")

    print(f"\nTotal datasets generated: {len(all_datasets)}")
    return all_datasets


#-----------------------------------------------------------------------------------------------------------------------
# LOADER: reconstruct datasets from CSV
#-----------------------------------------------------------------------------------------------------------------------
def load_datasets_from_csv(csv_path, seed=None):
    """
    Reconstructs dataset dicts from a previously saved CSV.
    Semicolon-joined array strings are parsed back into numpy arrays.

    Use this in Validate_Weibull_CI.py when working CSV-based:
        from generate_synthetic_data import load_datasets_from_csv
        datasets = load_datasets_from_csv('synth_2P_a2_5_b0_52_n100_cr0_2.csv')
    """
    df       = pd.read_csv(csv_path, sep=',', decimal='.')
    datasets = []

    if seed is None:
        for _, row in df.iterrows():
            ds = row.to_dict()
            for key in ['failures', 'censored', 'holdout', 'raw_data']:
                val     = ds.get(key, '')
                ds[key] = (np.array([float(x) for x in val.split(';')]) if isinstance(val, str) and val else np.array([]))
            if 'censored' in ds:
                ds['suspensions'] = ds.pop('censored')
            datasets.append(ds)
    else:
        if seed >= 0 and seed % 1 == 0:
            filtered = df[df['seed'] == seed]
            for _, row in filtered.iterrows():
                ds = row.to_dict()
                for key in ['failures', 'censored', 'holdout', 'raw_data']:
                    val = ds.get(key, '')
                    ds[key] = (np.array([float(x) for x in val.split(';')]) if isinstance(val, str) and val else np.array([]))
                if 'censored' in ds:
                    ds['suspensions'] = ds.pop('censored')
                datasets.append(ds)
        else:
            warnings.warn(f"The seed is not specified correctly: seed = {seed}. None will be returned.", UserWarning)
            return None

    return datasets


#***********************************************************************************************************************
#-----------------------------------------------------------------------------------------------------------------------
# ENTRY POINT
#-----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    datasets_all = generate_all_datasets()
