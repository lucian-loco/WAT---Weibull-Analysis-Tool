#!/usr/bin/python3
import os
import gc
import re
import time
import psutil
import warnings
import threading
import autograd.numpy as anp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from types import SimpleNamespace
from matplotlib.ticker import LogLocator, FuncFormatter
from Synthetic_Data import load_datasets_from_csv
from weibull import make_minor_label_formatter
from reliability.Distributions import Weibull_Distribution
from reliability.Fitters import Fit_Weibull_2P, Fit_Weibull_3P, Fit_Weibull_Mixture, Fit_Weibull_CR
from weibull_ci import _weibull_cdf, _weibull_3p_cdf, _mixture_cdf, _cr_cdf
from weibull_ci import _compute_covariance, _sample_and_compute_bounds
from weibull_ci import weibull_2p_analytical_bounds, weibull_3p_analytical_bounds, weibull_mixture_analytical_bounds, weibull_cr_analytical_bounds
from weibull_ci import weibull_2p_bootstrap_bounds, weibull_3p_bootstrap_bounds, weibull_cr_bootstrap_bounds, weibull_mixture_bootstrap_bounds



#RAM guard to limit the usage
def ram_watchdog(threshold_ram: float = 1, threshold_swap: float = 0.995,
                 interval: float = 0.5, verbose: bool = True):
    def _watch():
        while True:
            ram_pct = psutil.virtual_memory().percent / 100.0
            swap_pct = psutil.swap_memory().percent / 100.0

            if ram_pct >= threshold_ram:
                if verbose:
                    print(f"[WATCHDOG] RAM {ram_pct*100:.1f}% — terminating.", flush=True)
                os._exit(1)

            if swap_pct >= threshold_swap:
                if verbose:
                    print(f"[WATCHDOG] Page File {swap_pct*100:.1f}% — terminating.", flush=True)
                os._exit(1)

            time.sleep(interval)

    threading.Thread(target=_watch, daemon=True).start()


ram_watchdog(interval=0.5)


def check_ram_guard(n_samples: int, n_xvals: int, dtype_bytes: int = 8,
                    safety_factor: float = 0.96, allow_swap: bool = True):
    estimated_bytes = n_samples * n_xvals * dtype_bytes * 3

    ram = psutil.virtual_memory()
    swap = psutil.swap_memory()

    ram_available = ram.available
    swap_available = swap.total - swap.used  # freier Page File

    if allow_swap:
        # Physischer RAM zählt voll, Page File mit Penalty (wegen Geschwindigkeit)
        effective_available = ram_available + swap_available * 0.95
    else:
        effective_available = ram_available

    if estimated_bytes > safety_factor * effective_available:
        est_gb = estimated_bytes / 1e9
        eff_gb = effective_available / 1e9
        raise MemoryError(
            f"RAM guard triggered: ~{est_gb:.2f} GB benötigt, "
            f"aber nur {eff_gb:.2f} GB effektiv verfügbar "
            f"(RAM: {ram_available/1e9:.1f} GB + Page File: {swap_available/1e9:.1f} GB × 0.9). "
            f"Reduziere n_samples (aktuell: {n_samples:,})."
        )


#***********************************************************************************************************************
# -----------------------------------------------------------------------------------------------------------------------
# Function to calculate the parametric Monte Carlo confidence bounds for Weibull 2P
# -----------------------------------------------------------------------------------------------------------------------
def weibull_2p_fisher_bounds(fit, xvals, failures, right_censored=None, CI=0.95, n_samples=10000, return_sf=False, seed=42):
    T_f  = np.asarray(failures)
    T_rc = np.asarray(right_censored) if right_censored is not None else np.array([])

    params = np.array([
        np.log(fit.alpha),
        np.log(fit.beta)
    ])

    def neg_loglik(p):
        return Fit_Weibull_2P.LL(anp.exp(p), T_f, T_rc)

    cov = _compute_covariance(neg_loglik, params)

    def cdf_fn_log(t, log_alpha, log_beta):
        return _weibull_cdf(t, np.exp(log_alpha), np.exp(log_beta))

    # RAM Check
    check_ram_guard(n_samples=n_samples, n_xvals=len(xvals))

    return _sample_and_compute_bounds(
        cdf_fn=cdf_fn_log,
        params=params,
        cov=cov,
        xvals=xvals,
        CI=CI,
        n_samples=n_samples,
        return_sf=return_sf,
        seed=seed
    )


def weibull_3p_fisher_bounds(fit, xvals, failures, right_censored=None, CI=0.95, n_samples=10000, return_sf=False, seed=42):
    T_f  = np.asarray(failures)
    T_rc = np.asarray(right_censored) if right_censored is not None else np.array([])

    params = np.array([
        np.log(fit.alpha),
        np.log(fit.beta),
        fit.gamma
    ])

    def neg_loglik(p):
        alpha = anp.exp(p[0])
        beta  = anp.exp(p[1])
        gamma = p[2]
        return Fit_Weibull_3P.LL(anp.array([alpha, beta, gamma]), T_f, T_rc)

    cov = _compute_covariance(neg_loglik, params)

    def cdf_fn_log_3p(t, log_alpha, log_beta, gamma):
        return _weibull_3p_cdf(t, np.exp(log_alpha), np.exp(log_beta), gamma)

    # RAM Check
    check_ram_guard(n_samples=n_samples, n_xvals=len(xvals))

    return _sample_and_compute_bounds(
        cdf_fn=cdf_fn_log_3p,
        params=params,
        cov=cov,
        xvals=xvals,
        CI=CI,
        n_samples=n_samples,
        return_sf=return_sf,
        seed=seed,
        min_failure=min(T_f)
    )


def weibull_mixture_fisher_bounds(fit, xvals, failures, right_censored=None, CI=0.95, n_samples=10000, return_sf=False, seed=42):
    T_f  = np.asarray(failures)
    T_rc = np.asarray(right_censored) if right_censored is not None else np.array([])

    params = np.array([
        np.log(fit.alpha_1),
        np.log(fit.beta_1),
        np.log(fit.alpha_2),
        np.log(fit.beta_2),
        fit.proportion_1
    ])

    def neg_loglik(p):
        p_orig = anp.array([
            anp.exp(p[0]),
            anp.exp(p[1]),
            anp.exp(p[2]),
            anp.exp(p[3]),
            p[4]
        ])
        return Fit_Weibull_Mixture.LL(p_orig, T_f, T_rc)

    cov = _compute_covariance(neg_loglik, params)

    def cdf_fn_log_mix(t, log_a1, log_b1, log_a2, log_b2, prop):
        return _mixture_cdf(t, np.exp(log_a1), np.exp(log_b1), np.exp(log_a2), np.exp(log_b2), prop)

    # RAM Check
    check_ram_guard(n_samples=n_samples, n_xvals=len(xvals))

    return _sample_and_compute_bounds(
        cdf_fn=cdf_fn_log_mix,
        params=params,
        cov=cov,
        xvals=xvals,
        CI=CI,
        n_samples=n_samples,
        return_sf=return_sf,
        seed=seed
    )


def weibull_cr_fisher_bounds(fit, xvals, failures, right_censored=None, CI=0.95, n_samples=10000, return_sf=False, seed=42):
    T_f  = np.asarray(failures)
    T_rc = np.asarray(right_censored) if right_censored is not None else np.array([])

    params = np.array([
        np.log(fit.alpha_1),
        np.log(fit.beta_1),
        np.log(fit.alpha_2),
        np.log(fit.beta_2)
    ])

    def neg_loglik(p):
        p_orig = anp.array([
            anp.exp(p[0]),
            anp.exp(p[1]),
            anp.exp(p[2]),
            anp.exp(p[3]),
        ])
        return Fit_Weibull_CR.LL(p_orig, T_f, T_rc)

    cov = _compute_covariance(neg_loglik, params)

    def cdf_fn_log_cr(t, log_a1, log_b1, log_a2, log_b2):
        return _cr_cdf(t, np.exp(log_a1), np.exp(log_b1), np.exp(log_a2), np.exp(log_b2))

    # RAM Check
    check_ram_guard(n_samples=n_samples, n_xvals=len(xvals))

    return _sample_and_compute_bounds(
        cdf_fn=cdf_fn_log_cr,
        params=params,
        cov=cov,
        xvals=xvals,
        CI=CI,
        n_samples=n_samples,
        return_sf=return_sf,
        seed=seed
    )


#***********************************************************************************************************************
#-----------------------------------------------------------------------------------------------------------------------
# Analytical approach to calculate the confidence bounds on reliability / failure probability
#-----------------------------------------------------------------------------------------------------------------------
"""
This approach is used directly from weibull_ci.py and imported at the top.
"""


#***********************************************************************************************************************
#-----------------------------------------------------------------------------------------------------------------------
# Non-parametric Bootstrap approach to calculate the confidence bounds on reliability / failure probability
#-----------------------------------------------------------------------------------------------------------------------
"""
This approach is used directly from weibull_ci.py and imported at the top.
"""


#***********************************************************************************************************************
#-----------------------------------------------------------------------------------------------------------------------
# Help functions
#-----------------------------------------------------------------------------------------------------------------------
def _calculate_xvals(x, spacing=0.1):
    """
    This function finds what the x limits of probability plots should be
    and returns these limits. This is similar to autoscaling, but the rules here
    are different to the matplotlib defaults.
    It is used extensively by the functions within the probability_plotting
    module to achieve the plotting style used within the reliability library.

    Parameters
    ----------
    x       : list, array
                Failures from the data
    spacing : float
                The spacing between the points and the edge of the plot. Default is 0.1
                for 10% spacing.

    Returns
    -------
    xmin, xmax  : float
    xvalues     : ndarray
    """
    # remove inf
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    min_x = min(x)
    max_x = max(x)

    # x limits
    min_x_log = np.log10(min_x)
    max_x_log = np.log10(max_x)
    dx_log = max_x_log - min_x_log
    xlim_lower = 10 ** (min_x_log - dx_log * spacing)
    xlim_upper = 10 ** (max_x_log + dx_log * spacing)

    if xlim_lower == xlim_upper:
        xlim_lower = 10 ** (np.log10(xlim_lower) - 10 * spacing)
        xlim_upper = 10 ** (np.log10(xlim_upper) + 10 * spacing)

    if xlim_lower < 0:
        xlim_lower = 0

    xmin_rel, xmax_rel = xlim_lower * 0.8, xlim_upper * 1.5

    xvalues = np.logspace(np.log10(xmin_rel), np.log10(xmax_rel), 1000)

    return xmin_rel, xmax_rel, xvalues


def _parse_proportion_from_filename(csv_name):
    """
    Extracts the proportion factor from the filename.
    Example: ‘synth_Mix_a1000_b1_5_n1000_cr0_7_a21000_b23_0_p0_2’ → 0.2
    Returns None if no p-token is found.
    """
    match = re.search(r'_p(\d+)_(\d+)(?:_|$)', csv_name)
    if match:
        return float(f"{match.group(1)}.{match.group(2)}")
    return None


def _load_and_validate_reliasoft(reliasoft_path, csv_name, data_type):
    """
    Liest die Reliasoft-Ergebnisdatei ein und gibt nur gültige Seeds zurück.
    Ungültig: F_lower oder F_upper über alle t-Werte konstant (Fisher-Matrix-Inversion fehlgeschlagen).

    Returns
    -------
    reliasoft_by_seed : dict  {seed_int: (lower_rs, upper_rs)}  — nur gültige Seeds
    valid_seeds       : set
    """
    if not os.path.exists(reliasoft_path):
        print(f'[Reliasoft] Keine Datei gefunden: {reliasoft_path} — Reliasoft-Vergleich wird übersprungen.')
        return {}, set()

    df_rs = pd.read_csv(reliasoft_path)
    df_rs['seed'] = df_rs['seed'].astype(int)

    reliasoft_by_seed = {}
    for seed, grp in df_rs.groupby('seed'):
        grp = grp.sort_values('t').reset_index(drop=True)
        lower_vals = grp['F_lower'].values
        upper_vals = grp['F_upper'].values

        if np.all(lower_vals == lower_vals[0]) or np.all(upper_vals == upper_vals[0]):
            print(f'[Reliasoft] {csv_name} seed {seed}: Bounds konstant → ungültig, übersprungen.')
            continue

        t_vals = grp['t'].values
        rs_params = {}
        for col in ['alpha', 'beta', 'gamma', 'alpha_1', 'beta_1', 'p_1', 'alpha_2', 'beta_2']:
            if col in grp.columns:
                rs_params[col] = grp[col].iloc[0]

        if data_type == '3P':
            t_vals_minus_gamma = grp['t-gamma'].values
            reliasoft_by_seed[seed] = (t_vals, lower_vals, upper_vals, t_vals_minus_gamma, rs_params)
        else:
            reliasoft_by_seed[seed] = (t_vals, lower_vals, upper_vals, rs_params)

    valid_seeds = set(reliasoft_by_seed.keys())

    if not valid_seeds:
        print(f'[Reliasoft] WARNUNG: {csv_name} hat keine gültigen Seeds!')

    return reliasoft_by_seed, valid_seeds


def _make_agg_rows(df, pct_cols, group_label_col, group_label_val, x_labels=('MEAN_ALL', 'MIN_ALL', 'MAX_ALL')):
    """Calculates MEAN/MIN/MAX for the pct_diff-columns of the given rows."""
    rows = []
    for label, fn in zip(x_labels, [np.nanmean, np.nanmin, np.nanmax]):
        row = {col: np.nan for col in df.columns}
        row['x'] = label
        row[group_label_col] = group_label_val
        for col in pct_cols:
            row[col] = fn(df[col].values.astype(float))
        rows.append(row)

    return pd.DataFrame(rows)


def _get_true_cdf(data, xvals):
    """
    Calculates the true CDF at xvals based on data_type and the
    true_* parameters in data. Returns None if parameters are missing.

    Supported data_type values: ‘2P’, ‘3P’, ‘Mixture’, ‘CR’
    """
    data_type = data.get('data_type')

    try:
        if data_type == '2P':
            alpha = float(data['true_alpha_1'])
            beta  = float(data['true_beta_1'])
            if np.isnan(alpha) or np.isnan(beta):
                return None
            return Weibull_Distribution(alpha=alpha, beta=beta).CDF(xvals, show_plot=False)

        elif data_type == '3P':
            alpha = float(data['true_alpha_1'])
            beta  = float(data['true_beta_1'])
            gamma = float(data['true_gamma'])
            if any(np.isnan(v) for v in [alpha, beta, gamma]):
                return None
            return Weibull_Distribution(alpha=alpha, beta=beta, gamma=gamma).CDF(xvals, show_plot=False)

        elif data_type == 'Mix':
            a1 = float(data['true_alpha_1']); b1 = float(data['true_beta_1'])
            a2 = float(data['true_alpha_2']); b2 = float(data['true_beta_2'])
            p1 = data.get('proportion_1')
            if p1 is None:
                return None
            p1 = float(p1)
            if any(np.isnan(v) for v in [a1, b1, a2, b2, p1]):
                return None
            cdf1 = Weibull_Distribution(alpha=a1, beta=b1).CDF(xvals, show_plot=False)
            cdf2 = Weibull_Distribution(alpha=a2, beta=b2).CDF(xvals, show_plot=False)
            return p1 * cdf1 + (1 - p1) * cdf2

        elif data_type == 'CR':
            a1 = float(data['true_alpha_1']); b1 = float(data['true_beta_1'])
            a2 = float(data['true_alpha_2']); b2 = float(data['true_beta_2'])
            if any(np.isnan(v) for v in [a1, b1, a2, b2]):
                return None
            # Competing Risk: 1 - S1(t)*S2(t)
            sf1 = Weibull_Distribution(alpha=a1, beta=b1).SF(xvals, show_plot=False)
            sf2 = Weibull_Distribution(alpha=a2, beta=b2).SF(xvals, show_plot=False)
            return 1.0 - sf1 * sf2

        else:
            return None

    except (KeyError, TypeError, ValueError):
        return None


def _compute_parameter_coverage(lower, upper, true_cdf_vals):
    """
    true_cdf_vals : ndarray — true CDF at xvals, precalculated by the caller
    """
    inside = (true_cdf_vals >= lower) & (true_cdf_vals <= upper)
    return float(np.mean(inside))   # → should be around 0.95


def _compute_predictive_coverage(lower, upper, xvals, holdout, true_cdf_vals):
    """
    true_cdf_vals : ndarray — true CDF at xvals, precomputed by the caller
    holdout       : array   — future defaults
    """
    holdout = np.asarray(holdout)
    holdout = holdout[(holdout >= xvals[0]) & (holdout <= xvals[-1])]
    if len(holdout) == 0:
        return np.nan

    # True CDF at holdout points obtained by interpolation from xvals
    true_cdf_holdout = np.interp(holdout, xvals, true_cdf_vals)
    lower_holdout = np.interp(holdout, xvals, lower)
    upper_holdout = np.interp(holdout, xvals, upper)

    inside = (true_cdf_holdout >= lower_holdout) & (true_cdf_holdout <= upper_holdout)
    return float(np.mean(inside))


def _compute_ci_bounds(data_type, fit, xvals, xvals_analytical, failures, right_censored, ci, n_samples, bootstrapping):
    """
    Calculates lower/upper for MC, analytical, and bootstrap
    depending on data_type. Returns a dictionary.
    """
    bounds = {}

    if data_type == '2P':
        bounds['mc'] = weibull_2p_fisher_bounds(fit=fit, xvals=xvals, failures=failures, right_censored=right_censored, CI=ci, n_samples=n_samples)
        bounds['analytical'] = weibull_2p_analytical_bounds(fit=fit, xvals=xvals, failures=failures, right_censored=right_censored, CI=ci)
        if bootstrapping:
            bounds['bootstrap'] = weibull_2p_bootstrap_bounds(xvals=xvals, failures=failures, right_censored=right_censored, CI=ci)

    elif data_type == '3P':
        bounds['mc'] = weibull_3p_fisher_bounds(fit=fit, xvals=xvals, failures=failures, right_censored=right_censored, CI=ci, n_samples=n_samples)
        bounds['analytical'] = weibull_3p_analytical_bounds(fit=fit, xvals=xvals_analytical, failures=failures, right_censored=right_censored, CI=ci)
        if bootstrapping:
            bounds['bootstrap'] = weibull_3p_bootstrap_bounds(xvals=xvals, failures=failures, right_censored=right_censored, CI=ci)

    elif data_type == 'Mix':
        bounds['mc'] = weibull_mixture_fisher_bounds(fit=fit, xvals=xvals, failures=failures, right_censored=right_censored, CI=ci, n_samples=n_samples)
        bounds['analytical'] = weibull_mixture_analytical_bounds(fit=fit, xvals=xvals, failures=failures, right_censored=right_censored, CI=ci)
        if bootstrapping:
            bounds['bootstrap'] = weibull_mixture_bootstrap_bounds(xvals=xvals, failures=failures, right_censored=right_censored, CI=ci)

    elif data_type == 'CR':
        bounds['mc'] = weibull_cr_fisher_bounds(fit=fit, xvals=xvals, failures=failures, right_censored=right_censored, CI=ci, n_samples=n_samples)
        bounds['analytical'] = weibull_cr_analytical_bounds(fit=fit, xvals=xvals, failures=failures, right_censored=right_censored, CI=ci)
        if bootstrapping:
            bounds['bootstrap'] = weibull_cr_bootstrap_bounds(xvals=xvals, failures=failures, right_censored=right_censored, CI=ci)

    else:
        raise ValueError(f'Unbekannter data_type: {data_type}')

    return bounds


#-----------------------------------------------------------------------------------------------------------------------
# Function to make the result analysis .csv and plots to validate the confidence bound methods
#-----------------------------------------------------------------------------------------------------------------------
def data_analysis(data, fit, ax_wb, ax_dev, params, n_samples, ci, results_reliasoft=None, fig=None, save_path=None,
                  bootstrapping=False, library_usage=False):
    """
    Builds the deviation DataFrame and draws the comparison plot on ax_dev.
    Also saves the figure if save_path is given.

    Parameters
    ----------
    data                : dict  — failures, suspensions, holdout, true_alpha_1, true_beta_1, data_type, ...
    fit                 : fitted Weibull object from reliability library
    ax_wb, ax_dev       : matplotlib Axes — Weibull plot axis and deviation plot axis
    params              : str   — label for this dataset/seed
    n_samples           : int   — Monte Carlo samples
    ci                  : float — confidence interval level (e.g. 0.95)
    results_reliasoft   : tuple (t_vals, lower_rs, upper_rs, t-gamma) or None, t-gamma only for 3P available
    fig                 : matplotlib Figure, optional — needed for savefig
    save_path           : str, optional
    bootstrapping       : bool
    library_usage       : bool

    Returns
    -------
    df : pd.DataFrame with all bounds, relative deviations, coverage metrics, and aggregation rows
    """
    rs_params = {}
    data_type = data.get('data_type')

    ax_wb.set_ylabel('Failure probability')
    ax_wb.set_ylim(0.001, 0.999)
    xmin_rel, xmax_rel, xvals_raw = _calculate_xvals(data['failures'])
    ax_wb.set_xlim(xmin_rel, xmax_rel)

    if data_type == '3P':
        gamma_hat = fit.gamma
        xvals = xvals_raw[xvals_raw > gamma_hat]
        xvals_shifted = xvals - gamma_hat
    else:
        xvals = xvals_raw
        xvals_shifted = xvals

    decade_span = np.log10(xmax_rel) - np.log10(xmin_rel)

    ax_wb.xaxis.set_major_locator(LogLocator(base=10.0, numticks=6))
    ax_wb.xaxis.set_minor_locator(LogLocator(base=10.0, subs='auto', numticks=10))
    ax_wb.xaxis.set_minor_formatter(FuncFormatter(make_minor_label_formatter(decade_span)))

    if data_type is None:
        raise ValueError(f'"data_type" is missing in the data dictionary for params={params}')

    ci_bounds = _compute_ci_bounds(data_type=data_type, fit=fit, xvals=xvals, xvals_analytical=xvals_shifted, failures=data['failures'], right_censored=data['suspensions'],
                                   ci=ci, n_samples=n_samples, bootstrapping=bootstrapping)

    # Calculation of the Confidence Interval with parametric Monte Carlo:-----------------------------------------------
    lower_mc, upper_mc = ci_bounds['mc']

    if lower_mc is not None and upper_mc is not None:
        ax_wb.fill_between(xvals_shifted, lower_mc, upper_mc, alpha=0.3, facecolor='none', edgecolor='red',
                           label=f"{int(ci * 100)}% Monte Carlo CI", hatch='///')
    # ------------------------------------------------------------------------------------------------------------------

    # Calculation of the Confidence Interval analytically:----------------------------------------------------------
    lower_analytic, upper_analytic, _, _ = ci_bounds['analytical']

    if lower_analytic is not None and upper_analytic is not None:
        ax_wb.fill_between(xvals_shifted, lower_analytic, upper_analytic, alpha=0.3, facecolor='none', edgecolor='green',
                               label=f"{int(ci * 100)}% analytical Fisher CI", hatch='\\\\')
    # ------------------------------------------------------------------------------------------------------------------

    # Calculation of the Confidence Interval with bootstrap:------------------------------------------------------------
    if bootstrapping:
        lower_bootstrap, upper_bootstrap = ci_bounds.get('bootstrap', (None, None))

        if lower_bootstrap is not None and upper_bootstrap is not None:
            ax_wb.fill_between(xvals_shifted, lower_bootstrap, upper_bootstrap, alpha=0.3, facecolor='none', edgecolor='magenta',
                               label=f"{int(ci * 100)}% Bootstrapping CI", hatch='oo')
    # ------------------------------------------------------------------------------------------------------------------

    # Reliasoft's Weibull++ confidence bounds:--------------------------------------------------------------------------
    if results_reliasoft is not None:
        if data_type == '3P':
            t_rs, lower_rs_raw, upper_rs_raw, t_rs_minus_gamma, rs_params = results_reliasoft
            lower_rs = np.interp(xvals_shifted, t_rs_minus_gamma, lower_rs_raw)
            upper_rs = np.interp(xvals_shifted, t_rs_minus_gamma, upper_rs_raw)
        else:
            t_rs, lower_rs_raw, upper_rs_raw, rs_params = results_reliasoft
            lower_rs = np.interp(xvals_shifted, t_rs, lower_rs_raw)
            upper_rs = np.interp(xvals_shifted, t_rs, upper_rs_raw)

        lower_rs = np.clip(lower_rs, 1e-9, 1 - 1e-9)
        upper_rs = np.clip(upper_rs, 1e-9, 1 - 1e-9)

        ax_wb.fill_between(xvals_shifted, lower_rs, upper_rs, alpha=0.3, facecolor='none', edgecolor='black',
                           label=f'{int(ci * 100)}% Reliasoft CI', hatch='xx')
    # ------------------------------------------------------------------------------------------------------------------

    ax_wb.legend(loc='upper left')

    if library_usage:
        try:
            lower_lib, point_lib, upper_lib = fit.distribution.CDF(CI_x=xvals, CI=ci, CI_type='reliability', show_plot=False)
        except Exception as e:
            warnings.warn(f'Library CDF failed for {params}: {e}')
            lower_lib, upper_lib = None, None
            library_usage = False

    # --- Reliasoft-Parameter basierter Fake-Fit ---
    fit_rs = None
    ci_bounds_rs_fit = {}

    if rs_params:
        if data_type == '2P':
            fit_rs = SimpleNamespace(alpha=rs_params['alpha'], beta=rs_params['beta'])
        elif data_type == '3P':
            fit_rs = SimpleNamespace(alpha=rs_params['alpha'], beta=rs_params['beta'], gamma=rs_params['gamma'])
        elif data_type == 'Mix':
            fit_rs = SimpleNamespace(alpha_1=rs_params['alpha_1'], beta_1=rs_params['beta_1'],
                                     alpha_2=rs_params['alpha_2'], beta_2=rs_params['beta_2'],
                                     proportion_1=rs_params['p_1'])
        elif data_type == 'CR':
            fit_rs = SimpleNamespace(alpha_1=rs_params['alpha_1'], beta_1=rs_params['beta_1'],
                                     alpha_2=rs_params['alpha_2'], beta_2=rs_params['beta_2'])

        if fit_rs is not None:
            # xvals für RS-Fit: basierend auf RS-gamma für 3P
            if data_type == '3P':
                gamma_rs = rs_params['gamma']
                xvals_rs = t_rs
                xvals_analytical_rs = t_rs_minus_gamma
            else:
                xvals_rs = t_rs
                xvals_analytical_rs = xvals_rs

            ci_bounds_rs_fit = _compute_ci_bounds(data_type=data_type,
                                                  fit=fit_rs,
                                                  xvals=xvals_rs,
                                                  xvals_analytical=xvals_analytical_rs,
                                                  failures=data['failures'],
                                                  right_censored=data.get('suspensions'),
                                                  ci=ci,
                                                  n_samples=n_samples,
                                                  bootstrapping=False)

    lower_mc_rsfit, upper_mc_rsfit = ci_bounds_rs_fit.get('mc', (None, None))
    lower_an_rsfit, upper_an_rsfit, *_ = ci_bounds_rs_fit.get('analytical', (None, None, None, None))

    # Skalare Aggregation RS-Fit vs. RS-Bounds (nur Algorithmus-Unterschied)
    rsfit_agg_rows = []
    if results_reliasoft is not None and lower_rs is not None and ci_bounds_rs_fit:
        lower_rs_clipped = np.clip(lower_rs_raw, 1e-9, 1 - 1e-9)
        upper_rs_clipped = np.clip(upper_rs_raw, 1e-9, 1 - 1e-9)

        comparisons = {}
        if lower_mc_rsfit is not None:
            comparisons['pct_diff_lower_mc_rsfit_vs_rs'] = (lower_mc_rsfit - lower_rs_clipped) / lower_rs_clipped * 100
            comparisons['pct_diff_upper_mc_rsfit_vs_rs'] = (upper_mc_rsfit - upper_rs_clipped) / upper_rs_clipped * 100
        if lower_an_rsfit is not None:
            comparisons['pct_diff_lower_an_rsfit_vs_rs'] = (lower_an_rsfit - lower_rs_clipped) / lower_rs_clipped * 100
            comparisons['pct_diff_upper_an_rsfit_vs_rs'] = (upper_an_rsfit - upper_rs_clipped) / upper_rs_clipped * 100

        for label, fn in zip(
                ['MEANxvals_rsfit', 'MINxvals_rsfit', 'MAXxvals_rsfit'],
                [np.nanmean, np.nanmin, np.nanmax]
        ):
            row = {'params': params, 'x': label}
            for col, vals in comparisons.items():
                row[col] = fn(vals)
            rsfit_agg_rows.append(row)

    # DataFrame for the analytics
    df_dict = {'params': params,
               'sample_size': n_samples,
               'x': xvals_shifted,
               'lower_mc': lower_mc,
               'upper_mc': upper_mc,
               'lower_analytical': lower_analytic,
               'upper_analytical': upper_analytic}

    if bootstrapping:
        if lower_bootstrap is not None and upper_bootstrap is not None:
            df_dict['lower_bootstrap'] = lower_bootstrap
            df_dict['upper_bootstrap'] = upper_bootstrap

    if library_usage:
        if lower_lib is not None and upper_lib is not None:
            df_dict['lower_lib'] = lower_lib
            df_dict['upper_lib'] = upper_lib

    if results_reliasoft is not None:
        if lower_rs is not None and upper_rs is not None:
            df_dict['lower_rs'] = lower_rs
            df_dict['upper_rs'] = upper_rs

    if library_usage:
        if lower_lib is not None and upper_lib is not None:
            if lower_mc is not None and upper_mc is not None:
                df_dict['pct_diff_lower_mc_vs_lib'] = (lower_mc - lower_lib) / np.clip(lower_lib, 1e-9, None) * 100
                df_dict['pct_diff_upper_mc_vs_lib'] = (upper_mc - upper_lib) / np.clip(upper_lib, 1e-9, None) * 100
            if lower_analytic is not None and upper_analytic is not None:
                df_dict['pct_diff_lower_analytical_vs_lib'] = (lower_analytic - lower_lib) / np.clip(lower_lib, 1e-9, None) * 100
                df_dict['pct_diff_upper_analytical_vs_lib'] = (upper_analytic - upper_lib) / np.clip(upper_lib, 1e-9, None) * 100

    if all(v is not None for v in [lower_analytic, upper_analytic, lower_mc, upper_mc]):
        df_dict['pct_diff_lower_mc_vs_analytic'] = (lower_mc - lower_analytic) / np.clip(lower_analytic, 1e-9, None) * 100
        df_dict['pct_diff_upper_mc_vs_analytic'] = (upper_mc - upper_analytic) / np.clip(upper_analytic, 1e-9, None) * 100

    if results_reliasoft is not None:
        if lower_rs is not None and upper_rs is not None:
            if lower_mc is not None and upper_mc is not None:
                df_dict['pct_diff_lower_mc_vs_rs'] = (lower_mc - lower_rs) / np.clip(lower_rs, 1e-9, None) * 100
                df_dict['pct_diff_upper_mc_vs_rs'] = (upper_mc - upper_rs) / np.clip(upper_rs, 1e-9, None) * 100
            if lower_analytic is not None and upper_analytic is not None:
                df_dict['pct_diff_lower_analytical_vs_rs'] = (lower_analytic - lower_rs) / np.clip(lower_rs, 1e-9, None) * 100
                df_dict['pct_diff_upper_analytical_vs_rs'] = (upper_analytic - upper_rs) / np.clip(upper_rs, 1e-9, None) * 100
            if library_usage:
                if lower_lib is not None and upper_lib is not None:
                    df_dict['pct_diff_lower_lib_vs_rs'] = (lower_lib - lower_rs) / np.clip(lower_rs, 1e-9, None) * 100
                    df_dict['pct_diff_upper_lib_vs_rs'] = (upper_lib - upper_rs) / np.clip(upper_rs, 1e-9, None) * 100

    if bootstrapping:
        if lower_bootstrap is not None and upper_bootstrap is not None:
            if library_usage:
                if lower_lib is not None and upper_lib is not None:
                    df_dict['pct_diff_lower_bootstrap_vs_lib'] = (lower_bootstrap - lower_lib) / np.clip(lower_lib, 1e-9, None) * 100
                    df_dict['pct_diff_upper_bootstrap_vs_lib'] = (upper_bootstrap - upper_lib) / np.clip(upper_lib, 1e-9, None) * 100
            if lower_mc is not None and upper_mc is not None:
                df_dict['pct_diff_lower_mc_vs_bootstrap'] = (lower_mc - lower_bootstrap) / np.clip(lower_bootstrap, 1e-9, None) * 100
                df_dict['pct_diff_upper_mc_vs_bootstrap'] = (upper_mc - upper_bootstrap) / np.clip(upper_bootstrap, 1e-9, None) * 100
            if lower_analytic is not None and upper_analytic is not None:
                df_dict['pct_diff_lower_analytical_vs_bootstrap'] = (lower_analytic - lower_bootstrap) / np.clip(lower_bootstrap, 1e-9, None) * 100
                df_dict['pct_diff_upper_analytical_vs_bootstrap'] = (upper_analytic - upper_bootstrap) / np.clip(upper_bootstrap, 1e-9, None) * 100
            if results_reliasoft is not None:
                if lower_rs is not None and upper_rs is not None:
                    df_dict['pct_diff_lower_rs_vs_bootstrap'] = (lower_rs - lower_bootstrap) / np.clip(lower_bootstrap, 1e-9, None) * 100
                    df_dict['pct_diff_upper_rs_vs_bootstrap'] = (upper_rs - upper_bootstrap) / np.clip(upper_bootstrap, 1e-9, None) * 100

    df = pd.DataFrame(df_dict)

    # Comparison plot
    if library_usage:
        if lower_lib is not None and upper_lib is not None:
            if lower_mc is not None and upper_mc is not None:
                ax_dev.plot(xvals_shifted, df['pct_diff_lower_mc_vs_lib'], color='blue', linestyle='solid', label='lower MC vs. lib')
                ax_dev.plot(xvals_shifted, df['pct_diff_upper_mc_vs_lib'], color='cyan', linestyle='solid', label='upper MC vs. lib')
            if lower_analytic is not None and upper_analytic is not None:
                ax_dev.plot(xvals_shifted, df['pct_diff_lower_analytical_vs_lib'], color='green', linestyle='solid', label='lower analytical vs. lib')
                ax_dev.plot(xvals_shifted, df['pct_diff_upper_analytical_vs_lib'], color='lime', linestyle='dashed', label='upper analytical vs. lib')

    if all(v is not None for v in [lower_analytic, upper_analytic, lower_mc, upper_mc]):
        ax_dev.plot(xvals_shifted, df['pct_diff_lower_mc_vs_analytic'], color='red', linestyle='dashdot', label='lower MC vs. analytical')
        ax_dev.plot(xvals_shifted, df['pct_diff_upper_mc_vs_analytic'], color='deeppink', linestyle='dashdot', label='upper MC vs. analytical')

    if results_reliasoft is not None:
        if lower_rs is not None and upper_rs is not None:
            if lower_mc is not None and upper_mc is not None:
                ax_dev.plot(xvals_shifted, df['pct_diff_lower_mc_vs_rs'], color='navy', linestyle='solid', label='lower MC vs. Reliasoft')
                ax_dev.plot(xvals_shifted, df['pct_diff_upper_mc_vs_rs'], color='steelblue', linestyle='solid', label='upper MC vs. Reliasoft')
            if lower_analytic is not None and upper_analytic is not None:
                ax_dev.plot(xvals_shifted, df['pct_diff_lower_analytical_vs_rs'], color='darkgreen', linestyle='dashed', label='lower analytical vs. Reliasoft')
                ax_dev.plot(xvals_shifted, df['pct_diff_upper_analytical_vs_rs'], color='limegreen', linestyle='dashed', label='upper analytical vs. Reliasoft')
            if library_usage:
                if lower_lib is not None and upper_lib is not None:
                    ax_dev.plot(xvals_shifted, df['pct_diff_lower_lib_vs_rs'], color='gray', linestyle='dotted', label='lower lib vs. Reliasoft')
                    ax_dev.plot(xvals_shifted, df['pct_diff_upper_lib_vs_rs'], color='silver', linestyle='dotted', label='upper lib vs. Reliasoft')

    if bootstrapping:
        if lower_bootstrap is not None and upper_bootstrap is not None:
            if library_usage:
                if lower_lib is not None and upper_lib is not None:
                    ax_dev.plot(xvals_shifted, df['pct_diff_lower_bootstrap_vs_lib'], color='darkorange', linestyle='solid', label='lower Bootstrap vs. lib')
                    ax_dev.plot(xvals_shifted, df['pct_diff_upper_bootstrap_vs_lib'], color='gold', linestyle='dashed', label='upper Bootstrap vs. lib')
            if lower_mc is not None and upper_mc is not None:
                ax_dev.plot(xvals_shifted, df['pct_diff_lower_mc_vs_bootstrap'], color='darkviolet', linestyle='dashdot', label='lower MC vs. Bootstrap')
                ax_dev.plot(xvals_shifted, df['pct_diff_upper_mc_vs_bootstrap'], color='violet', linestyle='dotted', label='upper MC vs. Bootstrap')
            if lower_analytic is not None and upper_analytic is not None:
                ax_dev.plot(xvals_shifted, df['pct_diff_lower_analytical_vs_bootstrap'], color='indigo', linestyle='dashdot', label='lower analytical vs. Bootstrap')
                ax_dev.plot(xvals_shifted, df['pct_diff_upper_analytical_vs_bootstrap'], color='mediumpurple', linestyle='dotted', label='upper analytical vs. Bootstrap')
            if results_reliasoft is not None:
                if lower_rs is not None and upper_rs is not None:
                    ax_dev.plot(xvals_shifted, df['pct_diff_lower_rs_vs_bootstrap'], color='chocolate', linestyle='dashdot', label='lower Reliasoft vs. Bootstrap')
                    ax_dev.plot(xvals_shifted, df['pct_diff_upper_rs_vs_bootstrap'], color='peru', linestyle='dotted', label='upper Reliasoft vs. Bootstrap')

    ax_dev.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax_dev.set_xscale('log')
    ax_dev.set_xlabel('Time in days')
    ax_dev.set_ylabel('Relative difference [%]')
    ax_dev.set_title(f'Relative deviation for {params} ({n_samples} samples): (method - reference) / reference × 100')
    ax_dev.legend()
    ax_dev.grid(True, which='both', linestyle='--', alpha=0.5)

    if save_path and fig is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

    # Coverage metrics:
    true_cdf_vals = _get_true_cdf(data, xvals)
    holdout = data.get('holdout')

    cov_row = {'params': params, 'sample_size': n_samples, 'x': 'COVERAGE'}

    if true_cdf_vals is not None:
        methods_for_coverage = {}
        valid_analytic = None

        if lower_mc is not None and upper_mc is not None:
            methods_for_coverage['mc'] = (lower_mc, upper_mc)
        if lower_analytic is not None and upper_analytic is not None:
            valid_analytic = (lower_analytic >= 0.001) & (lower_analytic <= 0.999)
            methods_for_coverage['analytical'] = (lower_analytic[valid_analytic], upper_analytic[valid_analytic])
        if library_usage and lower_lib is not None and upper_lib is not None:
            methods_for_coverage['lib'] = (lower_lib, upper_lib)
        if results_reliasoft is not None and lower_rs is not None and upper_rs is not None:
            methods_for_coverage['rs'] = (lower_rs, upper_rs)
        if bootstrapping and lower_bootstrap is not None and upper_bootstrap is not None:
            methods_for_coverage['bootstrap'] = (lower_bootstrap, upper_bootstrap)

        for method_name, (lower, upper) in methods_for_coverage.items():
            if method_name == 'analytical' and valid_analytic is not None:
                true_cdf_cov = true_cdf_vals[valid_analytic]
                xvals_cov = xvals[valid_analytic]
            else:
                true_cdf_cov = true_cdf_vals
                xvals_cov = xvals
            cov_row[f'param_coverage_{method_name}'] = _compute_parameter_coverage(lower, upper, true_cdf_cov)
            if holdout is not None and len(holdout) > 0:
                cov_row[f'predictive_coverage_{method_name}'] = _compute_predictive_coverage(lower, upper, xvals_cov, holdout, true_cdf_cov)

    df = pd.concat([df, pd.DataFrame([cov_row])], ignore_index=True)

    # Aggregation pct_diff for xvals
    pct_cols = [c for c in df.columns if c.startswith('pct_diff_')]
    xval_only = df[~df['x'].astype(str).str.startswith(('MEAN', 'MIN', 'MAX', 'COVERAGE'))]

    if lower_analytic is not None:
        valid_lower_cdf = (lower_analytic >= 0.001) & (lower_analytic <= 0.999)
        xval_only = xval_only[valid_lower_cdf]

    agg_rows = _make_agg_rows(
        df=xval_only,
        pct_cols=pct_cols,
        group_label_col='params',
        group_label_val=params,
        x_labels=('MEAN_xvals', 'MIN_xvals', 'MAX_xvals')
    )
    df = pd.concat([df, agg_rows], ignore_index=True)

    # Vergleich nur im Bereich der failures
    failures_arr = np.asarray(data['failures'])
    failure_min = float(np.min(failures_arr))
    failure_max = float(np.max(failures_arr))

    if data_type == '3P':
        gamma_hat = fit.gamma
        failure_min = failure_min - gamma_hat
        failure_max = failure_max - gamma_hat

    # xval_only ist bereits gefiltert (valide analytical bounds), kommt aus dem obigen Block
    xval_range = xval_only[(xval_only['x'].astype(float) >= failure_min) & (xval_only['x'].astype(float) <= failure_max)]

    if not xval_range.empty:
        agg_rows_range = _make_agg_rows(
            df=xval_range,
            pct_cols=pct_cols,
            group_label_col='params',
            group_label_val=params,
            x_labels=('MEAN_xvals_range', 'MIN_xvals_range', 'MAX_xvals_range')
        )
        df = pd.concat([df, agg_rows_range], ignore_index=True)

    if rsfit_agg_rows:
        df = pd.concat([df, pd.DataFrame(rsfit_agg_rows)], ignore_index=True)

    return df


#-----------------------------------------------------------------------------------------------------------------------
# Function for Weibull 2P, 3P, Mixture and Competing Risk with additional parametric Monte Carlo / analytical Delta Method / non-parametric bootstrap
# confidence intervals to validate own implementation
#-----------------------------------------------------------------------------------------------------------------------
'''
This script compares and validates the algorithm in "weibull_ci.py" to generate the confidence intervals with 
parametric Monte Carlo, analytical Delta method or non-parametric bootstrapping.
The methods are compared with the analytical method implemented in the reliability library by MatthewReid854 and 
with the results of the Weibull++ API for the same synthetic data set.
This is done for the Weibull 2P and Weibull 3P with synthetic data previously generated with Synthetic_Data.py.
For Weibull Mixture and Weibull Competing Risk the comparison is only done between the own methods and the results of the Weibull++ API.
'''
_FIT_CLASSES = {'2P':  Fit_Weibull_2P,
                '3P':  Fit_Weibull_3P,
                'Mix': Fit_Weibull_Mixture,
                'CR':  Fit_Weibull_CR}

_FIT_LABELS = {'2P':  'Weibull 2 Parameter fit | MLE',
               '3P':  'Weibull 3 Parameter fit | MLE',
               'Mix': 'Weibull Mixture fit | MLE',
               'CR':  'Weibull Competing Risk fit | MLE'}

def _build_title(data_type, csv_name, seed, n_samples, wb, ci):
    """Baut den Plot-Titel je nach data_type."""
    base = f'Weibull Probability Plot for [{csv_name} | seed {seed}]\n {n_samples} samples'
    if data_type == '2P':
        return f'{base} (α={wb.alpha:.4f}, β={wb.beta:.4f}, CI={ci:.3f})'
    elif data_type == '3P':
        return f'{base} (α={wb.alpha:.4f}, β={wb.beta:.4f}, γ={wb.gamma:.4f}, CI={ci:.3f})'
    elif data_type == 'Mix':
        return (f'{base} (α1={wb.alpha_1:.4f}, β1={wb.beta_1:.4f}, '
                f'α2={wb.alpha_2:.4f}, β2={wb.beta_2:.4f}, p={wb.proportion_1:.4f}, CI={ci:.3f})')
    elif data_type == 'CR':
        return (f'{base} (α1={wb.alpha_1:.4f}, β1={wb.beta_1:.4f}, '
                f'α2={wb.alpha_2:.4f}, β2={wb.beta_2:.4f}, CI={ci:.3f})')
    return base


def weibull_validate(data, seed, csv_name, ci=0.95, n_samples=10000, save_path=None, reliasoft_bounds=None, bootstrapping=False, library_usage=False):
    if not data:
        raise RuntimeError(f'No data loaded from {csv_name} for seed={seed}')

    data_type = data.get('data_type')
    if data_type not in _FIT_CLASSES:
        raise ValueError(f'Unbekannter data_type: {data_type}')

    if data_type in ('Mix', 'CR'):
        library_usage = False

    failure_size = len(data['failures'])
    suspension_size = len(data['suspensions']) if data.get('suspensions') is not None else 0
    sample_size = failure_size + suspension_size

    # Prevent zeros in the right censored data
    if data.get('suspensions') is not None and any(t == 0 for t in data['suspensions']):
        data['suspensions'] = [t for t in data['suspensions'] if t > 0]
        with warnings.catch_warnings():
            warnings.simplefilter('always', UserWarning)
            warnings.warn(f'The suspension data contained zeros as running_time. Removed in {csv_name}, seed {seed}.', UserWarning)

    if data.get('suspensions') is None or len(data['suspensions']) == 0:
        data['suspensions'] = None

    fig = plt.figure(figsize=(10, 12))
    gs = fig.add_gridspec(2, 1, height_ratios=[2, 1], hspace=0.4)

    ax_wb = fig.add_subplot(gs[0])
    ax_dev = fig.add_subplot(gs[1])

    plt.sca(ax_wb)

    FitClass = _FIT_CLASSES[data_type]
    label = f'{_FIT_LABELS[data_type]}\n (n = {sample_size} (f: {failure_size} | s: {suspension_size})'

    # Mixture und CR brauchen keinen CI_type/CI Parameter im Fit
    try:
        if data_type in ('2P', '3P'):
            wb = FitClass(failures=data['failures'], right_censored=data['suspensions'],
                          show_probability_plot=True, print_results=False,
                          method='MLE', optimizer='best',
                          CI_type='reliability', CI=ci, label=label)
        else:  # Mix, CR
            wb = FitClass(failures=data['failures'], right_censored=data['suspensions'],
                          show_probability_plot=True, print_results=False,
                          optimizer='best',
                          CI=ci, label=label)
    except Exception as e:
        raise RuntimeError(f'Weibull {data_type} fit failed for "{csv_name}" and seed {seed}: {e}')

    if data_type == '3P':
        if wb.gamma <= 0.1:
            warnings.warn(f'Seed {seed} in {csv_name}: gamma={wb.gamma:.6f} <= 0 — seed wird übersprungen.', UserWarning)
            plt.close(fig)
            return None, None

    ax_wb.set_title(_build_title(data_type, csv_name, seed, n_samples, wb, ci))

    if data_type == '3P':
        ax_wb.set_xlabel(f'Time in days minus failure free time γ={wb.gamma:.4f}')
    else:
        ax_wb.set_xlabel('Time in days')

    df = data_analysis(data=data, fit=wb, ax_wb=ax_wb, ax_dev=ax_dev, params=f'{csv_name}_seed{seed}', n_samples=n_samples, ci=ci, results_reliasoft=reliasoft_bounds,
                       bootstrapping=bootstrapping, library_usage=library_usage, fig=fig, save_path=save_path)

    df.insert(0, 'csv_name', csv_name)
    df.insert(1, 'seed', seed)

    return wb.results, df


#-----------------------------------------------------------------------------------------------------------------------
# Main part of the code for validation and some analysis
#-----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    # To be defined
    csv_dir = r"C:\...\Synthetic-Data"
    out_dir = r"C:\...\Validate_CI"
    out_dir_detail = r"C:\...\Validate_CI\Detailed_Results\csv_files"
    out_dir_png = r"C:\...\Validate_CI\Detailed_Results\plots"
    reliasoft_dir = r"C:\...\Validate_CI\Reliasoft_Results"

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_dir_detail, exist_ok=True)
    os.makedirs(out_dir_png, exist_ok=True)

    n_samples = 10000
    ci = 0.95

    # Konfiguration je Typ: (Dateiprefix, data_type, library_usage, summary_label)
    TYPE_CONFIG = [('synth_2P', '2P', True, 'ALL_2P'),
                   ('synth_3P', '3P', True, 'ALL_3P'),
                   ('synth_Mix', 'Mix', False, 'ALL_Mix'),
                   ('synth_CR', 'CR', False, 'ALL_CR')
                   ]

    for prefix, data_type, use_library, summary_label in TYPE_CONFIG:
        csv_files = sorted([f for f in os.listdir(csv_dir)
                            if f.startswith(prefix) and f.endswith('.csv')])
        if not csv_files:
            print(f'Keine CSV-Dateien für {prefix} gefunden — übersprungen.')
            continue

        all_param_agg_rows = []

        for csv_file in csv_files:
            csv_path = os.path.join(csv_dir, csv_file)
            csv_name = os.path.splitext(csv_file)[0]
            reliasoft_path = os.path.join(reliasoft_dir, f'{csv_name}_reliasoft.csv')

            print(f'Processing [{data_type}] {csv_name} ...', flush=True)

            reliasoft_by_seed, valid_seeds = _load_and_validate_reliasoft(reliasoft_path=reliasoft_path, csv_name=csv_name,
                                                                          data_type=data_type)

            datasets = load_datasets_from_csv(csv_path)
            if not datasets:
                warnings.warn(f'No datasets in {csv_file} – skipped.', UserWarning)
                continue

            # Mixture: proportion aus Dateiname laden
            if data_type == 'Mix':
                proportion = _parse_proportion_from_filename(csv_name)
                for ds in datasets:
                    ds['proportion_1'] = proportion

            datasets = [ds for ds in datasets if ds['seed'] in valid_seeds]
            if not datasets:
                print(f'[WARNUNG] {csv_name}: Keine gemeinsamen gültigen Seeds — übersprungen.')
                continue

            seed_dfs = []

            for ds in datasets:
                seed = ds['seed']
                png_path = os.path.join(out_dir_png, f'{csv_name}_seed{seed}.png')

                try:
                    _, df_seed = weibull_validate(data=ds, seed=seed, csv_name=csv_name, ci=ci, n_samples=n_samples,
                                                  save_path=png_path, reliasoft_bounds=reliasoft_by_seed.get(seed),
                                                  bootstrapping=False, library_usage=use_library)
                    if df_seed is None:
                        print(f'Seed {seed} skipped (gamma <= 0.1)')
                        continue
                    seed_dfs.append(df_seed)
                except Exception as e:
                    warnings.warn(f'Failed: {csv_name} seed {seed}: {e}', UserWarning)
                    continue

            if not seed_dfs:
                continue

            df_param = pd.concat(seed_dfs, ignore_index=True)

            pct_cols = [c for c in df_param.columns if c.startswith('pct_diff_')]
            coverage_cols = [c for c in df_param.columns if c.startswith('param_coverage_') or c.startswith('predictive_coverage_')]

            xval_rows = df_param[~df_param['x'].astype(str).str.startswith(('MEAN', 'MIN', 'MAX', 'COVERAGE'))]

            if 'lower_analytical' in xval_rows.columns:
                valid = (xval_rows['lower_analytical'] >= 0.001) & (xval_rows['lower_analytical'] <= 0.999)
                xval_rows = xval_rows[valid]

            cov_rows = df_param[df_param['x'] == 'COVERAGE']

            if not xval_rows.empty:
                agg_pct = _make_agg_rows(df=xval_rows, pct_cols=pct_cols,
                                         group_label_col='csv_name', group_label_val=csv_name,
                                         x_labels=('MEAN_seeds', 'MIN_seeds', 'MAX_seeds'))
            else:
                agg_pct = pd.DataFrame()

            agg_cov = _make_agg_rows(df=cov_rows, pct_cols=coverage_cols,
                                     group_label_col='csv_name', group_label_val=csv_name,
                                     x_labels=('MEAN_seeds_coverage', 'MIN_seeds_coverage', 'MAX_seeds_coverage'))

            xval_range_rows = df_param[df_param['x'] == 'MEAN_xvals_range']
            if not xval_range_rows.empty:
                agg_pct_range = _make_agg_rows(df=xval_range_rows, pct_cols=pct_cols,
                                               group_label_col='csv_name', group_label_val=csv_name,
                                               x_labels=('MEAN_seeds_range', 'MIN_seeds_range', 'MAX_seeds_range'))
            else:
                agg_pct_range = pd.DataFrame()

            # Aggregation rsfit über Seeds
            rsfit_pct_cols = [c for c in df_param.columns if 'rsfit' in c and c.startswith('pct_diff')]
            rsfit_rows = df_param[df_param['x'].astype(str) == 'MEANxvals_rsfit']
            if not rsfit_rows.empty and rsfit_pct_cols:
                agg_rsfit = _make_agg_rows(
                    df=rsfit_rows,
                    pct_cols=rsfit_pct_cols,
                    group_label_col='csv_name',
                    group_label_val=csv_name,
                    x_labels=['MEANseeds_rsfit', 'MINseeds_rsfit', 'MAXseeds_rsfit']
                )
            else:
                agg_rsfit = pd.DataFrame()

            agg_seed_rows = pd.concat([agg_pct, agg_cov, agg_pct_range, agg_rsfit], ignore_index=True)
            agg_seed_rows['csv_name'] = csv_name
            agg_seed_rows['seed'] = 'ALL'

            df_param = pd.concat([df_param, agg_seed_rows], ignore_index=True)
            df_param = df_param.sort_values(['seed', 'x']).reset_index(drop=True)

            param_csv = os.path.join(out_dir_detail, f'{csv_name}_results.csv')
            df_param.to_csv(param_csv, index=False, float_format='%.4f')
            print(f'  → Saved: {param_csv}')

            all_param_agg_rows.append(agg_seed_rows.copy())
            del df_param, seed_dfs
            gc.collect()

        # Globale Aggregation pro Typ
        if all_param_agg_rows:
            df_all_agg = pd.concat(all_param_agg_rows, ignore_index=True)

            mean_pct = df_all_agg[df_all_agg['x'] == 'MEAN_seeds']
            pct_cols = [c for c in mean_pct.columns if c.startswith('pct_diff_')]
            global_pct = _make_agg_rows(df=mean_pct, pct_cols=pct_cols,
                                        group_label_col='csv_name', group_label_val=summary_label,
                                        x_labels=('MEAN_global', 'MIN_global', 'MAX_global'))

            mean_cov = df_all_agg[df_all_agg['x'] == 'MEAN_seeds_coverage']
            coverage_cols = [c for c in mean_cov.columns if c.startswith('param_coverage_')
                             or c.startswith('predictive_coverage_')]
            global_cov = _make_agg_rows(df=mean_cov, pct_cols=coverage_cols,
                                        group_label_col='csv_name', group_label_val=summary_label,
                                        x_labels=('MEAN_global_coverage', 'MIN_global_coverage', 'MAX_global_coverage'))

            mean_range = df_all_agg[df_all_agg['x'] == 'MEAN_seeds_range']
            if not mean_range.empty:
                pct_cols_range = [c for c in mean_range.columns if c.startswith('pct_diff_')]
                global_range = _make_agg_rows(df=mean_range, pct_cols=pct_cols_range,
                                              group_label_col='csv_name', group_label_val=summary_label,
                                              x_labels=('MEAN_global_range', 'MIN_global_range', 'MAX_global_range'))
            else:
                global_range = pd.DataFrame()

            mean_rsfit = df_all_agg[df_all_agg['x'] == 'MEANseeds_rsfit']
            rsfit_pct_cols_global = [c for c in mean_rsfit.columns if 'rsfit' in c and c.startswith('pct_diff')]
            if not mean_rsfit.empty and rsfit_pct_cols_global:
                global_rsfit = _make_agg_rows(
                    df=mean_rsfit,
                    pct_cols=rsfit_pct_cols_global,
                    group_label_col='csv_name',
                    group_label_val=summary_label,
                    x_labels=['MEANglobal_rsfit', 'MINglobal_rsfit', 'MAXglobal_rsfit']
                )
            else:
                global_rsfit = pd.DataFrame()

            global_agg = pd.concat([global_pct, global_cov, global_range, global_rsfit], ignore_index=True)
            global_agg['csv_name'] = summary_label
            global_agg['seed'] = 'ALL'

            df_summary = pd.concat([df_all_agg, global_agg], ignore_index=True)
            summary_csv = os.path.join(out_dir, f'Weibull{data_type}_summary_all_params.csv')
            df_summary.to_csv(summary_csv, index=False, float_format='%.4f')
            print(f'\nGlobal summary saved: {summary_csv}')

