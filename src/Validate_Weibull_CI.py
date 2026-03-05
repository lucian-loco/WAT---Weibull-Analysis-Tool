#!/usr/bin/python3
import os
import autograd
import autograd.numpy as anp
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from weibull import plot_settings
from data_weibull import get_data
from reliability.Fitters import Fit_Weibull_2P, Fit_Weibull_CR
from weibull_ci import _weibull_cdf
# from weibull_ci import _compute_covariance
# from weibull_ci import _sample_and_compute_bounds



'''
This script validates the algorithm in "weibull_ci.py" to generate the confidence intervals with parametric bootstrapping.
The parametric bootstrapping method is compared with the analytical method implemented in the reliability library by MatthewReid854.
This is done for the Weibull 2P with several part data as well as synthetic data.
'''
#-----------------------------------------------------------------------------------------------------------------------
# Function for Weibull 2P with additional parametric bootstrapping / Monte Carlo confidence intervals to validate own implementation
#-----------------------------------------------------------------------------------------------------------------------
def weibull_2p_param_bootstrapping(part, ci=0.95, save_path=None):
    if not part:
        raise RuntimeError('Invalid request ("part" not specified)')

    data = get_data(part)

    failure_size = len(data['failures'])
    suspension_size = len(data['suspensions'])
    sample_size = failure_size + suspension_size

    # Prevent zeros in the right censored data
    if any(t == 0 for t in data['suspensions']):
        data['suspensions'] = [t for t in data['suspensions'] if t > 0]
        with warnings.catch_warnings():
            warnings.simplefilter('always', UserWarning)
            warnings.warn(f'The suspension data contained zeros as running_time. These assets have been ignored. Data need to be checked for {part}.', UserWarning)

    if not data['suspensions']:
        data['suspensions'] = None

    fig = plt.figure(figsize=(10, 12))
    gs = fig.add_gridspec(2, 1, height_ratios=[2, 1], hspace=0.4)

    ax_wb = fig.add_subplot(gs[0])
    ax_dev = fig.add_subplot(gs[1])

    plt.sca(ax_wb)

    # see https://reliability.readthedocs.io/en/latest/API/Fitters.html for parameters description
    wb = Fit_Weibull_2P(failures=data['failures'], right_censored=data['suspensions'],
                        show_probability_plot=True, print_results=False,    # Results can be found in the returned variables as well
                        method='MLE', optimizer='best',                     # Run with all Optimizers: “TNC”, “L-BFGS-B”, “nelder-mead”, and “powell”
                        CI_type='reliability', CI=ci,
                        label=f'Weibull 2 Parameter fit | MLE \n (n = {sample_size} (f: {failure_size} | s: {suspension_size})')

    ax_wb.set_title(f'Weibull Probability Plot for {part} with CI calculated by parametric bootstrapping and \n (α={wb.alpha:.4f}, β={wb.beta:.4f}, CI={ci:.3f})')
    ax_wb.set_xlabel('Time in days')
    ax_wb.set_ylabel('Failure probability')
    ax_wb.set_ylim(0.001, 0.999)
    xmin, xmax = ax_wb.get_xlim()
    xmin_rel, xmax_rel = xmin * 0.8, xmax * 1.2
    ax_wb.set_xlim(xmin_rel, xmax_rel)

    labels = ax_wb.get_xticklabels()
    for i, label in enumerate(labels):
        label.set_visible(i < 3 or (i - 3) % 2 == 0)

    # Calculation of the Confidence Interval:---------------------------------------------------------------------------
    xvals = np.logspace(np.log10(xmin_rel), np.log10(xmax_rel), 800)

    lower, upper = weibull_2p_fisher_bounds(fit=wb, xvals=xvals, failures=data['failures'],
                                                 right_censored=data['suspensions'], CI=ci)

    if lower is not None and upper is not None:
        ax_wb.fill_between(
            xvals,
            lower,
            upper,
            alpha=0.3,
            color='r',
            label=f"{int(ci * 100)}% Monte Carlo CI"
        )
    # -------------------------------------------------------------------------------------------------------------------

    ax_wb.legend(loc='upper left')

    lower_lib, point_lib, upper_lib = wb.distribution.CDF(CI_x=xvals, CI=ci, CI_type='reliability', show_plot=False)

    # DataFrame for the analytics
    df = pd.DataFrame({
        'part': part,
        'x': xvals,
        'lower_mc': lower,
        'lower_lib': lower_lib,
        'upper_mc': upper,
        'upper_lib': upper_lib,
        'diff_lower': lower - lower_lib,
        'diff_upper': upper - upper_lib,
        'pct_diff_lower': (lower - lower_lib) / np.clip(lower_lib, 1e-9, None) * 100,
        'pct_diff_upper': (upper - upper_lib) / np.clip(upper_lib, 1e-9, None) * 100,
    })

    # Comparison plot
    ax_dev.plot(xvals, df['pct_diff_lower'], color='steelblue', label='lower bound diff')
    ax_dev.plot(xvals, df['pct_diff_upper'], color='tomato', label='upper bound diff')
    ax_dev.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax_dev.set_xscale('log')
    ax_dev.set_xlabel('Time in days')
    ax_dev.set_ylabel('Relative difference [%]')
    ax_dev.set_title(f'Relative deviation for {part}: (MC - Analytical) / Analytical × 100')
    ax_dev.legend()
    ax_dev.grid(True, which='both', linestyle='--', alpha=0.5)

    print(f"\nMax  |diff| lower: {df['diff_lower'].abs().max() * 100:.4f} %")
    print(f"Max  |diff| upper: {df['diff_upper'].abs().max() * 100:.4f} %")
    print(f"Mean |diff| lower: {df['diff_lower'].abs().mean() * 100:.4f} %")
    print(f"Mean |diff| upper: {df['diff_upper'].abs().mean() * 100:.4f} %")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    # print(f'Goodness of fit values for the Weibull 2P: \n {wb.goodness_of_fit} \n\n')

    return wb.results, df


#-----------------------------------------------------------------------------------------------------------------------
# Help functions to validate
#-----------------------------------------------------------------------------------------------------------------------
def _u_transform(F):
    """CDF -> log-log transformed scala (Weibull paper y-axis)"""
    F = np.clip(F, 1e-9, 1 - 1e-9)
    return np.log(-np.log(1.0 - F))

def _u_inverse(u):
    """Back transformation from u-scala -> CDF"""
    return 1.0 - np.exp(-np.exp(u))


def _compute_covariance(neg_loglik_fn, params):
    """
    Computes the covariance matrix as the inverse of the Fisher information matrix.

    The Fisher information matrix is the Hessian of the negative log-likelihood
    evaluated at the MLE. The Hessian is computed exactly via automatic
    differentiation (autograd) as the Jacobian of the gradient, avoiding the
    need for step-size tuning inherent in numerical differentiation.

    Parameters
    ----------
    neg_loglik_fn : callable
                    Function f(params) -> float returning the negative log-likelihood.
                    Must be written using autograd.numpy operations to support AD.
    params : array-like, dtype=float
             Maximum likelihood parameter estimate (MLE). Must be a float array
             for autograd compatibility.

    Returns
    -------
    cov : ndarray, shape (n_params, n_params) or None
        Estimated covariance matrix of the parameter estimates.
        Returns None if the Hessian is singular, the covariance matrix has
        negative diagonal entries, or contains NaN values (with UserWarning).
    """
    grad_fn = autograd.grad(neg_loglik_fn)
    hess_fn = autograd.jacobian(grad_fn)

    H = hess_fn(params)

    try:
        cov = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        warnings.warn("Hessian is singular – the fit may not have converged or there may be too few failures for a stable estimation.", UserWarning)
        return None

    if np.any(np.diag(cov) < 0):
        warnings.warn("The covariance matrix has negative diagonal entries. Try a different step size (e.g., step=1e-3 or step=1e-5).", UserWarning)
        return None

    if np.any(np.isnan(cov)):
        warnings.warn("The covariance matrix contains 'nan' values. It's not possible to calculate a confidence interval. Result without interval.", UserWarning)
        return None

    return cov


def _sample_and_compute_bounds(cdf_fn, params, cov, xvals, CI, n_samples, return_sf, seed):
    """
    Estimates confidence interval bounds via parametric Monte Carlo sampling.

    Draws n_samples parameter vectors from a multivariate normal distribution
    N(params, cov), evaluates the CDF curve over xvals for each sample, and
    returns the CI percentile bounds. For CDF bounds, percentiles are computed
    on the Weibull linearization scale (u = log(-log(1-F))) and back-transformed
    to CDF space to reduce the distortion introduced by the nonlinearity of the
    Weibull CDF.

    Parameters
    ----------
    cdf_fn    : callable
                CDF function with signature f(t, *params) -> array.
    params    : array-like
                MLE parameter vector (on the transformed scale, e.g. log-scale).
    cov       : ndarray or None
                Covariance matrix of the parameter estimates. If None, (None, None)
                is returned immediately.
    xvals     : array-like
                x-values at which the bounds are evaluated.
    CI        : float
                Confidence level (e.g. 0.95).
    n_samples : int
                Number of Monte Carlo samples.
    return_sf : bool
                If True, returns survival function (SF) bounds instead of CDF bounds.
                SF bounds are computed directly via percentiles without u-transformation.
    seed      : int
                Random seed for reproducibility.

    Returns
    -------
    lower : ndarray
            Lower confidence bound on the CDF (or SF if return_sf=True) scale.
    upper : ndarray
            Upper confidence bound on the CDF (or SF if return_sf=True) scale.
    Returns (None, None) if cov is None or no samples are generated.
    """
    # If cov contains NaN then just return None, None as upper and lower --> no calculation of the CI
    if cov is None:
        return None, None

    rng = np.random.default_rng(seed=seed)
    samples = rng.multivariate_normal(params, cov, size=n_samples)

    if len(samples) == 0:
        return None, None

    xvals = np.asarray(xvals)
    curves = np.stack([cdf_fn(xvals, *s) for s in samples], axis=0)

    if return_sf:
        curves = 1.0 - curves
        alpha_tail = (1.0 - CI) / 2.0
        lower = np.percentile(curves, alpha_tail * 100.0, axis=0)
        upper = np.percentile(curves, (1.0 - alpha_tail) * 100.0, axis=0)
    else:
        # Calculate the percentiles on the u-scale and transform afterwards back
        u_curves = _u_transform(curves)
        alpha_tail = (1.0 - CI) / 2.0
        u_lower = np.percentile(u_curves, alpha_tail * 100.0, axis=0)
        u_upper = np.percentile(u_curves, (1.0 - alpha_tail) * 100.0, axis=0)
        lower = _u_inverse(u_lower)
        upper = _u_inverse(u_upper)

    return lower, upper


# -----------------------------------------------------------------------------------------------------------------------
# Function to calculate the confidence bounds for Weibull 2P
# -----------------------------------------------------------------------------------------------------------------------
def weibull_2p_fisher_bounds(fit, xvals, failures, right_censored=None, CI=0.95, n_samples=10000, return_sf=False, seed=42):
    """
    Computes confidence intervals for a Weibull 2P model using Fisher-matrix-based
    Monte Carlo sampling.

    The covariance matrix is derived analytically via automatic differentiation
    (autograd) of the negative log-likelihood. Parameter sampling is performed on
    the log-scale (ln(alpha), ln(beta)) to ensure positivity and improve the
    normality assumption. Percentiles are evaluated on the Weibull linearization
    scale (u-scale) before back-transformation to CDF space.

    Parameters
    ----------
    fit           : Fit_Weibull_2P
                    Already fitted model object from the reliability library.
    xvals         : array-like
                    x-values at which the CI is evaluated. Should be derived
                    from the raw data range, not from ax.get_xlim().
    failures      : list or array
                    Failure times.
    right_censored: list or array, optional
                    Suspension (right-censored) times.
    CI            : float, default 0.95
                    Confidence level, e.g. 0.95 for a 95% CI.
    n_samples     : int, default 10000
                    Number of Monte Carlo samples drawn from the parameter distribution.
    return_sf     : bool, default False
                    If True, returns bounds for the survival function (SF) instead of the CDF.
    seed          : int, default 42
                    Random seed for reproducibility.

    Returns
    -------
    lower : ndarray, shape (len(xvals),)
            Lower confidence bound.
    upper : ndarray, shape (len(xvals),)
            Upper confidence bound.
    """
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


#-----------------------------------------------------------------------------------------------------------------------
# Call the functions and do some analysis
#-----------------------------------------------------------------------------------------------------------------------
part_names = ['HCCBWDB', 'HCCFCRJ', 'HCCBMIA', 'HCCVORB', 'HCCVREC', 'HCCTDAG', 'HCCFFIE', 'HCCVFEA', 'HCCVSWB',
              'HCCVUNC', 'HCCVFEB', 'HCCFEIA', 'HCCVSEA', 'HCCFCRB', 'HCCVAED', 'HCCTRI', 'HCCVSWA', 'HCCTRP']


base_dir = r"C:\Users\lgroha\cernbox\Documents\Masterthesis\4_Python-Tool\Validate_CI"
os.makedirs(base_dir, exist_ok=True)

all_dfs = []

for part in part_names:
    png_path = os.path.join(base_dir, f"{part}_weibull_CI-Comparison.png")
    results, df = weibull_2p_param_bootstrapping(part, save_path=png_path)
    all_dfs.append(df)

df_all = pd.concat(all_dfs, ignore_index=True)
df_all = df_all.sort_values(['part', 'x']).reset_index(drop=True)

csv_path = os.path.join(base_dir, "Weibull_CI-comparison_all_parts.csv")
df_all.to_csv(csv_path, index=False, float_format='%.6f')


# weibull_2p_analytic(part=part_name)

# weibull_2p_param_bootstrapping(part=part_name)

