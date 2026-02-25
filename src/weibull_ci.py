#!/usr/bin/python3
import warnings
import numpy as np
import numdifftools as nd
from reliability.Fitters import Fit_Weibull_Mixture, Fit_Weibull_CR



"""
Fisher Matrix based confidence intervals for 
Weibull Mixture and Weibull Competing Risks
"""
#-----------------------------------------------------------------------------------------------------------------------
# Help functions
#-----------------------------------------------------------------------------------------------------------------------
def _weibull_cdf(t, alpha, beta):
    """Single Weibull CDF"""
    return 1.0 - np.exp(-(t / alpha) ** beta)


def _mixture_cdf(t, a1, b1, a2, b2, p):
    """
    Weibull Mixture CDF:
        F(t) = p * F1(t) + (1-p) * F2(t)
    """
    return p * _weibull_cdf(t, a1, b1) + (1.0 - p) * _weibull_cdf(t, a2, b2)


def _cr_cdf(t, a1, b1, a2, b2):
    """
    Weibull Competing Risks CDF:
        F(t) = 1 - S1(t) * S2(t)
    """
    sf1 = 1.0 - _weibull_cdf(t, a1, b1)
    sf2 = 1.0 - _weibull_cdf(t, a2, b2)
    return 1.0 - sf1 * sf2


def _compute_covariance(neg_loglik_fn, params, step=1e-4):
    """
    Computes the covariance matrix as the inverse of the Fisher information matrix.

    The Fisher information matrix is defined as the Hessian matrix of the negative
    log-likelihood function, evaluated at the maximum likelihood estimate (MLE).

    Parameters
    ----------
    neg_loglik_fn : callable
        Function f(params) -> float returning the negative log-likelihood.
    params : array-like
        Maximum likelihood parameter estimate (MLE).
    step : float
        Step size used for numerical differentiation.

    Returns
    -------
    cov : ndarray, shape (n_params, n_params) or None
        Estimated covariance matrix of the parameter estimates or None if calculation fails. (includes UserWarning)
    """
    hess_fn = nd.Hessian(neg_loglik_fn, step=step)
    H = hess_fn(params)

    try:
        cov = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        warnings.warn("Hessian is singular – the fit may not have converged or there may be too few failures for a stable estimation.", UserWarning)
        return None

    if np.any(np.diag(cov) < 0):
        warnings.warn("The covariance matrix has negative diagonal entries. Try a different step size (e.g., step=1e-3 or step=1e-5).", UserWarning)
        return None

    return cov


def _sample_and_compute_bounds(cdf_fn, params, cov, xvals, CI, n_samples, return_sf, seed):
    """
    Parametric Monte Carlo sampling for confidence interval (CI) estimation.

    Draws n_samples parameter vectors from N(params, cov),
    computes the CDF curve over xvals for each sample, and
    returns the corresponding CI percentiles.

    Parameters
    ----------
    cdf_fn    : callable
        Function with signature f(t, *params) -> array.
    params    : array-like
        Maximum likelihood parameter estimate (MLE).
    cov       : ndarray
        Covariance matrix of the parameter estimates.
    xvals     : array-like
        x-values at which the function is evaluated.
    CI        : float
        Confidence level (e.g., 0.95).
    n_samples : int
        Number of Monte Carlo samples.
    return_sf : bool
        If True, return survival function (SF) bounds instead of CDF bounds.
    seed      : int
        Random seed for reproducibility.

    Returns
    -------
    lower : ndarray
        Lower confidence bound.
    upper : ndarray
        Upper confidence bound.
    Or (None, None) if too few valid samples occur
    """
    rng = np.random.default_rng(seed=seed)
    samples = rng.multivariate_normal(params, cov, size=n_samples)

    # Physically invalid parameter space is rejected.
    # All alpha and beta parameters must be > 0.
    # The proportion parameter (last parameter in a mixture model) must lie within the interval [0, 1].

    n_params = len(params)
    valid = np.ones(len(samples), dtype=bool)

    for i in range(n_params - 1):
        valid &= samples[:, i] > 0  # alpha and beta > 0

    # Last parameter: proportion in the mixture model, otherwise it must also be > 0.
    if n_params == 5:
        # Mixture: proportion_1 must lie within [0, 1]
        valid &= (samples[:, 4] >= 0) & (samples[:, 4] <= 1)
    else:
        valid &= samples[:, n_params - 1] > 0

    samples = samples[valid]

    if len(samples) < 100:
        warnings.warn(f"Only {len(samples)} valid samples after filtering (out of {n_samples}). "
            "The covariance matrix may be too large — typical for small sample sizes (< 20 failures). "
            "Fisher-matrix-based confidence intervals are generally unreliable for small samples. CI will not be displayed.", UserWarning)
        return None, None

    xvals = np.asarray(xvals)
    curves = np.stack([cdf_fn(xvals, *s) for s in samples], axis=0)

    if return_sf:
        curves = 1.0 - curves

    alpha_tail = (1.0 - CI) / 2.0
    lower = np.percentile(curves, alpha_tail * 100.0, axis=0)
    upper = np.percentile(curves, (1.0 - alpha_tail) * 100.0, axis=0)

    return lower, upper

def to_weibull_y(F):
    F = np.clip(F, 1e-9, 1 - 1e-9)
    return np.log(-np.log(1.0 - F))


#-----------------------------------------------------------------------------------------------------------------------
# Main functions
#-----------------------------------------------------------------------------------------------------------------------
def weibull_mixture_fisher_bounds(fit, xvals, failures, right_censored=None, CI=0.95, n_samples=5000, return_sf=False, seed=42, hess_step=1e-4):
    """
    Fisher-matrix-based confidence intervals for a Weibull mixture model.

    Parameters
    ----------
    fit            : Fit_Weibull_Mixture
                     Already fitted model object
    xvals          : array-like
                     x-values at which the confidence intervals are evaluated
                     IMPORTANT: Derive from raw data (not from ax.get_xlim()),
                     e.g.: np.logspace(np.log10(min(failures)*0.5), np.log10(max(failures)*2.0), 400)
    failures       : list or array-like
                     Failure times (required for Hessian computation)
    right_censored : list or array-like, optional
                     Right-censored times
    CI             : float
                     Confidence level, default 0.95
    n_samples      : int
                     Number of Monte Carlo samples, default 5000
    return_sf      : bool
                     If True, return bounds for the survival function (SF)
                     instead of the CDF
    seed           : int
                     Random seed for reproducibility
    hess_step      : float
                     Step size for numerical Hessian computation

    Returns
    -------
    lower : ndarray, shape (len(xvals),)
    upper : ndarray, shape (len(xvals),)
    """
    T_f  = np.asarray(failures)
    T_rc = np.asarray(right_censored) if right_censored is not None else np.array([])

    params = np.array([
        fit.alpha_1,
        fit.beta_1,
        fit.alpha_2,
        fit.beta_2,
        fit.proportion_1
    ])

    def neg_loglik(p):
        return Fit_Weibull_Mixture.LL(p, T_f, T_rc)

    cov = _compute_covariance(neg_loglik, params, step=hess_step)

    return _sample_and_compute_bounds(
        cdf_fn=_mixture_cdf,
        params=params,
        cov=cov,
        xvals=xvals,
        CI=CI,
        n_samples=n_samples,
        return_sf=return_sf,
        seed=seed
    )


def weibull_cr_fisher_bounds(fit, xvals, failures, right_censored=None, CI=0.95, n_samples=5000, return_sf=False, seed=42, hess_step=1e-4):
    """
    Fisher-matrix-based confidence intervals for a Weibull competing risks model.

    Parameters
    ----------
    fit             : Fit_Weibull_CR
                      Already fitted model object
    xvals           : array
                      x-values (derive from raw data, not from ax.get_xlim())
    failures        : list or array
    right_censored  : list or array, optional
    CI              : float, default 0.95
    n_samples       : int, default 5000
    return_sf       : bool
    seed            : int
    hess_step       : float

    Returns
    -------
    lower : array, shape (len(xvals),)
    upper : array, shape (len(xvals),)
    """
    T_f  = np.asarray(failures)
    T_rc = np.asarray(right_censored) if right_censored is not None else np.array([])

    params = np.array([
        fit.alpha_1,
        fit.beta_1,
        fit.alpha_2,
        fit.beta_2
    ])

    def neg_loglik(p):
        return Fit_Weibull_CR.LL(p, T_f, T_rc)

    cov = _compute_covariance(neg_loglik, params, step=hess_step)

    return _sample_and_compute_bounds(
        cdf_fn=_cr_cdf,
        params=params,
        cov=cov,
        xvals=xvals,
        CI=CI,
        n_samples=n_samples,
        return_sf=return_sf,
        seed=seed
    )
