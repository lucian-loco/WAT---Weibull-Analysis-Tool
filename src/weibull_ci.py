#!/usr/bin/python3
"""
weibull_ci.py
==============

Confidence-interval (CI) computation engine for Weibull reliability models, implementing three independent methods 
for bounding the fitted CDF/SF curves of Weibull_2P, Weibull_3P, Weibull_Mixture, and Weibull_CR (Competing Risks) 
models fitted via the `reliability` package.

Three complementary CI approaches are provided:

1. **Parametric Monte Carlo (Fisher-matrix based)** — `*_fisher_bounds` functions. 
   The Hessian of the negative log-likelihood is computed exactly at the MLE via automatic differentiation (`autograd`),
   inverted to obtain the asymptotic covariance matrix, and used to draw random parameter vectors from a multivariate 
   normal distribution. Each sampled curve is evaluated on the Weibull linearization (u) scale, and pointwise
   percentiles yield the bounds. Physically invalid samples/curves (e.g. out-of-range mixture proportions, 
   non-monotonic CDFs) are discarded.

2. **Analytical delta method** — `*_analytical_bounds` functions. Uses the same Fisher-information covariance matrix, 
   but derives bounds analytically via the delta method on the u-scale (Var(u) = grad(u)ᵀ·C·grad(u)) instead of sampling. 
   This is faster and produces smooth, non-stochastic bounds, directly comparable to the Monte Carlo approach since 
   both share the same covariance matrix. The mixture proportion parameter is bounded separately via a logit
   transform to keep it within [0, 1].

3. **Non-parametric bootstrap** — `*_bootstrap_bounds` functions. Resamples the original failure/suspension units 
   with replacement (preserving censoring structure), refits the model via MLE on each resample, and derives pointwise 
   percentile bounds from the resulting family of curves. Requires a minimum number of valid bootstrap fits (500) to
   return bounds; warns if the valid-sample fraction falls below 75%.

All three approaches support returning either CDF (failure probability) or SF (reliability/survival) bounds 
via the `return_sf` flag, and are designed to plug into the plotting functions of `weibull_analysis.py`.

Dependencies: `autograd` (automatic differentiation for exact Hessians/Jacobians), 
`scipy.stats` (normal quantiles for the delta method), and the `reliability` package's `Fit_Weibull_*` classes 
(for `.LL()` log-likelihood functions and fitted parameter attributes).

Author: Lucian Groha
"""
import scipy
import warnings
import numpy as np
import autograd
import autograd.numpy as anp
from reliability.Fitters import Fit_Weibull_Mixture, Fit_Weibull_CR, Fit_Weibull_2P, Fit_Weibull_3P


# ToDo: Think of adjusting the number of xvals automated to the range of the failures, whats the best case?
# ToDo: If the spread between upper and lower is more than 0.99 then the confidence bounds are not suitable...think of a way to catch this
"""
Fisher Matrix based parametric Monte Carlo confidence intervals 
for Weibull Mixture and Weibull Competing Risks
"""
"""
How does it work so far theoretically:
    1. Covariance matrix: The Hessian of the negative log-likelihood is computed exactly at the
       MLE using automatic differentiation (autograd). Its inverse yields the Fisher-information-
       based covariance matrix C = H^-1, which quantifies parameter uncertainty.
    2. Monte Carlo sampling: N parameter vectors are drawn from the asymptotic multivariate normal
       distribution N(params, C) of the MLE. All scale and shape parameters (alpha, beta) are
       sampled on log-scale to guarantee positivity. For mixture models, samples with a proportion
       parameter outside [0, 1] are discarded as physically invalid.
    3. Pointwise CI: For each sampled parameter vector, the full CDF curve is evaluated over the
       x-grid. Percentiles are computed on the Weibull linearization scale (u = log(-log(1-F)))
       and back-transformed to CDF space, reducing distortion from the nonlinearity of the Weibull
       CDF. Pointwise alpha/2 and 1-alpha/2 percentiles across all valid curves yield the lower
       and upper confidence bounds at each time point.
This approach is similar to the classical analytical Fisher-matrix method (Delta method), but
avoids explicit gradient derivation of the composite CDF. This makes it directly applicable to
multi-parameter models such as Weibull Mixture and Weibull Competing Risks, where closed-form
derivatives are difficult to derive. Censored observations are correctly accounted for through
the likelihood function used to compute the Hessian.
"""
#-----------------------------------------------------------------------------------------------------------------------
# Help functions for parametric Monte Carlo
#-----------------------------------------------------------------------------------------------------------------------
def _weibull_cdf(t, alpha, beta):
    """Computes the CDF of a two-parameter Weibull distribution"""
    return 1.0 - np.exp(-(t / alpha) ** beta)


def _weibull_3p_cdf(t, alpha, beta, gamma):
    """
    Computes the CDF of a three-parameter Weibull distribution:
        F(t) = 1 - exp(-((t - gamma) / alpha)^beta)
    Valid only for t > gamma; returns 0 for t <= gamma.
    """
    t = np.asarray(t, dtype=float)
    result = np.where(t > gamma, _weibull_cdf(t - gamma, alpha, beta), 0.0)
    return result


def _mixture_cdf(t, a1, b1, a2, b2, p):
    """
    Computes the CDF of a two-component Weibull mixture model:
        F(t) = p * F1(t) + (1-p) * F2(t)
    """
    return p * _weibull_cdf(t, a1, b1) + (1.0 - p) * _weibull_cdf(t, a2, b2)


def _cr_cdf(t, a1, b1, a2, b2):
    """
    Computes the CDF of a two-component Weibull competing risks model:
        F(t) = 1 - S1(t) * S2(t)
    """
    sf1 = 1.0 - _weibull_cdf(t, a1, b1)
    sf2 = 1.0 - _weibull_cdf(t, a2, b2)
    return 1.0 - sf1 * sf2


def _u_transform(F):
    """Transforms CDF values to the Weibull linearization scale (u-scale) in log scale"""
    F = np.clip(F, 1e-9, 1 - 1e-9)
    return np.log(-np.log(1.0 - F))


def _u_inverse(u):
    """Back-transforms values from the Weibull linearization scale (u-scale) to CDF space"""
    return 1.0 - np.exp(-np.exp(u))


# This function is also used for the analytical Delta method
def _compute_covariance(neg_loglik_fn, params):
    """
    Computes the covariance matrix as the inverse of the Fisher information matrix.

    The Fisher information matrix is the Hessian of the negative log-likelihood evaluated at the MLE.
    The Hessian is computed exactly via automatic differentiation (autograd) as the Jacobian of the gradient,
    avoiding the need for step-size tuning inherent in numerical differentiation.

    Parameters
    ----------
    neg_loglik_fn : callable
                    Negative log-likelihood function f(params) -> float.
                    Must use autograd.numpy operations to support automatic differentiation.
    params        : ndarray, dtype=float
                    MLE parameter vector. Must be a float array for autograd compatibility.

    Returns
    -------
    cov : ndarray, shape (n_params, n_params), or None
          Estimated covariance matrix of the parameter estimates. Returns None with a UserWarning if the Hessian is
          singular, the covariance matrix has negative diagonal entries, or contains NaN values.
    """
    grad_fn = autograd.grad(neg_loglik_fn)
    hess_fn = autograd.jacobian(grad_fn)

    try:
        H = hess_fn(params)
    except Exception as e:
        warnings.warn(f'Hessian computation failed: {e}', UserWarning)
        return None

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

    # # Matrix muss positiv semi-definit sein
    # eigenvalues = np.linalg.eigvalsh(cov)
    # print(f'Eigenwerte: {eigenvalues}')
    #
    # # Konditionszahl — hohe Werte (~>1e10) deuten auf numerische Instabilität
    # print(f'Konditionszahl: {np.linalg.cond(cov)}')

    return cov


def _sample_and_compute_bounds(cdf_fn, params, cov, xvals, CI, n_samples, return_sf, seed, min_failure=None):
    """
    Estimate pointwise CI bounds via parametric Monte Carlo sampling from the asymptotic parameter distribution.

    Draws `n_samples` parameter vectors from N(params, cov), evaluates the CDF curve over `xvals` for each valid sample,
    and computes percentile bounds on the Weibull linearization (u) scale before back-transforming to CDF (or SF) space.
    Physically invalid samples (mixture proportions outside [0,1], 3P gamma exceeding the earliest failure, or
     non-monotonic/out-of-range CDF curves) are discarded before computing percentiles.

    Parameters
    ----------
    cdf_fn : callable
        CDF function with signature f(t, *params) -> array. Must accept parameters on the same scale as `params`
        (e.g. log-scale for alpha/beta).
    params : ndarray
        MLE parameter vector on the sampling scale (log-scale for alpha/beta, linear for gamma/proportion).
    cov : ndarray or None
        Covariance matrix of the parameter estimates. If None, (None, None) is returned immediately.
    xvals : array-like
        Time values at which bounds are evaluated.
    CI : float
        Confidence level (e.g. 0.95 for a 95% CI).
    n_samples : int
        Number of Monte Carlo samples to draw.
    return_sf : bool
        If True, returns bounds for the survival function (SF = 1 - CDF) with bounds correctly swapped
        (SF_lower = 1 - CDF_upper).
    seed : int
        Random seed for reproducibility.
    min_failure : float, optional
        For Weibull 3P only: the minimum observed failure time. Samples with gamma >= min_failure are discarded as
        physically invalid (failure-free time cannot exceed the earliest observed failure).

    Returns
    -------
    tuple
        (lower, upper): ndarrays of pointwise confidence bounds on the CDF (or SF) scale, clipped to [1e-9, 1-1e-9].
        Returns (None, None) if `cov` is None or no valid samples/curves remain after filtering.
    """
    # If cov contains NaN then just return None, None as upper and lower --> no calculation of the CI
    if cov is None:
        return None, None

    rng = np.random.default_rng(seed=seed)
    samples = rng.multivariate_normal(params, cov, size=n_samples)

    # Filter samples with proportion factor out of (0,1) for Weibull Mixture
    if len(params) == 5:
        valid = (samples[:, 4] > 0) & (samples[:, 4] < 1)
        samples = samples[valid]

#ToDo: Maybe include also logit transformation in params for proportion factor p to prevent bad samples

    # Filter samples with gamma < min(T_f) since it is not physical correct
    # Only applies to Weibull 3P
    if min_failure is not None:
        valid = samples[:, 2] < min_failure
        samples = samples[valid]

    if len(samples) == 0:
        return None, None

    xvals = np.asarray(xvals)
    curves = np.stack([cdf_fn(xvals, *s) for s in samples], axis=0)

    # Filter curves that are physically invalid
    valid_curves = (
            np.all(curves >= 0, axis=1) &  # CDF must be non-negative
            np.all(curves <= 1, axis=1) &  # CDF must not exceed 1
            np.all(np.diff(curves, axis=1) >= 0, axis=1)  # CDF must be monotonically increasing
    )
    curves = curves[valid_curves]

    if len(curves) == 0:
        return None, None

    # Calculate the percentiles on the u-scale and transform back afterward
    u_curves = _u_transform(curves)
    alpha_tail = (1.0 - CI) / 2.0
    u_lower = np.percentile(u_curves, alpha_tail * 100.0, axis=0)
    u_upper = np.percentile(u_curves, (1.0 - alpha_tail) * 100.0, axis=0)
    lower = _u_inverse(u_lower)
    upper = _u_inverse(u_upper)
    # alpha_tail = (1.0 - CI) / 2.0
    # lower = np.percentile(curves, alpha_tail * 100.0, axis=0)
    # upper = np.percentile(curves, (1.0 - alpha_tail) * 100.0, axis=0)

    if return_sf:
        lower_sf = 1 - upper
        upper_sf = 1 - lower

        return np.clip(lower_sf, 1e-9, 1 - 1e-9), np.clip(upper_sf, 1e-9, 1 - 1e-9)

    lower = np.clip(lower, 1e-9, 1 - 1e-9)
    upper = np.clip(upper, 1e-9, 1 - 1e-9)

    return lower, upper


#-----------------------------------------------------------------------------------------------------------------------
# Main functions parametric Monte Carlo: Mixture, Competing Risk, 3P and 2P
#-----------------------------------------------------------------------------------------------------------------------
def weibull_mixture_fisher_bounds(fit, xvals, failures, right_censored=None, CI=0.95, n_samples=10000, return_sf=False, seed=42):
    """
    Compute parametric Monte Carlo (Fisher-matrix based) confidence bounds for a fitted Weibull Mixture model's
    CDF or SF curve.

    Parameters
    ----------
    fit : Fit_Weibull_Mixture
        Already-fitted mixture model object exposing alpha_1, beta_1, alpha_2, beta_2, proportion_1.
    xvals : array-like
        x-values at which bounds are evaluated. Should be derived from the raw data range (e.g. via `np.logspace`),
        not from `ax.get_xlim()`.
    failures : list or array-like
        Failure times, required to reconstruct the Hessian/log-likelihood.
    right_censored : list or array-like, optional
        Right-censored (suspension) times.
    CI : float, optional
        Confidence level (default: 0.95).
    n_samples : int, optional
        Number of Monte Carlo samples (default: 10000).
    return_sf : bool, optional
        If True, return bounds for the survival function instead of CDF.
    seed : int, optional
        Random seed for reproducibility (default: 42).

    Returns
    -------
    tuple
        (lower, upper): ndarrays of shape (len(xvals),), or (None, None) if the covariance matrix could not be computed
        or no valid samples remain.
    """
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
    """
    Compute parametric Monte Carlo (Fisher-matrix based) confidence bounds for a fitted Weibull Competing Risks model's
    CDF or SF curve.

    Parameters
    ----------
    fit : Fit_Weibull_CR
        Already-fitted competing risks model object exposing alpha_1, beta_1, alpha_2, beta_2.
    xvals : array-like
        x-values at which bounds are evaluated (derive from raw data, not `ax.get_xlim()`).
    failures : list or array-like
        Failure times.
    right_censored : list or array-like, optional
        Right-censored (suspension) times.
    CI : float, optional
        Confidence level (default: 0.95).
    n_samples : int, optional
        Number of Monte Carlo samples (default: 10000).
    return_sf : bool, optional
        If True, return bounds for the survival function instead of CDF.
    seed : int, optional
        Random seed for reproducibility (default: 42).

    Returns
    -------
    tuple
        (lower, upper): ndarrays of shape (len(xvals),), or (None, None) if the covariance matrix could not be computed
        or no valid samples remain.
    """
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


def weibull_2p_fisher_bounds(fit, xvals, failures, right_censored=None, CI=0.95, n_samples=10000, return_sf=False, seed=42):
    """
    Compute parametric Monte Carlo (Fisher-matrix based) confidence bounds for a fitted Weibull 2P model's 
    CDF or SF curve.

    The covariance matrix is derived via automatic differentiation of the negative log-likelihood; 
    parameters are sampled on the log-scale (ln(alpha), ln(beta)) to guarantee positivity, and percentiles are computed 
    on the Weibull linearization (u) scale before back-transform.

    Parameters
    ----------
    fit : Fit_Weibull_2P
        Already-fitted 2-parameter Weibull model object from `reliability`.
    xvals : array-like
        x-values at which the CI is evaluated.
    failures : list or array
        Failure times.
    right_censored : list or array, optional
        Suspension (right-censored) times.
    CI : float, optional
        Confidence level (default: 0.95).
    n_samples : int, optional
        Number of Monte Carlo samples drawn (default: 10000).
    return_sf : bool, optional
        If True, return bounds for the survival function instead of CDF (default: False).
    seed : int, optional
        Random seed for reproducibility (default: 42).

    Returns
    -------
    tuple
        (lower, upper): ndarrays of shape (len(xvals),) with the lower and upper confidence bounds, or (None, None) 
        if the covariance matrix could not be computed.
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

    def cdf_fn_log_2p(t, log_alpha, log_beta):
        return _weibull_cdf(t, np.exp(log_alpha), np.exp(log_beta))

    return _sample_and_compute_bounds(
        cdf_fn=cdf_fn_log_2p,
        params=params,
        cov=cov,
        xvals=xvals,
        CI=CI,
        n_samples=n_samples,
        return_sf=return_sf,
        seed=seed
    )


def weibull_3p_fisher_bounds(fit, xvals, failures, right_censored=None, CI=0.95, n_samples=10000, return_sf=False, seed=42):
    """
    Compute parametric Monte Carlo (Fisher-matrix based) confidence bounds for a fitted Weibull 3P model's
    CDF or SF curve.

    The covariance matrix is derived via automatic differentiation of the negative log-likelihood;
    alpha and beta are sampled on the log-scale for positivity while gamma (location parameter) is sampled on the
    linear scale. Percentiles are computed on the Weibull linearization (u) scale before back-transforming to CDF space.
    Samples where gamma would exceed the minimum observed failure time are discarded as physically invalid.

    Parameters
    ----------
    fit : Fit_Weibull_3P
        Already-fitted 3-parameter Weibull model object from `reliability`.
    xvals : array-like
        x-values at which the CI is evaluated. Should be derived from the raw data range, not `ax.get_xlim()`.
    failures : list or array
        Failure times.
    right_censored : list or array, optional
        Suspension (right-censored) times.
    CI : float, optional
        Confidence level (default: 0.95).
    n_samples : int, optional
        Number of Monte Carlo samples drawn (default: 10000).
    return_sf : bool, optional
        If True, return bounds for the survival function instead of CDF (default: False).
    seed : int, optional
        Random seed for reproducibility (default: 42).

    Returns
    -------
    tuple
        (lower, upper): ndarrays of shape (len(xvals),) with the lower and upper confidence bounds, or (None, None)
        if the covariance matrix could not be computed.
    """
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


#***********************************************************************************************************************
"""
Analytical Fisher-matrix-based confidence intervals for Weibull Mixture and Weibull Competing Risks models 
using the delta method.

How it works:
    1. Covariance matrix: identical to the Monte Carlo approach – the Hessian of
       the negative log-likelihood is computed at the MLE via automatic differentiation
       (autograd), and its inverse yields the Fisher-information-based covariance
       matrix C = H^-1.

    2. u-scale linearization: the CDF is transformed to the Weibull linearization
       scale u = ln(-ln(1 - F)), on which the normal approximation underlying the
       delta method holds more accurately.

    3. Delta method: for each time point t, the variance of u is computed analytically as
           Var(u) = grad(u)^T * C * grad(u)
       where grad(u) = du/d(theta) is obtained via autograd.jacobian. This is the
       exact equivalent of the classical Fisher-matrix formula used in ReliaSoft Weibull++,
       applied here to multi-parameter models without requiring closed-form derivatives.

    4. Confidence bounds: symmetric bounds are formed on the u-scale as
           u_U/L = u_hat +/- K_alpha * sqrt(Var(u))
       and back-transformed to CDF (or SF) space via F = 1 - exp(-exp(u)).

    5. Proportion parameter p (mixture only): bounded separately via a logit
       transformation to guarantee bounds within [0, 1], consistent with the
       ReliaSoft approach.

Compared to the parametric Monte Carlo approach above, this method is analytically
exact (within the asymptotic normal approximation), significantly faster, and
produces smooth, non-stochastic bounds. Both methods share the same covariance
matrix and are therefore directly comparable.
"""
#-----------------------------------------------------------------------------------------------------------------------
# Analytical approach to calculate the confidence bounds on reliability / failure probability
#-----------------------------------------------------------------------------------------------------------------------
# u-scale functions (Weibull linearization)
# Also used for Weibull 3P just with shifted t
def u_2p(t, params):
    """u = ln(-ln(1 - F)) for Weibull 2P"""
    log_alpha, log_beta = params[0], params[1]
    F = 1 - anp.exp(-(t / anp.exp(log_alpha)) ** anp.exp(log_beta))
    F = anp.clip(F, 1e-9, 1 - 1e-9)
    return anp.log(-anp.log(1.0 - F))


def u_mixture(t, params):
    """u = ln(-ln(1 - F)) for Weibull Mixture"""
    log_a1, log_b1, log_a2, log_b2, prop = params[0], params[1], params[2], params[3], params[4]
    F = (prop * (1 - anp.exp(-(t / anp.exp(log_a1)) ** anp.exp(log_b1)))
         + (1 - prop) * (1 - anp.exp(-(t / anp.exp(log_a2)) ** anp.exp(log_b2))))
    F = anp.clip(F, 1e-9, 1 - 1e-9)
    return anp.log(-anp.log(1.0 - F))


def u_cr(t, params):
    """u = ln(-ln(1 - F)) for Weibull Competing Risks"""
    log_a1, log_b1, log_a2, log_b2 = params[0], params[1], params[2], params[3]
    S1 = anp.exp(-(t / anp.exp(log_a1)) ** anp.exp(log_b1))
    S2 = anp.exp(-(t / anp.exp(log_a2)) ** anp.exp(log_b2))
    F = 1.0 - S1 * S2
    F = anp.clip(F, 1e-9, 1 - 1e-9)
    return anp.log(-anp.log(1.0 - F))


# -----------------------------------------------------------------------
# Jacobians – defined once, evaluated per t in the loop
# -----------------------------------------------------------------------
# Weibull 2P / 3P
_du_dparams_2p = autograd.jacobian(u_2p, argnum=1)

# Mixture:       5 Parameter (log_a1, log_b1, log_a2, log_b2, prop)
_du_dparams_mixture = autograd.jacobian(u_mixture, argnum=1)

# Competing Risks: 4 Parameter (log_a1, log_b1, log_a2, log_b2)
_du_dparams_cr = autograd.jacobian(u_cr, argnum=1)


#-----------------------------------------------------------------------------------------------------------------------
# Help functions for analytical Delta method
#-----------------------------------------------------------------------------------------------------------------------
# logit transformation for proportional factor
def _logit_bounds_proportion(p_hat, var_p, Z):
    """
    Compute confidence bounds for the Weibull Mixture proportion parameter via a logit transform,
    guaranteeing bounds stay within [0, 1].

    Parameters
    ----------
    p_hat : float
        MLE estimate of the mixing proportion.
    var_p : float
        Variance of the proportion estimate (the [4,4] entry of the parameter covariance matrix,
        since p is linear in the parameter vector).
    Z : float
        Standard normal critical value corresponding to the desired confidence level (e.g. 1.96 for 95%).

    Returns
    -------
    tuple
        (p_lower, p_upper): confidence bounds on the proportion, clipped to [1e-9, 1-1e-9]. Computed via w = logit(p),
        Var(w) = Var(p) / (p(1-p))^2, bounds formed on w and back-transformed via the sigmoid function.
    """
    w_hat = np.log(p_hat / (1.0 - p_hat))
    var_w = var_p / (p_hat * (1.0 - p_hat)) ** 2
    w_lower = w_hat - Z * np.sqrt(var_w)
    w_upper = w_hat + Z * np.sqrt(var_w)
    p_lower = np.clip(1.0 / (1.0 + np.exp(-w_lower)), 1e-9, 1 - 1e-9)
    p_upper = np.clip(1.0 / (1.0 + np.exp(-w_upper)), 1e-9, 1 - 1e-9)
    return p_lower, p_upper


def _u_to_F(u):
    return 1.0 - np.exp(-np.exp(u))


def _delta_bounds(u_fn, grad_fn, params, cov, xvals, Z, return_sf):
    """
    Core analytical delta-method loop shared by the Mixture and Competing-Risks (and, via reuse, 2P)
    confidence bound calculations.

    For each time point, computes the u-scale estimate and its variance via Var(u) = grad(u)^T · cov · grad(u),
    forms symmetric bounds on the u-scale, and back-transforms them to CDF (or SF) space.

    Parameters
    ----------
    u_fn : callable
        Function u_fn(t, params) -> float, computing the u-scale CDF transform (one of `u_2p`, `u_mixture`, `u_cr`).
    grad_fn : callable
        Autograd-jacobian of `u_fn` with respect to `params`, i.e. grad_fn(t, params) -> ndarray of partial derivatives.
    params : ndarray
        MLE parameter vector matching `u_fn`'s expected scale.
    cov : ndarray
        Covariance matrix of the parameter estimates.
    xvals : array-like
        Time values at which to compute bounds.
    Z : float
        Standard normal critical value for the desired confidence level.
    return_sf : bool
        If True, returns survival-function bounds (1 - CDF bounds, correctly swapped) instead of CDF bounds.

    Returns
    -------
    tuple
        (lower, upper): ndarrays of pointwise bounds, or (None, None) if a negative variance is encountered
        at any time point (indicating an invalid covariance matrix), in which case a UserWarning is raised.
    """
    lower, upper = [], []
    for t in np.asarray(xvals):
        u_hat = u_fn(t, params)
        grads = grad_fn(t, params)
        var_u = grads @ cov @ grads

        if var_u < 0.0:
            warnings.warn(f'Negative variance encountered in delta method at t={t:.4g}: var_u = {var_u:.3e}. '
                          f'The covariance matrix may be invalid. No confidence bounds are returned.', UserWarning)

            return None, None

        se_u  = np.sqrt(var_u)
        FL = np.clip(_u_to_F(u_hat - Z * se_u), 1e-9, 1 - 1e-9)
        FU = np.clip(_u_to_F(u_hat + Z * se_u), 1e-9, 1 - 1e-9)
        if return_sf:
            lower.append(1.0 - FU)
            upper.append(1.0 - FL)
        else:
            lower.append(FL)
            upper.append(FU)
    return np.array(lower), np.array(upper)


def _calculate_delta_bounds(fit, xvals, cov, params, CI=0.95, return_sf=False):
    """
    Dispatch analytical Fisher-matrix (delta method) confidence bound computation to the appropriate model based
    on the parameter vector length, for Weibull 2P, 3P, Mixture, and Competing Risks.

    Parameters
    ----------
    fit : fitted model object
        The fitted distribution object (used only to access `proportion_1` for the Mixture case).
    xvals : array-like
        Time values at which bounds are evaluated. For Weibull 3P, these must already be gamma-shifted (t - gamma_hat).
    cov : ndarray or None
        Covariance matrix from `_compute_covariance()`. For Weibull 3P, pass the 2x2 (alpha, beta) covariance only.
    params : ndarray
        Parameter vector matching the model:
        - 2 elements [log_alpha, log_beta] for Weibull 2P or 3P.
        - 4 elements [log_a1, log_b1, log_a2, log_b2] for Competing Risks.
        - 5 elements [log_a1, log_b1, log_a2, log_b2, p] for Mixture.
    CI : float, optional
        Confidence level (default: 0.95).
    return_sf : bool, optional
        If True, return survival-function bounds instead of CDF bounds.

    Returns
    -------
    tuple
        (lower, upper, p_lower, p_upper): lower/upper are ndarrays of bounds on the CDF (or SF) scale; p_lower/p_upper
        are the proportion parameter's bounds (Mixture only, else None). Returns (None, None, None, None)
        if `cov` is None or the parameter vector length is not 2, 4, or 5 (with a UserWarning in the latter case).
    """
    if cov is None:
        return None, None, None, None

    Z = -scipy.stats.norm.ppf((1.0 - CI) / 2.0)

    if len(params) == 5:
        lower, upper = _delta_bounds(u_mixture, _du_dparams_mixture, params, cov, xvals, Z, return_sf)
        p_lower, p_upper = _logit_bounds_proportion(fit.proportion_1, cov[4, 4], Z)

        return lower, upper, p_lower, p_upper

    elif len(params) == 4:
        lower, upper = _delta_bounds(u_cr, _du_dparams_cr, params, cov, xvals, Z, return_sf)

        return lower, upper, None, None

    elif len(params) == 2:
        lower, upper = _delta_bounds(u_2p, _du_dparams_2p, params, cov, xvals, Z, return_sf)

        return lower, upper, None, None

    else:
        warnings.warn(f"Unexpected parameter vector length: {len(params)}. Expected 2, 4, or 5.", UserWarning)
        return None, None, None, None


#-----------------------------------------------------------------------------------------------------------------------
# Main functions for analytical Delta method: Mixture and Competing Risks
#-----------------------------------------------------------------------------------------------------------------------
def weibull_2p_analytical_bounds(fit, xvals, failures, right_censored=None, CI=0.95, return_sf=False):
    """
    Compute analytical Fisher-matrix (delta-method) confidence bounds for a fitted Weibull 2P model's CDF or SF curve.

    Faster and smoother (non-stochastic) than the Monte Carlo equivalent (`weibull_2p_fisher_bounds`),
    using the same underlying covariance matrix but deriving bounds via the delta method on the u-scale.

    Parameters
    ----------
    fit : Fit_Weibull_2P
        Already-fitted 2-parameter Weibull model object.
    xvals : array-like
        Time values at which bounds are evaluated.
    failures : list or array-like
        Failure times.
    right_censored : list or array-like, optional
        Suspension (right-censored) times.
    CI : float, optional
        Confidence level (default: 0.95).
    return_sf : bool, optional
        If True, return survival-function bounds instead of CDF bounds.

    Returns
    -------
    tuple
        (lower, upper, p_lower, p_upper): lower/upper are ndarrays of CDF (or SF) bounds;
        p_lower/p_upper are always None (no proportion parameter for this model).
        Returns (None, None, None, None) if the covariance matrix could not be computed.
    """
    T_f  = np.asarray(failures)
    T_rc = np.asarray(right_censored) if right_censored is not None else np.array([])

    params = np.array([np.log(fit.alpha), np.log(fit.beta)])

    def neg_loglik(p):
        return Fit_Weibull_2P.LL(anp.exp(p), T_f, T_rc)

    cov = _compute_covariance(neg_loglik, params)

    return _calculate_delta_bounds(fit=fit, xvals=xvals, cov=cov, params=params, CI=CI, return_sf=return_sf)


# ToDo: Make this function without xvals already shifted but then use _calcualte_xvals() instead of ax.get_xlim() method
def weibull_3p_analytical_bounds(fit, xvals, failures, right_censored=None, CI=0.95, return_sf=False):
    """
    Compute analytical Fisher-matrix (delta-method) confidence bounds for a fitted Weibull 3P model's CDF or SF curve,
    using a hybrid covariance strategy consistent with the `reliability` library and ReliaSoft Weibull++.

    Covariance strategy:
    1. Var(alpha), Var(beta), Cov(alpha, beta) are computed from a Weibull_2P log-likelihood on gamma-shifted data
       (T_f - gamma_hat), yielding a stable 2x2 Fisher information block.
    2. Var(gamma) is computed separately from the full Weibull_3P log-likelihood, but used only as a convergence check —
       it does not enter the u-scale variance calculation.
    3. Cross-covariances between (alpha, beta) and gamma are treated as zero, since `xvals` are expected to already
       be gamma-shifted (t - gamma_hat), making du/dgamma = 0.

    Parameters
    ----------
    fit : Fit_Weibull_3P
        Already-fitted 3-parameter Weibull model object.
    xvals : array-like
        Gamma-shifted time values (t - gamma_hat), as returned by `ax.get_xlim()` from a reliability Weibull 3P
        probability plot. Values <= 0 (corresponding to t <= gamma) should be excluded by the caller.
    failures : list or array
        Failure times on the original (unshifted) scale.
    right_censored : list or array, optional
        Suspension times on the original (unshifted) scale.
    CI : float, optional
        Confidence level (default: 0.95).
    return_sf : bool, optional
        If True, return survival-function bounds instead of CDF bounds.

    Returns
    -------
    tuple
        (lower, upper, None, None): lower/upper are ndarrays of CDF (or SF) bounds computed from the 2x2 (alpha, beta)
        covariance; the last two elements are always None (no proportion parameter for this model).
        Returns (None, None, None, None) if the 2x2 covariance could not be computed (with a UserWarning).

    Notes
    -----
    If the full 3P covariance (including gamma) fails to compute, a UserWarning is issued,
    but the function still proceeds using the stable 2x2 (alpha, beta) covariance, since gamma's variance is not
    used in the final bound calculation.
    """
    T_f   = np.asarray(failures)
    T_rc  = np.asarray(right_censored) if right_censored is not None else np.array([])
    gamma_hat = fit.gamma

    # ----------------------------------------------------------------
    # Step 1a: 2×2 covariance for (log_alpha, log_beta)
    #          via Weibull_2P LL on gamma-shifted data
    # ----------------------------------------------------------------
    T_f_shifted  = T_f  - gamma_hat
    T_rc_shifted = T_rc - gamma_hat if len(T_rc) > 0 else np.array([])

    params_2p = np.array([np.log(fit.alpha), np.log(fit.beta)])

    def neg_loglik_2p(p):
        return Fit_Weibull_2P.LL(anp.exp(p), T_f_shifted, T_rc_shifted)

    cov_2p = _compute_covariance(neg_loglik_2p, params_2p)

    if cov_2p is None:
        warnings.warn("Weibull 3P analytical bounds: 2P covariance (alpha, beta) could not be computed.", UserWarning)
        return None, None, None, None

    # ----------------------------------------------------------------
    # Step 1b: Var(gamma) from full Weibull_3P LL (diagonal entry only)
    # ----------------------------------------------------------------
    params_3p_full = np.array([np.log(fit.alpha), np.log(fit.beta), fit.gamma])

    def neg_loglik_3p(p):
        return Fit_Weibull_3P.LL(anp.array([anp.exp(p[0]), anp.exp(p[1]), p[2]]), T_f, T_rc)

    cov_3p_full = _compute_covariance(neg_loglik_3p, params_3p_full)

    if cov_3p_full is None:
        warnings.warn("Weibull 3P analytical bounds: 3P covariance (gamma) could not be computed.", UserWarning)

    # ----------------------------------------------------------------
    # Step 2: Delegate to _calculate_delta_bounds.
    #         xvals are already gamma-shifted (from ax.get_xlim()).
    #         Exclude any values <= 0 (correspond to t <= gamma).
    # ----------------------------------------------------------------

    return _calculate_delta_bounds(fit=fit, xvals=xvals, cov=cov_2p, params=params_2p, CI=CI, return_sf=return_sf)


def weibull_mixture_analytical_bounds(fit, xvals, failures, right_censored=None, CI=0.95, return_sf=False):
    """
    Compute analytical Fisher-matrix (delta-method) confidence bounds for a fitted Weibull Mixture model's CDF or SF
    curve, including bounds on the mixing proportion parameter (via logit transform).

    Parameters
    ----------
    fit : Fit_Weibull_Mixture
        Already-fitted mixture model object exposing alpha_1, beta_1, alpha_2, beta_2, proportion_1.
    xvals : array-like
        Time values at which bounds are evaluated.
    failures : list or array-like
        Failure times.
    right_censored : list or array-like, optional
        Suspension (right-censored) times.
    CI : float, optional
        Confidence level (default: 0.95).
    return_sf : bool, optional
        If True, return survival-function bounds instead of CDF bounds.

    Returns
    -------
    tuple
        (lower, upper, p_lower, p_upper): lower/upper are ndarrays of CDF (or SF) bounds;
        p_lower/p_upper are the confidence bounds on the mixing proportion.
        Returns (None, None, None, None) if the covariance matrix could not be computed.
    """
    T_f  = np.asarray(failures)
    T_rc = np.asarray(right_censored) if right_censored is not None else np.array([])

    params = np.array([np.log(fit.alpha_1), np.log(fit.beta_1),
                       np.log(fit.alpha_2), np.log(fit.beta_2),
                       fit.proportion_1])

    def neg_loglik(p):
        p_orig = anp.array([anp.exp(p[0]), anp.exp(p[1]),
                             anp.exp(p[2]), anp.exp(p[3]), p[4]])
        return Fit_Weibull_Mixture.LL(p_orig, T_f, T_rc)

    cov = _compute_covariance(neg_loglik, params)

    return _calculate_delta_bounds(fit=fit, xvals=xvals, cov=cov, params=params, CI=CI, return_sf=return_sf)


def weibull_cr_analytical_bounds(fit, xvals, failures, right_censored=None, CI=0.95, return_sf=False):
    """
    Compute analytical Fisher-matrix (delta-method) confidence bounds for a fitted Weibull Competing Risks model's CDF
    or SF curve.

    Parameters
    ----------
    fit : Fit_Weibull_CR
        Already-fitted competing risks model object exposing alpha_1, beta_1, alpha_2, beta_2.
    xvals : array-like
        Time values at which bounds are evaluated.
    failures : list or array-like
        Failure times.
    right_censored : list or array-like, optional
        Suspension (right-censored) times.
    CI : float, optional
        Confidence level (default: 0.95).
    return_sf : bool, optional
        If True, return survival-function bounds instead of CDF bounds.

    Returns
    -------
    tuple
        (lower, upper, None, None): lower/upper are ndarrays of CDF (or SF) bounds; the last two elements are
        always None (no proportion parameter for this model).
        Returns (None, None, None, None) if the covariance matrix could not be computed.
    """
    T_f  = np.asarray(failures)
    T_rc = np.asarray(right_censored) if right_censored is not None else np.array([])

    params = np.array([np.log(fit.alpha_1), np.log(fit.beta_1),
                       np.log(fit.alpha_2), np.log(fit.beta_2)])

    def neg_loglik(p):
        p_orig = anp.array([anp.exp(p[0]), anp.exp(p[1]),
                             anp.exp(p[2]), anp.exp(p[3])])
        return Fit_Weibull_CR.LL(p_orig, T_f, T_rc)

    cov = _compute_covariance(neg_loglik, params)

    return _calculate_delta_bounds(fit=fit, xvals=xvals, cov=cov, params=params, CI=CI, return_sf=return_sf)


#***********************************************************************************************************************
"""
Bootstrap Confidence Bounds for Weibull Reliability Models
==========================================================

This module implements a non-parametric bootstrap approach to compute
pointwise confidence interval (CI) bounds for reliability (SF) or
failure probability (CDF) curves based on Weibull distributions.

Supported Models
----------------
- Weibull 2P      : Two-parameter Weibull distribution (alpha, beta)
- Weibull 3P      : Three-parameter Weibull distribution (alpha, beta, gamma)
- Weibull Mixture : Two-component Weibull mixture model
                    (alpha_1, beta_1, alpha_2, beta_2, proportion_1)
- Weibull CR      : Competing-risk model with two Weibull components
                    (alpha_1, beta_1, alpha_2, beta_2)

Methodology
-----------
For each bootstrap iteration, all units (failures + suspensions) are resampled
with replacement, preserving the original censoring structure. The model is
refitted via MLE for each sample, the CDF is evaluated over the specified time
values, and pointwise percentile confidence bounds are derived.

Samples with too few failures, non-converging fits, or physically invalid curves
(non-monotonic, outside [0, 1]) are silently discarded.

Notes
-----
- Returns (None, None) if fewer than 100 valid bootstrap samples remain.
- A warning is raised if fewer than 75% of samples are valid.
- Set return_sf=True to obtain bounds on the survival function (SF) instead of the CDF.
"""
#-----------------------------------------------------------------------------------------------------------------------
# Non-parametric Bootstrap approach to calculate the confidence bounds on reliability / failure probability
#-----------------------------------------------------------------------------------------------------------------------
def _bootstrap_bounds(cdf_fn_orig, fit_fn, data_failures, data_suspensions, xvals, CI, n_bootstrap, return_sf, seed, min_failures):
    """
    Compute pointwise confidence interval bounds via non-parametric bootstrap resampling,
    refitting the model on each resample.

    For each of `n_bootstrap` iterations, resamples all units (failures + suspensions, preserving each unit's
    (time, status) pairing and hence the original censoring structure) with replacement, refits the model via `fit_fn`,
    evaluates the CDF over `xvals`, and discards samples that have too few failures, fail to converge,
    or yield a physically invalid curve (non-monotonic or outside [0, 1]).

    Parameters
    ----------
    cdf_fn_orig : callable
        CDF function f(xvals, *params) -> array, on the original (non-log) parameter scale.
    fit_fn : callable
        fit_fn(failures, suspensions) -> ndarray of fitted parameters, or None if the fit did not converge.
    data_failures : array-like
        Original observed failure times.
    data_suspensions : array-like
        Original suspension times (may be empty).
    xvals : array-like
        Time values at which bounds are evaluated.
    CI : float
        Confidence level (e.g. 0.95).
    n_bootstrap : int
        Number of bootstrap resamples to attempt.
    return_sf : bool
        If True, return survival-function bounds instead of CDF bounds.
    seed : int
        Random seed for reproducibility.
    min_failures : int
        Minimum number of failures a bootstrap resample must contain to be used for fitting
        (model-specific: 2P=2, 3P=3, CR=4, Mixture=5).

    Returns
    -------
    tuple
        (lower, upper): ndarrays of pointwise percentile bounds, or (None, None) if fewer than 500 valid bootstrap
        curves remain (with a UserWarning if the valid fraction is below 75%).
    """
    rng    = np.random.default_rng(seed=seed)
    xvals  = np.asarray(xvals)

    # Build unit pool: (time, status) — 1 = failure, 0 = suspension
    failures_arr    = np.column_stack([data_failures, np.ones(len(data_failures))])
    suspensions_arr = (np.column_stack([data_suspensions, np.zeros(len(data_suspensions))])
                       if len(data_suspensions) > 0 else np.empty((0, 2)))
    all_units = (np.vstack([failures_arr, suspensions_arr])
                 if len(suspensions_arr) > 0 else failures_arr)

    n_units = len(all_units)
    curves  = []

    for _ in range(n_bootstrap):
        idx          = rng.choice(n_units, size=n_units, replace=True)
        boot_sample  = all_units[idx]

        boot_failures    = boot_sample[boot_sample[:, 1] == 1, 0]
        boot_suspensions = boot_sample[boot_sample[:, 1] == 0, 0]

        if len(boot_failures) < min_failures:
            continue

        try:
            params_b = fit_fn(boot_failures, boot_suspensions if len(boot_suspensions) > 0 else None)

            if params_b is None:
                continue

            curve = cdf_fn_orig(xvals, *params_b)

            if (np.any(curve < 0) or np.any(curve > 1) or np.any(np.diff(curve) < 0)):
                continue

            curves.append(curve)

        except Exception:
            continue

    n_valid = len(curves)
    print(f'\n{n_valid} fitted samples to calculate the bootstrap confidence interval.')
    threshold = int(0.75 * n_bootstrap)

    if n_valid < threshold:
        warnings.warn(f"Bootstrap: only {n_valid}/{n_bootstrap} valid samples ({100 * n_valid / n_bootstrap:.1f}% < 75% threshold). "
                      f"Confidence bounds may be unreliable — consider a simpler model.", UserWarning)

    if n_valid < 500:
        warnings.warn(f"Bootstrap: only {n_valid}/{n_bootstrap} valid samples. "
                      f"Confidence bounds may be unreliable — consider more initial samples.", UserWarning)
        return None, None

    curves = np.array(curves)
    alpha_tail = (1.0 - CI) / 2.0

    if return_sf:
        curves = 1.0 - curves

    lower = np.percentile(curves, alpha_tail * 100.0, axis=0)
    upper = np.percentile(curves, (1.0 - alpha_tail) * 100.0, axis=0)

    lower = np.clip(lower, 1e-9, 1 - 1e-9)
    upper = np.clip(upper, 1e-9, 1 - 1e-9)

    return lower, upper


#-----------------------------------------------------------------------------------------------------------------------
# Main function for non-parametric bootstrapping: 2P, 3P, Mixture and Competing Risk
#-----------------------------------------------------------------------------------------------------------------------
def weibull_2p_bootstrap_bounds(xvals, failures, right_censored=None, CI=0.95, n_bootstrap=2000, return_sf=False, seed=42):
    """
    Compute non-parametric bootstrap confidence bounds for a Weibull 2P model's CDF or SF curve,
    by resampling data and refitting via MLE.

    Parameters
    ----------
    xvals : array-like
        Time values for bound evaluation.
    failures : array-like
        Observed failure times.
    right_censored : array-like, optional
        Suspension times.
    CI : float, optional
        Confidence level (default: 0.95).
    n_bootstrap : int, optional
        Number of bootstrap resamples (default: 2000).
    return_sf : bool, optional
        If True, return SF bounds instead of CDF bounds.
    seed : int, optional
        Random seed for reproducibility (default: 42).

    Returns
    -------
    tuple
        (lower, upper): ndarrays of bounds, or (None, None) if fewer than 500 valid bootstrap fits remain.
        Each resample requires at least 2 failures to be fit.
    """
    T_f  = np.asarray(failures)
    T_rc = np.asarray(right_censored) if right_censored is not None else np.array([])

    def fit_fn(f, s):
        res = Fit_Weibull_2P(failures=f, right_censored=s, show_probability_plot=False, print_results=False,
                             method='MLE', CI_type='none', optimizer='best')

        if res.optimizer is None:  # If all optimizer fail --> no fit
            return None

        return np.array([res.alpha, res.beta])

    def cdf_fn(t, alpha, beta):

        return _weibull_cdf(t, alpha, beta)

    return _bootstrap_bounds(cdf_fn_orig=cdf_fn, fit_fn=fit_fn, data_failures=T_f, data_suspensions=T_rc, xvals=xvals,
                             CI=CI, n_bootstrap=n_bootstrap, return_sf=return_sf, seed=seed, min_failures=2)


def weibull_3p_bootstrap_bounds(xvals, failures, right_censored=None, CI=0.95, n_bootstrap=2000, return_sf=False, seed=42):
    """
    Compute non-parametric bootstrap confidence bounds for a Weibull 3P model's CDF or SF curve,
    by resampling data and refitting via MLE.

    Parameters
    ----------
    xvals : array-like
        Time values for bound evaluation.
    failures : array-like
        Observed failure times.
    right_censored : array-like, optional
        Suspension times.
    CI : float, optional
        Confidence level (default: 0.95).
    n_bootstrap : int, optional
        Number of bootstrap resamples (default: 2000).
    return_sf : bool, optional
        If True, return SF bounds instead of CDF bounds.
    seed : int, optional
        Random seed for reproducibility (default: 42).

    Returns
    -------
    tuple
        (lower, upper): ndarrays of bounds, or (None, None) if fewer than 500 valid bootstrap fits remain.
        Each resample requires at least 3 failures to be fit.
    """
    T_f  = np.asarray(failures)
    T_rc = np.asarray(right_censored) if right_censored is not None else np.array([])

    def fit_fn(f, s):
        res = Fit_Weibull_3P(failures=f, right_censored=s, show_probability_plot=False, print_results=False,
                             method='MLE', CI_type='none', optimizer='best')

        if res.optimizer is None:  # If all optimizer fail --> no fit
            return None

        return np.array([res.alpha, res.beta, res.gamma])

    def cdf_fn(t, alpha, beta, gamma):

        return _weibull_3p_cdf(t, alpha, beta, gamma)

    return _bootstrap_bounds(cdf_fn_orig=cdf_fn, fit_fn=fit_fn, data_failures=T_f, data_suspensions=T_rc, xvals=xvals,
                             CI=CI, n_bootstrap=n_bootstrap, return_sf=return_sf, seed=seed, min_failures=3)


def weibull_mixture_bootstrap_bounds(xvals, failures, right_censored=None, CI=0.95, n_bootstrap=2000, return_sf=False, seed=42):
    """
    Compute non-parametric bootstrap confidence bounds for a Weibull Mixture model's CDF or SF curve,
    by resampling data and refitting via MLE.

    Parameters
    ----------
    xvals : array-like
        Time values for bound evaluation.
    failures : array-like
        Observed failure times.
    right_censored : array-like, optional
        Suspension times.
    CI : float, optional
        Confidence level (default: 0.95).
    n_bootstrap : int, optional
        Number of bootstrap resamples (default: 2000).
    return_sf : bool, optional
        If True, return SF bounds instead of CDF bounds.
    seed : int, optional
        Random seed for reproducibility (default: 42).

    Returns
    -------
    tuple
        (lower, upper): ndarrays of bounds, or (None, None) if fewer than 500 valid bootstrap fits remain.
        Each resample requires at least 5 failures to be fit.
    """
    T_f  = np.asarray(failures)
    T_rc = np.asarray(right_censored) if right_censored is not None else np.array([])

    def fit_fn(f, s):
        res = Fit_Weibull_Mixture(failures=f, right_censored=s, show_probability_plot=False, print_results=False,
                                  method='MLE', CI_type='none', optimizer='best')

        if res.optimizer is None:  # If all optimizer fail --> no fit
            print(f'Fit was sorted out successfully. "res.optimizer" = {res.optimizer} works totally fine')
            return None

        return np.array([res.alpha_1, res.beta_1, res.alpha_2, res.beta_2, res.proportion_1])

    def cdf_fn(t, a1, b1, a2, b2, p):

        return _mixture_cdf(t, a1, b1, a2, b2, p)

    return _bootstrap_bounds(cdf_fn_orig=cdf_fn, fit_fn=fit_fn, data_failures=T_f, data_suspensions=T_rc, xvals=xvals,
                             CI=CI, n_bootstrap=n_bootstrap, return_sf=return_sf, seed=seed, min_failures=5)


def weibull_cr_bootstrap_bounds(xvals, failures, right_censored=None, CI=0.95, n_bootstrap=2000, return_sf=False, seed=42):
    """
    Compute non-parametric bootstrap confidence bounds for a Weibull Competing Risks model's CDF or SF curve,
    by resampling data and refitting via MLE.

    Parameters
    ----------
    xvals : array-like
        Time values for bound evaluation.
    failures : array-like
        Observed failure times.
    right_censored : array-like, optional
        Suspension times.
    CI : float, optional
        Confidence level (default: 0.95).
    n_bootstrap : int, optional
        Number of bootstrap resamples (default: 2000).
    return_sf : bool, optional
        If True, return SF bounds instead of CDF bounds.
    seed : int, optional
        Random seed for reproducibility (default: 42).

    Returns
    -------
    tuple
        (lower, upper): ndarrays of bounds, or (None, None) if fewer than 500 valid bootstrap fits remain.
        Each resample requires at least 4 failures to be fit.
    """
    T_f  = np.asarray(failures)
    T_rc = np.asarray(right_censored) if right_censored is not None else np.array([])

    def fit_fn(f, s):
        res = Fit_Weibull_CR(failures=f, right_censored=s, show_probability_plot=False, print_results=False,
                             method='MLE', CI_type='none', optimizer='best')

        if res.optimizer is None:  # If all optimizer fail --> no fit
            print(f'Fit was sorted out successfully. "res.optimizer" = {res.optimizer} works totally fine')
            return None

        return np.array([res.alpha_1, res.beta_1, res.alpha_2, res.beta_2])

    def cdf_fn(t, a1, b1, a2, b2):

        return _cr_cdf(t, a1, b1, a2, b2)

    return _bootstrap_bounds(cdf_fn_orig=cdf_fn, fit_fn=fit_fn, data_failures=T_f, data_suspensions=T_rc, xvals=xvals,
                             CI=CI, n_bootstrap=n_bootstrap, return_sf=return_sf, seed=seed, min_failures=4)
