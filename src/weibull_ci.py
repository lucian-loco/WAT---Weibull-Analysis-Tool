#!/usr/bin/python3
import warnings
import numpy as np
import autograd
import autograd.numpy as anp
from reliability.Fitters import Fit_Weibull_Mixture, Fit_Weibull_CR, Fit_Weibull_2P, Fit_Weibull_3P


# ToDo: Think of adjusting the number of xvals automated to the range of the failures, whats the best case?
"""
Fisher Matrix based confidence intervals for 
Weibull Mixture and Weibull Competing Risks
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
This approach is equivalent to the classical analytical Fisher-matrix method (Delta method), but
avoids explicit gradient derivation of the composite CDF. This makes it directly applicable to
multi-parameter models such as Weibull Mixture and Weibull Competing Risks, where closed-form
derivatives are difficult to derive. Censored observations are correctly accounted for through
the likelihood function used to compute the Hessian.
"""
#-----------------------------------------------------------------------------------------------------------------------
# Help functions
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
    """Transforms CDF values to the Weibull linearization scale (u-scale)"""
    F = np.clip(F, 1e-9, 1 - 1e-9)
    return np.log(-np.log(1.0 - F))


def _u_inverse(u):
    """Back-transforms values from the Weibull linearization scale (u-scale) to CDF space"""
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
                    Negative log-likelihood function f(params) -> float.
                    Must use autograd.numpy operations to support automatic differentiation.
    params        : ndarray, dtype=float
                    MLE parameter vector. Must be a float array for autograd compatibility.

    Returns
    -------
    cov : ndarray, shape (n_params, n_params), or None
          Estimated covariance matrix of the parameter estimates.
          Returns None with a UserWarning if the Hessian is singular, the covariance
          matrix has negative diagonal entries, or contains NaN values.
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
    returns the pointwise CI percentile bounds. For CDF bounds, percentiles are
    computed on the Weibull linearization scale (u = log(-log(1-F))) and
    back-transformed to CDF space to reduce distortion from the nonlinearity
    of the Weibull CDF. For mixture models (5 parameters), samples with a
    proportion outside [0, 1] are discarded as physically invalid.

    Parameters
    ----------
    cdf_fn    : callable
                CDF function with signature f(t, *params) -> array.
                Must accept log-scale parameters if sampling is done on log-scale.
    params    : ndarray
                MLE parameter vector on the sampling scale (e.g. log-scale for alpha, beta).
    cov       : ndarray or None
                Covariance matrix of the parameter estimates. If None, (None, None)
                is returned immediately.
    xvals     : array-like
                Time values at which the bounds are evaluated.
    CI        : float
                Confidence level (e.g. 0.95 for a 95% CI).
    n_samples : int
                Number of Monte Carlo samples.
    return_sf : bool
                If True, returns bounds for the survival function (SF = 1 - CDF)
                instead of the CDF. SF bounds are computed directly via percentiles
                without u-transformation.
    seed      : int
                Random seed for reproducibility.

    Returns
    -------
    lower : ndarray, Lower confidence bound on the CDF (or SF if return_sf=True) scale.
    upper : ndarray, Upper confidence bound on the CDF (or SF if return_sf=True) scale.
    Returns (None, None) if cov is None or no valid samples remain after filtering.
    """
    # If cov contains NaN then just return None, None as upper and lower --> no calculation of the CI
    if cov is None:
        return None, None

    rng = np.random.default_rng(seed=seed)
    samples = rng.multivariate_normal(params, cov, size=n_samples)

    if len(params) == 5:
        valid = (samples[:, 4] >= 0) & (samples[:, 4] <= 1)
        samples = samples[valid]

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


#-----------------------------------------------------------------------------------------------------------------------
# Main functions
#-----------------------------------------------------------------------------------------------------------------------
def weibull_mixture_fisher_bounds(fit, xvals, failures, right_censored=None, CI=0.95, n_samples=10000, return_sf=False, seed=42):
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

    Returns
    -------
    lower : ndarray, shape (len(xvals),)
    upper : ndarray, shape (len(xvals),)
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

    Returns
    -------
    lower : array, shape (len(xvals),)
    upper : array, shape (len(xvals),)
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
    Computes confidence intervals for a Weibull 3P model using Fisher-matrix-based
    Monte Carlo sampling.

    The covariance matrix is derived analytically via automatic differentiation
    (autograd) of the negative log-likelihood. Parameter sampling is performed on
    a mixed scale: log-scale for (ln(alpha), ln(beta)) to ensure positivity, and
    linear scale for gamma (location parameter). Percentiles are evaluated on the
    Weibull linearization scale (u-scale) before back-transformation to CDF space.

    Parameters
    ----------
    fit           : Fit_Weibull_3P
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
        seed=seed
    )