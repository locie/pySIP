"""Statistic functions module"""
import numpy as np
import scipy as sp
from scipy.signal import correlate


def lrtest(loglik: float, loglik_sub: float, n_par: int, n_par_sub: int, negative: bool = True):
    """Compute the pvalue of the likelihood ratio test

    The likelihood ratio test compares two nested models, M_sub and M,
    such that M_sub ⊂ M, e.g. M_sub can be obtained by setting some
    parameters of M to 0.

    Args:
    loglik: Log-likelihood of the model M
    loglik_sub: Log-likelihood of the sub-model, M_sub
    n_par: Number of free parameters in M
    n_par_sub: Number of free parameters in M_sub
    negative: True if negative log-likelihood is used
    """
    if not isinstance(loglik, float):
        raise TypeError('`loglik` must be a float')

    if not isinstance(loglik_sub, float):
        raise TypeError('`loglik_sub` must be a float')

    if not isinstance(n_par, int):
        raise TypeError('`n_par` must be an integer')

    if not isinstance(n_par_sub, int):
        raise TypeError('`n_par_sub` must be an integer')

    if not isinstance(negative, bool):
        raise TypeError('`negative` must be a boolean')

    if n_par_sub > n_par:
        raise ValueError('The sub-model must have less parameters ' 'than the larger model')

    if negative:
        lrt = loglik - loglik_sub
    else:
        lrt = loglik_sub - loglik

    return sp.stats.chi2.sf(-2.0 * lrt, n_par - n_par_sub)


def aic(loglik: float, n_par: int, negative: bool = True) -> float:
    """Akaike information criterion"""
    if not isinstance(loglik, float):
        raise TypeError('`loglik` must be a float')

    if not isinstance(n_par, int):
        raise TypeError('`n_par` must be an integer')

    if not isinstance(negative, bool):
        raise TypeError('`negative` must be a boolean')

    if negative:
        return 2.0 * n_par + 2.0 * loglik
    return 2.0 * n_par - 2.0 * loglik


def ttest(theta, sigma, N):
    """t-test: statistical signifiance of the maximum likelihood estimates
    :math:`\\hat{\\theta}`

    The following hypothesis is tested:
    :math:`H_0: \\hat{\\theta}=0` against :math:`H_1: \\hat{\\theta} \neq 0`

    Under the null hypothesis :math:`H_0`, the test quantity
    :math:`z=\\frac{\\hat{\\theta}}{\\sigma_{\\hat{\\theta}}}
    follows a t-distribution, centered on the null hypothesis value, with
    :math:`N-N_p` degrees of freedom where N is the sample size and
    :math:`N_p` the number of estimated parameters

    The null hypothesis :math:`H_0`is rejected if the :math:`p_{value}` of the
    test quantity :math:`z` is less or equal to the signifiance level defined.
    """
    if not isinstance(theta, np.ndarray):
        theta = np.array(theta)

    if not isinstance(sigma, np.ndarray):
        sigma = np.array(sigma)

    idx = sigma > 0
    if np.any(idx is False):
        print("Negative standard deviation in the t-test, the p-value is NaN")

    Np = len(theta)
    pvalue = np.full(Np, np.nan)
    pvalue[idx] = 2 * (1 - sp.stats.t.cdf(theta[idx] / sigma[idx], N - Np))

    return pvalue


def ccf(x, y=None, n_lags=None, ci=0.95):
    """Cross correlation (CCF) between two time-series x and y.

    If only one time-series is specified, the auto correlation of x is computed.

    Args:
        x, y: Time series of length N
        n_lags: Number of lags
        ci: confidence interval [0, 1], by default 95%
    """

    if y is None:
        y = x

    N = len(x)
    n_lags = int(n_lags) if n_lags else N - 1

    if len(x) != len(y):
        raise ValueError('x and y must have equal length')

    if n_lags >= N or n_lags < 0:
        raise ValueError(f'maxlags must belong to [1 {n_lags - 1}]')

    lags = np.arange(0, N)

    correlation_coeffs = correlate(x - np.mean(x), y - np.mean(y)) / (np.std(x) * np.std(y) * N)

    cut = range(N - 1, N + n_lags)

    # confidence interval lines
    confidence = np.ones(len(cut)) * sp.stats.norm.ppf((1 + ci) / 2) / np.sqrt(N)

    return lags, correlation_coeffs[cut], confidence


def check_ccf(lags, coeffs, confidence, threshold=0.95):
    """CCF Test
    Given cross-correlation coefficients values and corresponding confidence
    interval values, check that at least a minimum of the absolute ccf values
    fits within the confidence intervals.

    Args:
        lags, coeffs, confidences: outputs of the ccf function
        threshold (float): Threshold level (0.0 - 1.0)

    Returns:
        Boolean. True if the test has succeed, False otherwise.
        in_band: Percentage of value inside the credible intervals
    """
    in_band = np.sum(np.abs(coeffs) < confidence) / len(coeffs)
    return in_band >= threshold, in_band


def cpgram(ts):
    """Cumulative periodogram with 95% confidence intervals

    Args:
        ts: residual time series

    Notes:
        Adapted from the `R cpgram function
        <https://www.rdocumentation.org/packages/stats/versions/3.5.3/topics/cpgram>`_
    """
    spectrum = np.fft.fft(ts)
    n = len(ts)
    y = (np.sqrt(spectrum.real ** 2 + spectrum.imag ** 2)) ** 2 / n
    if n % 2 == 0:
        n -= 1
        y = y[:n]

    freq = np.linspace(0, 0.5, n, endpoint=True)
    crit = 1.358 / (np.sqrt(n) + 0.12 + 0.11 / np.sqrt(n))

    return y, freq, crit


def check_cpgram(y, freq, crit, threshold=0.95):
    """Cpgram test

    Given a spectrum and a criterion, check that the cumulative periodogram
    values fits within 95% confidence intervals

    Args:
        y, freq, crit: outputs of the cpgram function
        threshold (float): Threshold level (0.0 - 1.0)

    Returns:
        Boolean. True if the test has succeed, False otherwise.
        in_band: Percentage of value inside the credible intervals
    """
    cum_sum = np.cumsum(y) / np.sum(y) - 2 * freq
    in_band = np.sum(np.abs(cum_sum) < crit) / len(freq)
    return in_band >= threshold, in_band
