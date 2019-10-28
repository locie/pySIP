import random
import matplotlib.axes._subplots
import matplotlib.pyplot as plt
import numpy as np
import pytest

from pysip.utils import ccf, check_ccf, cpgram, check_cpgram, plot_ccf, plot_cpgram


@pytest.mark.skip(reason="todo")
def test_ttest():
    assert False


def test_ccf():
    lags, correlation_coeffs, confidence = ccf([1, 2, 3], [4, 5, 6])
    np.testing.assert_array_equal(lags, np.array([0, 1, 2]))
    np.testing.assert_array_equal(correlation_coeffs, np.array([1.0, 0.0, -0.5]))
    np.testing.assert_allclose(confidence, np.array([1.13158573, 1.13158573, 1.13158573]))


def test_ccf_raises():
    with pytest.raises(ValueError):
        ccf([1, 2, 3], [])

    with pytest.raises(ValueError):
        ccf([1, 2, 3], [4, 5, 6], n_lags=-2)

    with pytest.raises(ValueError):
        ccf([1, 2, 3], [4, 5, 6], n_lags=20)


def test_check_ccf():
    lags, correlation_coeffs, confidence = ccf([1, 2, 3], [4, 5, 6])
    assert check_ccf(lags, correlation_coeffs, confidence)[0]


def test_plot_ccf():
    lags, correlation_coeffs, confidence = ccf([1, 2, 3], [4, 5, 6])
    n_figs = plt.gcf().number
    ax = plot_ccf(lags, correlation_coeffs, confidence)

    assert isinstance(ax, matplotlib.axes.Axes)
    assert n_figs + 1 == plt.gcf().number


def test_cpgram():
    y, freq, crit = cpgram([random.random() for i in range(1400)])

    assert isinstance(y, np.ndarray) and len(y) == 1399
    assert isinstance(freq, np.ndarray) and len(freq) == 1399
    assert crit == pytest.approx(0.0361880988)


def test_check_cpgram():
    y, freq, crit = cpgram([random.random() for i in range(1400)])

    assert not check_cpgram(y, freq, crit)[0]


def test_plot_cpgram():
    y, freq, crit = cpgram([random.random() for i in range(1400)])

    n_figs = plt.gcf().number
    ax = plot_cpgram(y, freq, crit)

    assert isinstance(ax, matplotlib.axes.Axes)
    assert n_figs + 1 == plt.gcf().number


@pytest.mark.skip(reason="todo")
def test_autocorrf():
    assert False


@pytest.mark.skip(reason="todo")
def autocovf():
    assert False


@pytest.mark.skip(reason="todo")
def likelihood_ratio_test():
    assert False
