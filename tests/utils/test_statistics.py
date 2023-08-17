import random

import numpy as np
import pytest

from pysip.utils.statistics import (
    ccf,
    check_ccf,
    check_cpgram,
    cpgram,
)


def test_ccf():
    lags, correlation_coeffs, confidence = ccf([1, 2, 3], [4, 5, 6])
    np.testing.assert_array_equal(lags, np.array([0, 1, 2]))
    np.testing.assert_array_equal(correlation_coeffs, np.array([1.0, 0.0, -0.5]))
    np.testing.assert_allclose(
        confidence, np.array([1.13158573, 1.13158573, 1.13158573])
    )


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


def test_cpgram():
    y, freq, crit = cpgram([random.random() for i in range(1400)])

    assert isinstance(y, np.ndarray) and len(y) == 1399
    assert isinstance(freq, np.ndarray) and len(freq) == 1399
    assert crit == pytest.approx(0.0361880988)


def test_check_cpgram():
    y, freq, crit = cpgram([random.random() for i in range(1400)])

    assert not check_cpgram(y, freq, crit)[0]
