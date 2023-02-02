from collections import namedtuple

import numpy as np

s = []


def stack(f):
    s.append(f)
    return f


@stack
def eigen_values(model, summary, corr):
    """All the eigenvalues real parts of the state matrix A are positive"""
    return (-np.linalg.eigvals(model.ss.A).real > 0).all()


@stack
def summary(model, summary, corr):
    """All ??? are less than 10^-4"""
    return (summary["|g(\u03B7)|"] < 1e-4).all()


@stack
def summary_penalty(model, summary, corr):
    """??"""
    return (summary["|dpen(\u03B8)|"] < summary["|g(\u03B7)|"]).all()


@stack
def summary_pvalue(model, summary, corr):
    """P-value"""
    return (summary['pvalue'] <= 5e-2).all()


@stack
def corr(model, summary, corr):
    """Correlation values are all less than 0.9"""
    non_diag = np.eye(corr.shape[0], dtype=bool)
    return (np.ma.masked_array(corr, mask=non_diag) < 0.9).all()


Test = namedtuple('Test', 'name description function')
tests = [Test(f.__name__, f.__doc__, f) for f in s]


def check_model(model, summary=None, corr=None, verbose=False, raise_error=False):
    """
    Model diagnostic

    Check that a model fitting process has led to coherent results.

    Args:
        model: The model to check
        verbose: verbose mode
        raise_error: in case of error, raise an Exception

    Returns:
        Boolean. False if any test has failed, True otherwise.
    """

    def _print(s):
        if verbose:
            print(s)

    diag = True
    _print('')

    for idx, test in enumerate(tests):
        result = test.function(model, summary, corr)
        diag &= result
        _print(('\033[92m' if result else '\033[91m') + f'{idx}. {test.description}' + '\033[0m')
        if raise_error:
            raise Exception(f'Failed {test.name}')

    return diag
