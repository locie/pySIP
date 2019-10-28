"""Utility functions"""
import time
import numpy as np


def random_seed():
    '''Generate random seed according to the time'''
    t = round(time.time() * 1000.0)
    return (
        ((t & 0xFF000000) >> 24)
        + ((t & 0x00FF0000) >> 8)
        + ((t & 0x0000FF00) << 8)
        + ((t & 0x000000FF) << 24)
    )


def array_to_dict(chains: np.ndarray, names: list) -> dict:
    """Convert an array of size (n_chains, n_par, n_draws) to a dictionary of
    n_par keys where the correponsind values are array of shape (n_chains, n_draws)

    Args:
        chains: Array of chains (n_chains, n_par, n_draws)
        names: Parameter names list of size n_par

    Returns:
        Dictionary of Markov chains organized by parameter names
    """
    if len(names) != chains.shape[1]:
        raise ValueError('The length of `names` does not match`chains.shape[1]`')

    return {k: chains[:, i, :] for i, k in enumerate(names)}


def dict_to_array(chains: dict, names: list) -> np.ndarray:
    """Convert chains[Np keys](chain, draw) to ndarray (Np, chain * draw)

    Args:
        chains: Dictionary of Markov Chain traces, chains[key](chain, draw)
        names: Parameter names to get in chains.keys()

    Returns:
        np.ndarray of Markov Chain traces (Np, chain * draw)
    """
    return np.asarray([v.ravel() for k, v in chains.items() if k in names])
