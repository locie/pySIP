import itertools

import numpy as np
from pylfsr import LFSR


def prbs(Tmax, Tmin, initstate="random"):
    """Pseudo Random Binary signal (PRBS)

    Args:
        Tmax: maximum number of samples in one state
        Tmin: minimum number of samples in one state
        initstate: initial state of the linear feedback state register
            'ones': binary numpy array of ones, dimension = (length register,)
            'random': random binary numpy array, dimension = (length register,)

    Returns:
        PRBS as a numpy array

    Notes:
        The Linear Feedback Shift Register (LFSR) can be installed
        from PyPi: https://pypi.org/project/pylfsr/
        or from the source:
        https://github.com/Nikeshbajaj/Linear_Feedback_Shift_Register
    """
    if not isinstance(Tmax, int):
        raise TypeError("`Tmax` must be an integer")

    if Tmax < 2:
        raise ValueError("`Tmax` must be > 2")

    if not isinstance(Tmin, int):
        raise TypeError("`Tmax` must be an integer")

    if Tmin < 1:
        raise ValueError("`Tmin` must be > 1")

    if Tmin >= Tmax:
        raise ValueError("`Tmax` must be strictly superior to `Tmin`")

    __init_availabble__ = ["random", "ones"]
    if initstate not in __init_availabble__:
        raise ValueError(f"`initstate` must be either {__init_availabble__}")

    # get the register length
    n = np.ceil(Tmax / Tmin)
    if n < 2 or n > 31:
        raise ValueError(
            "The PRBS cannot be generated, " "decompose the signal in two sequences"
        )

    # Linear feedback register up to 32 bits
    fpoly = {
        2: [2, 1],
        3: [3, 1],
        4: [4, 1],
        5: [5, 2],
        6: [6, 1],
        7: [7, 1],
        8: [8, 4, 3, 2],
        9: [9, 4],
        10: [10, 3],
        11: [11, 2],
        12: [12, 6, 4, 1],
        13: [13, 4, 3, 1],
        14: [14, 8, 6, 1],
        15: [15, 1],
        16: [16, 12, 3, 1],
        17: [17, 3],
        18: [18, 7],
        19: [19, 5, 2, 1],
        20: [20, 3],
        21: [21, 2],
        22: [22, 1],
        23: [23, 5],
        24: [24, 7, 2, 1],
        25: [25, 3],
        26: [26, 6, 2, 1],
        27: [27, 5, 2, 1],
        28: [28, 3],
        29: [29, 2],
        30: [30, 23, 2, 1],
        31: [31, 3],
    }

    L = LFSR(fpoly=fpoly[n], initstate=initstate, verbose=False)

    seq = []
    for n in range(L.expectedPeriod):
        L.next()
        seq.append(L.state[0])

    seq_padded = np.repeat(seq, Tmin)

    # check generated PRBS
    assert seq_padded.shape[0] == L.expectedPeriod * Tmin
    assert max(len(list(v)) for g, v in itertools.groupby(seq_padded)) == Tmax
    assert min(len(list(v)) for g, v in itertools.groupby(seq_padded)) == Tmin

    return seq_padded


def generate_time(n=50, bounds=(0, 50), random=False):
    """Generates time array

    Args:
        n (int): Number of time instants to generate
        bounds (tuple): t ∈ [bounds[0], bounds[1]]
        random (bool):
            True: random
            False: linear time spacing
        period (float): Sinusoidal period
        amplitude (float): Sinuoidal amplitude
        noise_std (float): Gaussian noise standard deviation

    Returns:
        t (np.array): Time data
    """
    if not isinstance(n, int):
        raise ValueError("`n` must be an integer")

    if not isinstance(bounds, tuple):
        raise ValueError("`bounds` must be a tuple")

    bounds = np.asarray(bounds)
    if bounds[1] <= bounds[0]:
        raise ValueError("upper bound > lower bound")

    if not isinstance(random, bool):
        raise ValueError("`random` must be a boolean")

    if random:
        t = np.sort(bounds[0] + (bounds[1] - bounds[0]) * np.random.rand(n))
    else:
        t = np.linspace(bounds[0], bounds[1], num=n)

    return t


def generate_sine(
    n=50,
    bounds=(0, 50),
    random=False,
    period=24.0,
    phase=0.0,
    amplitude=10.0,
    offset=0.0,
    noise_std=1.0,
    clip_to_0=False,
):
    """Generate sinusoidal data

    Args:
        n (int): Number of time instants to generate
        bounds (tuple): t ∈ [bounds[0], bounds[1]]
        random (bool):
            True: random
            False: linear time spacing
        period (float): Sinusoidal period > 0
        phase (float): Phase shift [0, 2 * np.pi]
        amplitude (float): Sinuoidal amplitude > 0
        offset (float): Sinusoidal mean offset
        noise_std (float): Gaussian noise standard deviation > 0
        clip_to_0 (bool): Set negative values to 0

    Returns:
        t (np.array): Time data
        y (np.array): Sinusoidal data
    """
    if not isinstance(period, float) or period <= 0.0:
        raise ValueError("`period` must be a positive float")

    if not isinstance(amplitude, float) or amplitude <= 0.0:
        raise ValueError("`amplitude` must be a positive float")

    if not isinstance(noise_std, float) or noise_std < 0.0:
        raise ValueError("`noise_std` must be a positive float")

    if not isinstance(phase, float) or phase < 0.0:
        raise ValueError("`phase` must be a positive float")

    if not isinstance(offset, float):
        raise ValueError("`offset` must be a float")

    t = generate_time(n, bounds, random)
    y = (
        offset
        + amplitude * np.sin(2.0 * np.pi / period * t + phase)
        + noise_std * np.random.randn(t.shape[0])
    )

    if clip_to_0:
        y[y < 0] = 0

    return t, y


def generate_random_binary(n: int = 50, bounds: tuple = (0, 1e3)):
    """Generate random binary signal

    Args:
        n: Number of time instants to generate
        bounds: Binary signal ∈ [bounds[0], bounds[1]]

    Returns:
       Random binary signal
    """
    if not isinstance(n, int):
        raise ValueError("`n` must be an integer")

    if not isinstance(bounds, tuple):
        raise ValueError("`bounds` must be a tuple")

    bounds = np.asarray(bounds)
    if bounds[1] <= bounds[0]:
        raise ValueError("upper bound > lower bound")

    return np.asarray(
        [bounds[1] if np.random.randn() > 0 else bounds[0] for _ in range(n)]
    )
