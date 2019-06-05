import pytest
import numpy as np
from dataclasses import dataclass
from bopt.statespace.base import StateSpace


names = ['a', 'b']
Nx = 2
Nu = 3
Ny = 1
dt = 1
hold = 'foh'


@dataclass
class dummy(StateSpace):
    def __post_init__(self):
        self. _names = names
        self.Nx = Nx
        self.Nu = Nu
        self.Ny = Ny
        self.hold_order = hold
        super().__post_init__()


@pytest.fixture
def state_space():
    ss = dummy()
    return ss


def test_creation(state_space):
    assert state_space.A.shape == (Nx, Nx)
    assert state_space.B.shape == (Nx, Nu)
    assert state_space.C.shape == (Ny, Nx)
    assert state_space.D.shape == (Ny, Nu)
    assert state_space.Q.shape == (Nx, Nx)
    assert state_space.R.shape == (Ny, Ny)
    assert state_space.x0.shape == (Nx, 1)
    assert state_space.P0.shape == (Nx, Nx)
    for n in names:
        assert state_space.dA[n].shape == (Nx, Nx)
        assert state_space.dB[n].shape == (Nx, Nu)
        assert state_space.dC[n].shape == (Ny, Nx)
        assert state_space.dD[n].shape == (Ny, Nu)
        assert state_space.dQ[n].shape == (Nx, Nx)
        assert state_space.dR[n].shape == (Ny, Ny)
        assert state_space.dx0[n].shape == (Nx, 1)
        assert state_space.dP0[n].shape == (Nx, Nx)


def test_discretization(state_space):
    np.random.seed(0)
    state_space.A = np.random.random((Nx, Nx))

    np.random.seed(12)
    state_space.B = np.random.random((Nx, Nu))

    np.random.seed(24)
    state_space.Q = np.random.random((Nx, Nx))

    np.random.seed(36)
    state_space.dA['a'] = np.random.random((Nx, Nx))

    np.random.seed(48)
    state_space.dA['b'] = np.random.random((Nx, Nx))

    np.random.seed(60)
    state_space.dB['a'] = np.random.random((Nx, Nu))

    np.random.seed(72)
    state_space.dB['b'] = np.random.random((Nx, Nu))

    np.random.seed(84)
    state_space.dQ['a'] = np.random.random((Nx, Nx))

    np.random.seed(96)
    state_space.dQ['b'] = np.random.random((Nx, Nx))

    (idx, Ad, B0d, B1d, Qd,
     dAd, dB0d, dB1d, dQd) = state_space.discretization(dt)

    Ad_truth = np.array([[2.1174382, 1.32642097],
                         [1.11791089, 2.11014886]])

    B0d_truth = np.array([[0.5104434, 1.07628812, 0.87568465],
                          [0.83906776, 0.35740309, 1.44338833]])

    B1d_truth = np.array([[0.32690548, 0.60695049, 0.5610481],
                          [0.48174635, 0.2474188, 0.82859612]])

    Qd_truth = np.array([[-2.4570578, -1.71317024],
                         [0., -0.41906132]])

    dAd_truth = np.array([[[2.43511305, 1.91847834],
                           [2.44247172, 1.34511668]],
                          [[0.76121829, 2.02263513],
                           [0.83938153, 1.28124623]]])

    dB0d_truth = np.array([[[1.28128284, 1.21519526, 1.51936157],
                            [1.45248363, 1.5729971, 1.33256383]],
                           [[0.79648836, 1.38316173, 1.85195884],
                            [0.85535487, 1.13013016, 1.56275902]]])

    dB1d_truth = np.array([[[0.85062693, 0.83120937, 1.02224727],
                            [0.89964907, 1.00052096, 0.87915506]],
                           [[0.54348286, 0.84433146, 1.20262266],
                            [0.53500757, 0.71508781, 0.99368403]]])

    dQd_truth = np.array([[[-2.6452396, -2.94312339],
                           [0., 0.1700863]],
                          [[-2.80646159, -3.04089886],
                           [0., -0.08180627]]])

    np.testing.assert_allclose(Ad.squeeze(), Ad_truth)
    np.testing.assert_allclose(B0d.squeeze(), B0d_truth)
    np.testing.assert_allclose(B1d.squeeze(), B1d_truth)
    np.testing.assert_allclose(Qd.squeeze(), Qd_truth)
    np.testing.assert_allclose(dAd.squeeze(), dAd_truth)
    np.testing.assert_allclose(dB0d.squeeze(), dB0d_truth)
    np.testing.assert_allclose(dB1d.squeeze(), dB1d_truth)
    np.testing.assert_allclose(dQd.squeeze(), dQd_truth)
