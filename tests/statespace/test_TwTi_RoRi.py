import pytest
import numpy as np

from bopt.statespace.rc import TwTi_RoRi

names = ["Ro", "Ri", "Cw", "Ci", "sigw_w", "sigw_i", "sigv",
         "x0_w", "x0_i", "sigx0_w", "sigx0_i"]

Ro = 0.018653309141725055
Ri = 0.001998181006046002
Cw = 0.14815664757840966
Ci = 0.016517941708752162
sigw_w = 0.0026574397809898996
sigw_i = 0.0
sigv = 0.04884197506321087
x0_w = 0.26572048305578333
x0_i = 0.2670106194217502
sigx0_w = 1.0
sigx0_i = 1.0

theta = [Ro, Ri, Cw, Ci, sigw_w, sigw_i, sigv,
         x0_w, x0_i, sigx0_w, sigx0_i]

dt = 30 * 60
hold = 'foh'
Nx = 2
Nu = 2
Ny = 1
rtol = 1e-4


@pytest.fixture
def r2c2():
    ss = TwTi_RoRi(hold_order=hold)
    ss.parameters.theta = theta
    return ss


def test_creation(r2c2):
    A = np.array([[-3.73972388e-05, 3.37787855e-05],
                  [3.02976709e-04, -3.02976709e-04]])

    B = np.array([[3.61845328e-06, 0.00000000e+00],
                  [0.00000000e+00, 6.05402306e-07]])

    C = np.array([[0., 1.]])

    D = np.array([[0., 0.]])

    Q = np.array([[0.00265744, 0.],
                  [0., 0.]])

    R = np.array([[0.04884198]])

    x0 = np.array([[26.57204831], [26.70106194]])

    P0 = np.array([[1., 0.],
                   [0., 1.]])

    dA = {k: np.zeros((Nx, Nx)) for k in names}
    dB = {k: np.zeros((Nx, Nu)) for k in names}
    dC = {k: np.zeros((Ny, Nx)) for k in names}
    dD = {k: np.zeros((Ny, Nu)) for k in names}
    dQ = {k: np.zeros((Nx, Nx)) for k in names}
    dR = {k: np.zeros((Ny, Ny)) for k in names}
    dx0 = {k: np.zeros((Nx, 1)) for k in names}
    dP0 = {k: np.zeros((Nx, Nx)) for k in names}

    dA['Ro'] = np.array([[0.00019398, 0.],
                         [0., 0.]])

    dA['Ri'] = np.array([[0.01690477, -0.01690477],
                         [-0.15162626, 0.15162626]])

    dA['Cw'] = np.array([[0.00025242, -0.00022799],
                         [0., 0.]])

    dA['Ci'] = np.array([[0., 0.],
                         [-0.01834228, 0.01834228]])

    dB['Ro'] = np.array([[-0.00019398, 0.],
                         [0., 0.]])

    dB['Cw'] = np.array([[-2.44231585e-05, 0.00000000e+00],
                         [0.00000000e+00, 0.00000000e+00]])

    dB['Ci'] = np.array([[0.00000000e+00, 0.00000000e+00],
                         [0.00000000e+00, -3.66511952e-05]])

    dQ["sigw_w"][0, 0] = 1.

    dQ["sigw_i"][1, 1] = 1.

    dR["sigv"][0, 0] = 1.

    dx0["x0_w"][0, 0] = 100.

    dx0["x0_i"][1, 0] = 100.

    dP0["sigx0_w"][0, 0] = 1.

    dP0["sigx0_i"][1, 1] = 1.

    Ad_truth = np.array([[0.94823544, 0.04543517],
                         [0.40752788, 0.59100983]])

    B0d_truth = np.array([[6.32939203e-03, 2.72764212e-05],
                          [1.46228323e-03, 8.44512804e-04]])

    B1d_truth = np.array([[5.6461563, 0.03192978],
                          [1.71174872, 0.69361468]])

    Qd_truth = np.array([[-0.10957645, -0.02510687],
                         [0., -0.01360979]])

    dAd_truth = np.array([[[3.29664016e-01, 8.56057581e-03],
                           [7.67835506e-02, 1.43338399e-03]],

                          [[1.64663450e+01, -1.65328783e+01],
                           [-1.48290621e+02, 1.48888821e+02]],

                          [[3.38720812e-01, -2.97215250e-01],
                           [8.48017612e-02, -7.51345115e-02]],

                          [[-6.73914314e-01, 6.75532999e-01],
                           [-1.86126810e+01, 1.86866641e+01]],

                          [[0.00000000e+00, 0.00000000e+00],
                           [0.00000000e+00, 0.00000000e+00]],

                          [[0.00000000e+00, 0.00000000e+00],
                           [0.00000000e+00, 0.00000000e+00]],

                          [[0.00000000e+00, 0.00000000e+00],
                           [0.00000000e+00, 0.00000000e+00]],

                          [[0.00000000e+00, 0.00000000e+00],
                           [0.00000000e+00, 0.00000000e+00]],

                          [[0.00000000e+00, 0.00000000e+00],
                           [0.00000000e+00, 0.00000000e+00]],

                          [[0.00000000e+00, 0.00000000e+00],
                           [0.00000000e+00, 0.00000000e+00]],

                          [[0.00000000e+00, 0.00000000e+00],
                           [0.00000000e+00, 0.00000000e+00]]])

    dB0d_truth = np.array([[[-3.38224592e-01, 3.27857040e-06],
                            [-7.82169346e-02, 4.14409731e-07]],

                           [[6.65333680e-02, -1.11584147e-02],
                            [-5.98200277e-01, 1.00324938e-01]],

                           [[-4.15055613e-02, -1.80326197e-04],
                            [-9.66724970e-03, -3.01938433e-05]],

                           [[-1.61868562e-03, -1.38003105e-03],
                            [-7.39831759e-02, -3.87193684e-02]],

                           [[0.00000000e+00, 0.00000000e+00],
                            [0.00000000e+00, 0.00000000e+00]],

                           [[0.00000000e+00, 0.00000000e+00],
                            [0.00000000e+00, 0.00000000e+00]],

                           [[0.00000000e+00, 0.00000000e+00],
                            [0.00000000e+00, 0.00000000e+00]],

                           [[0.00000000e+00, 0.00000000e+00],
                            [0.00000000e+00, 0.00000000e+00]],

                           [[0.00000000e+00, 0.00000000e+00],
                            [0.00000000e+00, 0.00000000e+00]],

                           [[0.00000000e+00, 0.00000000e+00],
                            [0.00000000e+00, 0.00000000e+00]],

                           [[0.00000000e+00, 0.00000000e+00],
                            [0.00000000e+00, 0.00000000e+00]]])

    dB1d_truth = np.array([[[-3.01384004e+02, 4.37845723e-03],
                            [-9.15317628e+01, 5.90762046e-04]],

                           [[7.55954557e+01, -1.26832111e+01],
                            [-6.79944295e+02, 1.14079009e+02]],

                           [[-3.66868244e+01, -2.10510628e-01],
                            [-1.12854307e+01, -3.99296995e-02]],

                           [[-2.14062284e+00, -1.57422205e+00],
                            [-8.43937147e+01, -2.78326024e+01]],

                           [[0.00000000e+00, 0.00000000e+00],
                            [0.00000000e+00, 0.00000000e+00]],

                           [[0.00000000e+00, 0.00000000e+00],
                            [0.00000000e+00, 0.00000000e+00]],

                           [[0.00000000e+00, 0.00000000e+00],
                            [0.00000000e+00, 0.00000000e+00]],

                           [[0.00000000e+00, 0.00000000e+00],
                            [0.00000000e+00, 0.00000000e+00]],

                           [[0.00000000e+00, 0.00000000e+00],
                            [0.00000000e+00, 0.00000000e+00]],

                           [[0.00000000e+00, 0.00000000e+00],
                            [0.00000000e+00, 0.00000000e+00]],

                           [[0.00000000e+00, 0.00000000e+00],
                            [0.00000000e+00, 0.00000000e+00]]])

    dQd_truth = np.array([[[-1.87502260e-02, -4.36822862e-03],
                           [0.00000000e+00, -2.59883582e-05]],

                          [[-1.14346346e+00, 1.02127463e+01],
                           [0.00000000e+00, 5.05502731e+00]],

                          [[-2.08677955e-02, -4.86765440e-03],
                           [0.00000000e+00, -2.70236996e-04]],

                          [[2.76730533e-02, 1.27416657e+00],
                           [0.00000000e+00, 6.13902892e-01]],

                          [[-4.12338422e+01, -9.44776690e+00],
                           [0.00000000e+00, -5.12139109e+00]],

                          [[0.00000000e+00, 0.00000000e+00],
                           [0.00000000e+00, 0.00000000e+00]],

                          [[0.00000000e+00, 0.00000000e+00],
                           [0.00000000e+00, 0.00000000e+00]],

                          [[0.00000000e+00, 0.00000000e+00],
                           [0.00000000e+00, 0.00000000e+00]],

                          [[0.00000000e+00, 0.00000000e+00],
                           [0.00000000e+00, 0.00000000e+00]],

                          [[0.00000000e+00, 0.00000000e+00],
                           [0.00000000e+00, 0.00000000e+00]],

                          [[0.00000000e+00, 0.00000000e+00],
                           [0.00000000e+00, 0.00000000e+00]]])

    r2c2.update()
    _, Ad, B0d, B1d, Qd, dAd, dB0d, dB1d, dQd = r2c2.discretization(dt)

    np.testing.assert_allclose(r2c2.A, A, rtol)
    np.testing.assert_allclose(r2c2.B, B, rtol)
    np.testing.assert_allclose(r2c2.C, C, rtol)
    np.testing.assert_allclose(r2c2.D, D, rtol)
    np.testing.assert_allclose(r2c2.Q, Q, rtol)
    np.testing.assert_allclose(r2c2.R, R, rtol)
    np.testing.assert_allclose(r2c2.x0, x0, rtol)
    np.testing.assert_allclose(r2c2.P0, P0, rtol)
    for n in names:
        np.testing.assert_allclose(r2c2.dA[n], dA[n], rtol)
        np.testing.assert_allclose(r2c2.dB[n], dB[n], rtol)
        np.testing.assert_allclose(r2c2.dC[n], dC[n], rtol)
        np.testing.assert_allclose(r2c2.dD[n], dD[n], rtol)
        np.testing.assert_allclose(r2c2.dQ[n], dQ[n], rtol)
        np.testing.assert_allclose(r2c2.dR[n], dR[n], rtol)
        np.testing.assert_allclose(r2c2.dx0[n], dx0[n], rtol)
        np.testing.assert_allclose(r2c2.dP0[n], dP0[n], rtol)

    np.testing.assert_allclose(Ad.squeeze(), Ad_truth, rtol)
    np.testing.assert_allclose(B0d.squeeze(), B0d_truth, rtol)
    np.testing.assert_allclose(B1d.squeeze(), B1d_truth, rtol)
    np.testing.assert_allclose(Qd.squeeze(), Qd_truth, rtol)
    np.testing.assert_allclose(dAd.squeeze(), dAd_truth, rtol)
    np.testing.assert_allclose(dB0d.squeeze(), dB0d_truth, rtol)
    np.testing.assert_allclose(dB1d.squeeze(), dB1d_truth, rtol)
    np.testing.assert_allclose(dQd.squeeze(), dQd_truth, rtol)
