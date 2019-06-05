import numpy as np
from scipy.linalg import expm, expm_frechet
from bopt.statespace.rc import TwTi_RoRi
from bopt.regressors import Regressor


parameters = [
    dict(name="Ro", value=1e-2),
    dict(name="Ri", value=1e-3),
    dict(name="Cw", value=1e-2),
    dict(name="Ci", value=1e-3),
    dict(name="sigw_w", value=0.1),
    dict(name="sigw_i", value=0.2),
    dict(name="sigv", value=0.1),
    dict(name="x0_w", value=0.28),
    dict(name="x0_i", value=0.25),
    dict(name="sigx0_w", value=1.0),
    dict(name="sigx0_i", value=1.0)
]

reg = Regressor(TwTi_RoRi(parameters))
reg.ss.jacobian = False
reg.ss.update()

dt = 600
hold = 'foh'
Ad, B0d, B1d, Qd = reg.ss.discretization(dt, hold)

AA = np.zeros((reg.ss.Nx + reg.ss.Nu, reg.ss.Nx + reg.ss.Nu))
AA[:reg.ss.Nx, :reg.ss.Nx] = reg.ss.A
AA[:reg.ss.Nx, reg.ss.Nx:] = reg.ss.B
AAd = expm(AA * dt)
Ad2 = AAd[:reg.ss.Nx, :reg.ss.Nx]
B0d2 = AAd[:reg.ss.Nx, reg.ss.Nx:]

assert np.allclose(Ad, Ad2)
assert np.allclose(B0d, B0d2)
print('\nTest zero order hold: OK')

AA = np.zeros((reg.ss.Nx + 2 * reg.ss.Nu, reg.ss.Nx + 2 * reg.ss.Nu))
AA[:reg.ss.Nx, :reg.ss.Nx] = reg.ss.A
AA[:reg.ss.Nx, reg.ss.Nx:reg.ss.Nx + reg.ss.Nu] = reg.ss.B
AA[:reg.ss.Nx, -reg.ss.Nu:] = reg.ss.B
AA[reg.ss.Nx:reg.ss.Nx + reg.ss.Nu, -reg.ss.Nu:] = np.eye(reg.ss.Nu)
# AA[-reg.ss.Nu:, -2 * reg.ss.Nu:-reg.ss.Nu] = -np.eye(reg.ss.Nu)
AA[-reg.ss.Nu:, -reg.ss.Nu:] = np.eye(reg.ss.Nu)

AAd = expm(AA * dt)
Ad3 = AAd[:reg.ss.Nx, :reg.ss.Nx]
B0d3 = AAd[:reg.ss.Nx, reg.ss.Nx:reg.ss.Nx + reg.ss.Nu]
B1d3 = AAd[:reg.ss.Nx, -reg.ss.Nu:]

assert np.allclose(Ad, Ad3)
assert np.allclose(B0d, B0d3)
