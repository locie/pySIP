from typing import Tuple

import numpy as np
from numpy.linalg import LinAlgError, lstsq, solve
from scipy.linalg import (
    LinAlgWarning,
    block_diag,
    expm,
    solve_continuous_lyapunov,
    solve_sylvester,
)


def inv_2x2(X: np.ndarray) -> np.ndarray:
    """Inverse of 2-dimensional square matrix.

    Parameters
    ----------
    X : ndarray
        2-dimensional square matrix.

    Returns
    -------
    ndarray
        Inverse of `X`.
    """
    x00 = X[0, 0]
    x01 = X[0, 1]
    x10 = X[1, 0]
    x11 = X[1, 1]
    det = x00 * x11 - x01 * x10
    if det == 0.0:
        raise ValueError("The matrix cannot be inverted")

    return np.array([[x11, -x01], [-x10, x00]]) / det


def inv_3x3(X: np.ndarray) -> np.ndarray:
    """Inverse of 3-dimensional square matrix

    Parameters
    ----------
    X : array_like
        3-dimensional square matrix

    Returns
    -------
    array_like
        Inverse of `X`
    """
    x00, x01, x02, x10, x11, x12, x20, x21, x22 = X.ravel()
    det0 = x22 * x11 - x21 * x12
    det1 = x22 * x01 - x21 * x02
    det2 = x12 * x01 - x11 * x02
    det = x00 * det0 - x10 * det1 + x20 * det2
    if det == 0.0:
        raise ValueError("The matrix cannot be inverted")

    return (
        np.array(
            [
                [det0, -det1, det2],
                [x20 * x12 - x22 * x10, x22 * x00 - x20 * x02, x10 * x02 - x12 * x00],
                [x21 * x10 - x20 * x11, x20 * x01 - x21 * x00, x11 * x00 - x10 * x01],
            ]
        )
        / det
    )


def eigvals_2x2(X: np.ndarray) -> np.ndarray:
    """Eigenvalues of 2-dimensional square matrix

    Parameters
    ----------
    X : (N, N) array_like
        2-dimensional square matrix

    Returns
    -------
    eigenvalues : (N,) ndarray
        Eigenvalues of `X`

    References
    ----------
    M.J. Kronenbur. A Method for Fast Diagonalization ofa 2x2 or 3x3 Real
    Symmetric Matrix
    """
    x00 = X[0, 0]
    x01 = X[0, 1]
    x10 = X[1, 0]
    x11 = X[1, 1]
    tmp = np.sign(x00 - x11) * np.sqrt((x00 - x11) ** 2 + 4.0 * x01 * x10)
    return np.array([0.5 * (x00 + x11 + tmp), 0.5 * (x00 + x11 - tmp)])


def eig_2x2(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Eigenvalues and eigenvectors of 2-dimensional square matrix

    Parameters
    ----------
    X : array_like
        2-dimensional square matrix

    Returns
    -------
    w : ndarray
        Eigenvalues
    v : ndarray
        Eigenvectors

    References
    ----------
    M.J. Kronenbur. A Method for Fast Diagonalization ofa 2x2 or 3x3 Real Symmetric
    Matrix
    """
    x00 = X[0, 0]
    x01 = X[0, 1]
    x10 = X[1, 0]
    x11 = X[1, 1]
    d = x00 - x11
    tmp = np.sign(d) * trunc_sqrt(d**2 + 4.0 * x01 * x10)
    w1 = 0.5 * (x00 + x11 + tmp)
    w2 = 0.5 * (x00 + x11 - tmp)
    y1 = (w1 - x00) / x01
    y2 = (w2 - x00) / x01
    z1 = np.sqrt(1.0 + y1**2)
    z2 = np.sqrt(1.0 + y2**2)

    return np.array([w1, w2]), np.array([[1.0 / z1, 1.0 / z2], [y1 / z1, y2 / z2]])


def trunc_sqrt(x: float) -> float:
    """Truncated square root"""
    return np.sqrt(x) if x > 0.0 else 0.0


def trunc_arccos(x: float) -> float:
    """Truncated arccosine"""
    if x >= 1.0:
        return 0.0
    if x <= -1.0:
        return -1.0
    return np.arccos(x)


def eigvals_3x3(X: np.ndarray) -> np.ndarray:
    """Eigenvalues of 3x3 real square matrix

    Parameters
    ----------
    X: (3, 3) array
        Square matrix

    Returns
    -------
    eigenvalues: (3,) array
        Eigenvalues of `X`

    Notes
    -----
    This symbolic expression doesn't work for complex eigenvalues

    References
    ----------
    M.J. Kronenbur. A Method for Fast Diagonalization ofa 2x2 or 3x3 Real Symmetric
    Matrix
    """
    x00, x01, x02, x10, x11, x12, x20, x21, x22 = X.ravel()

    b = x00 + x11 + x22
    c = x00 * x11 + x00 * x22 - x01 * x10 - x02 * x20 + x11 * x22 - x12 * x21
    d = (
        -x00 * x11 * x22
        + x00 * x12 * x21
        + x01 * x10 * x22
        - x01 * x12 * x20
        - x02 * x10 * x21
        + x02 * x11 * x20
    )
    p = b**2 - 3.0 * c
    if p == 0.0:
        return (b / 3.0) * np.ones(3)

    q = 2.0 * b**3 - 9.0 * b * c - 27.0 * d
    if q**2 > 4.0 * p**3:
        raise ValueError("This symbolic expression works only for real eigenvalues")
    delta = trunc_arccos(q / (2.0 * trunc_sqrt(p**3)))
    tmp = 2.0 * trunc_sqrt(p)

    return np.array(
        [
            (b + tmp * np.cos(delta / 3.0)) / 3.0,
            (b + tmp * np.cos((delta + 2.0 * np.pi) / 3.0)) / 3.0,
            (b + tmp * np.cos((delta - 2.0 * np.pi) / 3.0)) / 3.0,
        ]
    )


def expm_2x2(X: np.ndarray) -> np.ndarray:
    """Matrix exponential of 2-dimensional square matrix

    Parameters
    ----------
    X : array_like
        2-dimensional square matrix

    Returns
    -------
    array_like
        Matrix exponential of `X`
    """
    w, v = eig_2x2(X)
    return v @ np.diag(np.exp(w)) @ inv_2x2(v)


def state_input(
    A: np.ndarray,
    B: np.ndarray,
    dt: float = 1.0,
    order_hold: int = 0,
    method: str = "expm",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Discretize the state and input matrices

    .. math::

        x_{k+1} = Ad x_k + B0d u_k + B1d \\Delta u_k

    Parameters
    ----------
    A : ndarray
        State matrix
    B : ndarray
        Input matrix
    dt : float
        Sampling time
    order_hold : int
        zero order hold = 0 or first order hold = 1
    method : str
        Augmented matrix exponential `expm` or symbolic `analytic`

    Returns
    -------
    Ad : ndarray
        Discrete state matrix
    B0d : ndarray
        Discrete input matrix (zero order hold)
    B1d : ndarray
        Discrete input matrix (first order hold)
    """
    if method == "expm":
        return state_input_expm(A, B, dt, order_hold)
    elif method == "analytic":
        Ad = state(A, dt)
        B0d, B1d = input_analytic(A, B, Ad, dt, order_hold)
        return Ad, B0d, B1d
    else:
        raise ValueError("`method must be `expm` or `analytic`")


def state(A: np.ndarray, dt: float = 1.0) -> np.ndarray:
    """Discretize the state matrix

    Parameters
    ----------
    A : array_like
        State matrix
    dt : float
        Sampling time

    Returns
    -------
    Ad : array_like
        Discrete state matrix
    """
    return expm(A * dt)


def input_analytic(
    A: np.ndarray, B: np.ndarray, Ad: np.ndarray, dt: float = 1.0, order_hold: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """Discretize the input matrix with the analytic expression

    Parameters
    ----------
    A : array_like
        State matrix
    B : array_like
        Input matrix
    Ad : array_like
        Discrete state matrix
    dt : float
        Sampling time
    order_hold : int
        zero order hold = 0 or first order hold = 1

    Returns
    -------
    B0d : array_like
        Discrete input matrix (zero order hold)
    B1d : array_like
        Discrete input matrix (first order hold)
    """
    nx, nu = B.shape
    Ix = np.eye(nx)
    if nx <= 3:
        if nx == 1:
            invA = 1.0 / A
        elif nx == 2:
            invA = inv_2x2(A)
        elif nx == 3:
            invA = inv_3x3(A)
        B0d = invA @ (Ad - Ix) @ B
        if order_hold == 0:
            B1d = np.zeros((nx, nu))
        else:
            B1d = invA @ (B0d - Ix * dt @ B)
    else:
        B0d = solve(A, (Ad - Ix) @ B)
        if order_hold == 0:
            B1d = np.zeros((nx, nu))
        else:
            B1d = solve(A, (B0d - Ix * dt @ B))

    return B0d, B1d


def state_input_expm(
    A: np.ndarray, B: np.ndarray, dt: float = 1.0, order_hold: int = 0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Discretize the state and input matrices with matrix exponential

    Parameters
    ----------
    A : numpy.array
        State matrix
    B : numpy.array
        Input matrix
    dt : float
        Sampling time
    order_hold : int
        zero order hold = 0 or first order hold = 1

    Returns
    -------
    Ad : numpy.array
        Discrete state matrix
    B0d : numpy.array
        Discrete input matrix (zero order hold)
    B1d : numpy.array
        Discrete input matrix (first order hold)
    """
    nx, nu = B.shape
    if order_hold == 0:
        F = np.zeros((nx + nu, nx + nu), dtype=A.dtype)
        F[:nx, :nx] = A
        F[:nx, nx:] = B
        Ad, B0d = np.split(expm(F * dt)[:nx, :], indices_or_sections=[nx], axis=1)
        B1d = np.zeros((nx, nu))
    else:
        F = np.zeros((nx + 2 * nu, nx + 2 * nu), dtype=A.dtype)
        F[:nx, :nx] = A
        F[:nx, nx : nx + nu] = B
        F[nx : nx + nu, nx + nu :] = np.eye(nu)
        Ad, B0d, B1d = np.split(
            expm(F * dt)[:nx, :], indices_or_sections=[nx, nx + nu], axis=1
        )
    return Ad, B0d, B1d


def expm_triu(
    a11: np.ndarray,
    a12: np.ndarray,
    a22: np.ndarray,
    dt: float,
    f11: np.ndarray,
    f22: np.ndarray,
) -> np.ndarray:
    """Compute the exponential of the upper triangular matrix A using the Parlett's
    method.

    Parameters
    ----------
    a11: Upper left input matrix
    a12: Upper right input matrix
    a22: Lower right input matrix
    dt: Sampling time
    f11: Upper left output matrix
    f22: Lower right output matrix

    Returns
    -------
    F: Matrix exponential of A

    References
    ----------
    A Schur-Parlett Algorithm for Computing Matrix Functions
    """
    F = block_diag(f11, f22)
    dim = a12.shape[0]
    F[:dim, dim:] = solve_sylvester(a11, -a22, f11 @ a12 - a12 @ f22)
    return F


def diffusion_mfd(A: np.ndarray, Q: np.ndarray, dt: float = 1.0) -> np.ndarray:
    """Discretize the diffusion matrix by Matrix Fraction Decomposition.

    Parameters
    ----------
    A : array_like
        State matrix.
    Q : array_like
        Diffusion matrix.
    dt : float
        Sampling time.

    Returns
    -------
    array_like
        Process noise covariance matrix.
    """
    if not Q.any():
        return Q

    nx = A.shape[0]
    F = np.zeros((2 * nx, 2 * nx), dtype=A.dtype)
    F[:nx, :nx] = A
    F[nx:, nx:] = -A.T
    F[:nx, nx:] = Q
    Fd = expm(F * dt)[:nx, :]

    return Fd[:, nx:] @ Fd[:, :nx].T


def diffusion_lyap(A: np.ndarray, Q: np.ndarray, Ad: np.ndarray) -> np.ndarray:
    """Discretize the diffusion matrix by solving the Lyapunov equation

    Parameters
    ----------
    A : array_like
        State matrix
    Q : array_like
        Diffusion matrix
    Ad : array_like
        Discrete state matrix

    Returns
    -------
    Qd : array_like
        Process noise covariance matrix
    """
    if not Q.any():
        return Q
    return solve_continuous_lyapunov(A, -Q + Ad @ Q @ Ad.T)


def diffusion_stationary(Pinf: np.ndarray, Ad: np.ndarray) -> np.ndarray:
    """
    Discretize the stationary covariance matrix

    Parameters
    ----------
    Pinf : numpy.array
        Stationary state covariance matrix
    Ad : numpy.array
        Discrete state matrix

    Returns
    -------
    numpy.array
        Process noise covariance matrix
    """
    if not Pinf.any():
        return Pinf
    return Pinf - Ad @ Pinf @ Ad.T


def diffusion_kron(A: np.ndarray, Q: np.ndarray, Ad: np.ndarray) -> np.ndarray:
    """Discretize diffusion matrix by solving indirectly the Lyapunov equation

    Charles C. Driver, Manuel C. Voelkle.
    Introduction to Hierarchical Continuous Time Dynamic Modelling With ctsem

    Parameters
    ----------
    A : array
        State matrix
    Q : array
        Diffusion matrix
    Ad : array
        Discrete state matrix

    Returns
    -------
    Qd : array
        Process noise covariance matrix
    """
    if not Q.any():
        return Q

    nx = Q.shape[0]
    Ix = np.eye(nx)
    A_sharp = np.kron(A, Ix) + np.kron(Ix, A)
    b = np.atleast_2d(Q.ravel()).T
    try:
        x = solve(-A_sharp, b)
    except (LinAlgWarning, LinAlgError):
        x = lstsq(-A_sharp, b, rcond=-1)[0]

    return diffusion_stationary(np.reshape(x, (nx, nx)), Ad)
