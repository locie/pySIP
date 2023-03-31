from copy import deepcopy
from typing import NamedTuple, Tuple

import numpy as np
from numpy.linalg import lstsq, solve
from scipy.linalg import LinAlgError, LinAlgWarning

from .base import BayesianFilter


def _predict(
    Ad: np.ndarray,
    B0d: np.ndarray,
    B1d: np.ndarray,
    Qd: np.ndarray,
    x: np.ndarray,
    P: np.ndarray,
    u: np.ndarray,
    u1: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """State prediction

    Args:
        Ad: State matrix
        B0d: Input matrix (zero order hold)
        B1d: Input matrix (first order hold)
        Qd: Process noise covariance matrix
        x: State mean
        P: State deviation
        u: Input data
        u1: Forward finite difference of the input data

    Returns:
        2-element tuple containing
            - **x**: Prior state mean
            - **P**: Prior state deviation
    """
    P = np.linalg.qr(np.vstack([P @ Ad.T, Qd]), "r")
    x = Ad @ x + B0d @ u + B1d @ u1

    return x, P


def _grad_predict(
    Ad: np.ndarray,
    dAd: np.ndarray,
    B0d: np.ndarray,
    dB0d: np.ndarray,
    B1d: np.ndarray,
    dB1d: np.ndarray,
    Qd: np.ndarray,
    dQd: np.ndarray,
    x: np.ndarray,
    dx: np.ndarray,
    P: np.ndarray,
    dP: np.ndarray,
    u: np.ndarray,
    u1: np.ndarray,
) -> Tuple[tuple([np.ndarray] * 4)]:
    """Derivative state prediction

    Args:
        Ad: State matrix
        dAd: Jacobian state matrix
        B0d: Input matrix (zero order hold)
        dB0d: Jacobian input matrix (zero order hold)
        B1d: Input matrix (first order hold)
        dB1d: Jacobian input matrix (first order hold)
        Qd: Process noise covariance matrix
        dQd: Jacobian process noise covariance matrix
        x: State mean
        dx: Jacobian state mean
        P: State deviation
        dP: Jacobian state deviation
        u: Input data
        u1: Forward finite difference of the input data

    Returns:
        4-element tuple containing
            - **x**: Prior state mean
            - **dx**: Derivative prior state mean
            - **P**: Prior state deviation
            - **dP**: Derivative prior state deviation
    """

    dArrp = np.hstack([dP @ Ad.T + P @ dAd.swapaxes(1, 2), dQd])
    Q, P = np.linalg.qr(np.vstack([P @ Ad.T, Qd]))
    inner = dArrp.swapaxes(1, 2) @ Q
    try:
        tmp = solve(P.T, inner).swapaxes(1, 2)
    except (LinAlgError, LinAlgWarning):
        tmp = inner.swapaxes(1, 2) @ np.linalg.pinv(P)
    dP = (np.swapaxes(np.tril(tmp, -1), 1, 2) + np.triu(tmp)) @ P

    dx = dAd @ x + Ad @ dx + dB0d @ u + dB1d @ u1
    x = Ad @ x + B0d @ u + B1d @ u1

    return x, dx, P, dP


def _update(
    C: np.ndarray,
    D: np.ndarray,
    R: np.ndarray,
    x: np.ndarray,
    P: np.ndarray,
    u: np.ndarray,
    y: np.ndarray,
) -> Tuple[tuple([np.ndarray] * 4)]:
    """State update

    Args:
        C: Output matrix
        D: Feedthrough matrix
        R: Measurement deviation matrix
        x: State mean
        P: State deviation
        u: Input data
        y: Output data

    Returns:
        4-element tuple containing
            - **x**: Filtered state mean
            - **P**: Filtered state deviation
            - **e**: Standardized residulas
            - **S**: Residuals deviation
    """
    ny, nx = C.shape
    x = x.copy()
    P = P.copy()
    Arru = np.zeros((nx + ny, nx + ny))

    Arru[:ny, :ny] = R
    Arru[ny:, :ny] = P @ C.T
    Arru[ny:, ny:] = P
    _, Post = np.linalg.qr(Arru)
    S = Post[:ny, :ny]
    e = np.zeros((ny, 1))
    if ny > 1:
        try:
            e = solve(S, y - C @ x - D @ u)
        except (LinAlgWarning, LinAlgError):
            e = lstsq(S, y - C @ x - D @ u, rcond=-1)[0]
    else:
        e = (y - C @ x - D @ u) / S

    x += Post[:ny, ny:].T @ e
    P = Post[ny:, ny:]

    return x, P, e, S


def _grad_update(
    C: np.ndarray,
    dC: np.ndarray,
    D: np.ndarray,
    dD: np.ndarray,
    R: np.ndarray,
    dR: np.ndarray,
    x: np.ndarray,
    dx: np.ndarray,
    P: np.ndarray,
    dP: np.ndarray,
    u: np.ndarray,
    y: np.ndarray,
) -> Tuple[tuple([np.ndarray] * 8)]:
    """Derivative state update

    Args:
        C: Output matrix
        dC: Jacobian output matrix
        D: Feedthrough matrix
        dD: Jacobian feedthrough matrix
        R: Measurement deviation matrix
        dR: Jacobian measurement deviation matrix
        x: State mean
        dx: Jacobian state mean
        P: State deviation
        dP: Jacobian state deviation
        u: Input data
        y: Output data

    Returns:
        8-element tuple containing
            - **x**: Filtered state mean
            - **dx**: Derivative filtered state mean
            - **P**: Filtered state deviation
            - **dP**: Derivative filtered state deviation
            - **e**: Standardized residulas
            - **de**: Derivative standardized residulas
            - **S**: Residuals deviation
            - **dS**: Derivative residuals deviation
    """
    npar, ny, nx = dC.shape

    x = x.copy()
    P = P.copy()
    dx = dx.copy()
    dP = dP.copy()
    Arru = np.zeros((nx + ny, nx + ny))
    dArru = np.zeros((npar, nx + ny, nx + ny))
    Arru[:ny, :ny] = R
    Arru[ny:, :ny] = P @ C.T
    Arru[ny:, ny:] = P

    dArru[:, :ny, :ny] = dR
    dArru[:, ny:, :ny] = dP @ C.T + P @ dC.swapaxes(1, 2)
    dArru[:, ny:, ny:] = dP

    Q, Post = np.linalg.qr(Arru)
    inner = dArru.swapaxes(1, 2) @ Q
    try:
        tmp = np.linalg.solve(Post.T, inner).swapaxes(1, 2)
    except (LinAlgError, LinAlgWarning):
        tmp = inner.swapaxes(1, 2) @ np.linalg.pinv(Post)
    dPost = (np.swapaxes(np.tril(tmp, -1), 1, 2) + np.triu(tmp)) @ Post

    K = Post[:ny, ny:].T
    S = Post[:ny, :ny]
    dS = dPost[:, :ny, :ny]

    if ny > 1:
        try:
            e = solve(S, y - C @ x - D @ u)
            de = solve(-S, dS @ e + dC @ x + C @ dx + dD @ u)
        except (LinAlgWarning, LinAlgError):
            invS = np.linalg.pinv(S)
            e = invS @ (y - C @ x - D @ u)
            de = -invS @ (dS @ e + dC @ x + C @ dx + dD @ u)
    else:
        e = (y - C @ x - D @ u) / S
        de = -(dS @ e + dC @ x + C @ dx + dD @ u) / S

    x += K @ e
    dx += dPost[:, :ny, ny:].swapaxes(1, 2) @ e + K @ de
    P = Post[ny:, ny:]
    dP = dPost[:, ny:, ny:]

    return x, dx, P, dP, e, de, S, dS


class KalmanQR(BayesianFilter):
    """Square-root Kalman filter and sensitivity equations

    References:
        Maria V. Kulikova, Julia Tsyganova, A unified square-root approach for the score
        and Fisher information matrix computation in linear dynamic systems. Mathematics
        and Computers in Simulation 119: 128-141 (2016)
    """

    def predict(
        self,
        Ad: np.ndarray,
        B0d: np.ndarray,
        B1d: np.ndarray,
        Qd: np.ndarray,
        x: np.ndarray,
        P: np.ndarray,
        u: np.ndarray,
        u1: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """State prediction

        Args:
            Ad: State matrix
            B0d: Input matrix (zero order hold)
            B1d: Input matrix (first order hold)
            Qd: Process noise covariance matrix
            x: State mean
            P: State deviation
            u: Input data
            u1: Forward finite difference of the input data

        Returns:
            2-element tuple containing
                - **x**: Prior state mean
                - **P**: Prior state deviation
        """
        return _predict(Ad, B0d, B1d, Qd, x, P, u, u1)

    def dpredict(
        self,
        Ad: np.ndarray,
        dAd: np.ndarray,
        B0d: np.ndarray,
        dB0d: np.ndarray,
        B1d: np.ndarray,
        dB1d: np.ndarray,
        Qd: np.ndarray,
        dQd: np.ndarray,
        x: np.ndarray,
        dx: np.ndarray,
        P: np.ndarray,
        dP: np.ndarray,
        u: np.ndarray,
        u1: np.ndarray,
    ) -> Tuple[tuple([np.ndarray] * 4)]:
        """Derivative state prediction

        Args:
            Ad: State matrix
            dAd: Jacobian state matrix
            B0d: Input matrix (zero order hold)
            dB0d: Jacobian input matrix (zero order hold)
            B1d: Input matrix (first order hold)
            dB1d: Jacobian input matrix (first order hold)
            Qd: Process noise covariance matrix
            dQd: Jacobian process noise covariance matrix
            x: State mean
            dx: Jacobian state mean
            P: State deviation
            dP: Jacobian state deviation
            u: Input data
            u1: Forward finite difference of the input data

        Returns:
            4-element tuple containing
                - **x**: Prior state mean
                - **dx**: Derivative prior state mean
                - **P**: Prior state deviation
                - **dP**: Derivative prior state deviation
        """

        return _grad_predict(
            Ad,
            dAd,
            B0d,
            dB0d,
            B1d,
            dB1d,
            Qd,
            dQd,
            x,
            dx,
            P,
            dP,
            u,
            u1,
        )

    def update(
        self,
        C: np.ndarray,
        D: np.ndarray,
        R: np.ndarray,
        x: np.ndarray,
        P: np.ndarray,
        u: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[tuple([np.ndarray] * 4)]:
        """State update

        Args:
            C: Output matrix
            D: Feedthrough matrix
            R: Measurement deviation matrix
            x: State mean
            P: State deviation
            u: Input data
            y: Output data

        Returns:
            4-element tuple containing
                - **x**: Filtered state mean
                - **P**: Filtered state deviation
                - **e**: Standardized residulas
                - **S**: Residuals deviation
        """
        return _update(C, D, R, x, P, u, y)

    def dupdate(
        self,
        C: np.ndarray,
        dC: np.ndarray,
        D: np.ndarray,
        dD: np.ndarray,
        R: np.ndarray,
        dR: np.ndarray,
        x: np.ndarray,
        dx: np.ndarray,
        P: np.ndarray,
        dP: np.ndarray,
        u: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """Derivative state update

        Args:
            C: Output matrix
            dC: Jacobian output matrix
            D: Feedthrough matrix
            dD: Jacobian feedthrough matrix
            R: Measurement deviation matrix
            dR: Jacobian measurement deviation matrix
            x: State mean
            dx: Jacobian state mean
            P: State deviation
            dP: Jacobian state deviation
            u: Input data
            y: Output data

        Returns:
            8-element tuple containing
                - **x**: Filtered state mean
                - **dx**: Derivative filtered state mean
                - **P**: Filtered state deviation
                - **dP**: Derivative filtered state deviation
                - **e**: Standardized residulas
                - **de**: Derivative standardized residulas
                - **S**: Residuals deviation
                - **dS**: Derivative residuals deviation
        """

        return _grad_update(
            C,
            dC,
            D,
            dD,
            R,
            dR,
            x,
            dx,
            P,
            dP,
            u,
            y,
        )

    def log_likelihood(
        self,
        ssm: NamedTuple,
        index: np.ndarray,
        u: np.ndarray,
        u1: np.ndarray,
        y: np.ndarray,
        point_wise: bool = False,
    ) -> float:
        """Evaluate the negative log-likelihood

        Args:
            ssm: Discrete state-space model
            index: Index of unique time steps
            u: Input data
            u1: Forward finite difference of the input data
            y: Output data
            point_wise: Return the log-likelihood evaluated point-wise

        Returns:
            loglik: Negative log-likelihood
        """
        T = y.shape[1]
        ny, nx = ssm.C.shape
        loglik = np.full(T, 0.5 * ny * np.log(2.0 * np.pi))
        x = deepcopy(ssm.x0)
        P = deepcopy(ssm.P0)

        do_update = ~np.isnan(y).any(axis=0)
        for t in range(T):
            if do_update[t]:
                x, P, e, S = self.update(
                    ssm.C, ssm.D, ssm.R, x, P, u[:, t : t + 1], y[:, t]
                )

                if ny > 1:
                    loglik[t] += np.linalg.slogdet(S)[1] + 0.5 * e.T @ e
                else:
                    loglik[t] += np.log(np.abs(S)) + 0.5 * e**2

            i = index[t]
            x, P = self.predict(
                ssm.A[i],
                ssm.B0[i],
                ssm.B1[i],
                ssm.Q[i],
                x,
                P,
                u[:, t : t + 1],
                u1[:, t : t + 1],
            )
        if not point_wise:
            loglik = loglik.sum()

        return loglik

    def dlog_likelihood(
        self,
        ssm: NamedTuple,
        dssm: NamedTuple,
        index: np.ndarray,
        u: np.ndarray,
        u1: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[float, np.ndarray]:
        """Evaluate the gradient of the negative log-likelihood

        Args:
            ssm: Discrete state-space model
            dssm: Discrete jacobian state-space model
            index: Index of unique time steps
            u: Input data
            u1: Forward finite difference of the input data
            y: Output data

        Returns:
            2-element tuple containing
                - **loglik**: Negative log-likelihood
                - **gradient**: Gradient of the negative log-likelihood
        """
        ny, T = y.shape
        x = deepcopy(ssm.x0)
        P = deepcopy(ssm.P0)
        dx = deepcopy(dssm.dx0)
        dP = deepcopy(dssm.dP0)

        loglik = 0.5 * T * ny * np.log(2.0 * np.pi)
        gradient = np.zeros(dP.shape[0])

        do_update = ~np.isnan(y).any(axis=0)
        for t in range(T):
            if do_update[t]:
                x, dx, P, dP, e, de, S, dS = self.dupdate(
                    ssm.C,
                    dssm.dC,
                    ssm.D,
                    dssm.dD,
                    ssm.R,
                    dssm.dR,
                    x,
                    dx,
                    P,
                    dP,
                    u[:, t : t + 1],
                    y[:, t],
                )

                if ny > 1:
                    loglik += np.linalg.slogdet(S)[1] + 0.5 * e.T @ e
                    tmp = np.linalg.solve(S, dS)
                    gradient += tmp.trace(0, 1, 2) + np.squeeze(e.T @ de)
                else:
                    loglik += np.log(np.abs(S)) + 0.5 * e**2
                    gradient += np.squeeze(dS / S + e * de)

            i = index[t]
            x, dx, P, dP = self.dpredict(
                ssm.A[i],
                dssm.dA[i],
                ssm.B0[i],
                dssm.dB0[i],
                ssm.B1[i],
                dssm.dB1[i],
                ssm.Q[i],
                dssm.dQ[i],
                x,
                dx,
                P,
                dP,
                u[:, t : t + 1],
                u1[:, t : t + 1],
            )

        return loglik[0, 0], gradient

    def filtering(
        self,
        ssm: NamedTuple,
        index: np.ndarray,
        u: np.ndarray,
        u1: np.ndarray,
        y: np.ndarray,
        x0: np.ndarray = None,
        P0: np.ndarray = None,
    ) -> Tuple[tuple([np.ndarray] * 4)]:
        """Compute the filtered state distribution

        Args:
            ssm: Discrete state-space model
            index: Index of unique time steps
            u: Input data
            u1: Forward finite difference of the input data
            y: Output data
            x0: Inital state mean different from `ssm`
            P0: Inital state deviation different from `ssm`

        Returns:
            4-element tuple containing
                - **xf**: Filtered state mean
                - **Pf**: Filtered state covariance
                - **res**: Standardized residuals
                - **dev_res**: Residuals deviation
        """
        T = y.shape[1]
        ny, nx = ssm.C.shape
        xf = np.empty((T, nx, 1))
        Pf = np.empty((T, nx, nx))
        res = np.zeros((T, ny, 1))
        dev_res = np.zeros((T, ny, ny))
        do_update = ~np.isnan(y).any(axis=0)

        if x0 is None:
            x = deepcopy(ssm.x0)
        else:
            if x0.shape != (nx, 1):
                raise ValueError(f"the dimensions of `x0` must be {(nx, 1)}")
            x = x0

        if P0 is None:
            P = deepcopy(ssm.P0)
        else:
            if P0.shape != (nx, nx):
                raise ValueError(f"the dimensions of `P0` must be {(nx, nx)}")
            P = P0

        for t in range(T):
            if do_update[t]:
                x, P, e, S = self.update(
                    ssm.C, ssm.D, ssm.R, x, P, u[:, t : t + 1], y[:, t]
                )
                res[t, :, :] = e
                dev_res[t, :, :] = S

            xf[t, :, :] = x
            Pf[t, :, :] = P.T @ P

            i = index[t]
            x, P = self.predict(
                ssm.A[i],
                ssm.B0[i],
                ssm.B1[i],
                ssm.Q[i],
                x,
                P,
                u[:, t : t + 1],
                u1[:, t : t + 1],
            )

        return xf, Pf, res, dev_res

    def smoothing(
        self,
        ssm: NamedTuple,
        index: np.array,
        u: np.ndarray,
        u1: np.ndarray,
        y: np.ndarray,
        x0: np.ndarray = None,
        P0: np.ndarray = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the smoothed state distribution Rauch-Tung-Striebel smoother

        Args:
            ssm: Discrete state-space model
            index: Index of unique time steps
            u: Input data
            u1: Forward finite difference of the input data
            y: Output data
            x0: Inital state mean different from `ssm`
            P0: Inital state deviation different from `ssm`

        Returns:
            2-element tuple containing
                - **xs**: Smoothed state mean
                - **Ps**: Smoothed state covariance
        """
        T = y.shape[1]
        nx = ssm.C.shape[1]
        xp = np.empty((T, nx, 1))
        xs = np.empty((T, nx, 1))
        Pp = np.empty((T, nx, nx))
        Ps = np.empty((T, nx, nx))
        do_update = ~np.isnan(y).any(axis=0)

        if x0 is None:
            x = deepcopy(ssm.x0)
        else:
            if x0.shape != (nx, 1):
                raise ValueError(f"the dimensions of `x0` must be {(nx, 1)}")
            x = x0

        if P0 is None:
            P = deepcopy(ssm.P0)
        else:
            if P0.shape != (nx, nx):
                raise ValueError(f"the dimensions of `P0` must be {(nx, nx)}")
            P = P0

        # Run forward Kalman filter
        for t in range(T):
            # Save prior state distributoin
            xp[t, :, :] = x
            Pp[t, :, :] = P.T @ P

            if do_update[t]:
                x, P, *_ = self.update(
                    ssm.C, ssm.D, ssm.R, x, P, u[:, t : t + 1], y[:, t]
                )

            # Save filtered state distribution
            xs[t, :, :] = x
            Ps[t, :, :] = P.T @ P

            i = index[t]
            x, P = self.predict(
                ssm.A[i],
                ssm.B0[i],
                ssm.B1[i],
                ssm.Q[i],
                x,
                P,
                u[:, t : t + 1],
                u1[:, t : t + 1],
            )

        # Run backward RTS smoother
        for t in reversed(range(T - 1)):
            # smoother gain (note: Pp and Ps are symmetric)
            try:
                G = solve(Pp[t + 1, :, :], ssm.A[index[t], :, :] @ Ps[t, :, :]).T
            except (LinAlgWarning, LinAlgError):
                G = lstsq(
                    Pp[t + 1, :, :], ssm.A[index[t], :, :] @ Ps[t, :, :], rcond=-1
                )[0].T

            # smoothed state mean and covariance
            xs[t, :, :] += G @ (xs[t + 1, :, :] - xp[t + 1, :, :])
            Ps[t, :, :] += G @ (Ps[t + 1, :, :] - Pp[t + 1, :, :]) @ G.T

        del xp, Pp

        return xs, Ps

    def simulate(
        self,
        ssm: NamedTuple,
        index: np.ndarray,
        u: np.ndarray,
        u1: np.ndarray,
        x0: np.ndarray = None,
    ) -> np.ndarray:
        """Stochastic output simulation

        Args:
            ssm: Discrete state-space model
            index: Index of unique time steps
            u: Input data
            u1: Forward finite difference of the input data
            x0: Inital state mean different from `ssm`

        Returns:
            y: Simulated output

        TODO:
            Random initialization of the state vector, such that x0 ~ Normal(0, P0)
        """
        T = u.shape[1]
        ny, nx = ssm.C.shape
        y = np.empty((T, ny, 1))

        if x0 is None:
            x = deepcopy(ssm.x0)
        else:
            if x0.shape != (nx, 1):
                raise ValueError(f"the dimensions of `x0` must be {(nx, 1)}")
            x = x0

        for t in range(T):
            y[t, :, :] = ssm.C @ x - ssm.D @ u[:, t : t + 1]
            i = index[t]
            x = (
                ssm.A[i] @ x
                + ssm.B0[i] @ u[:, t : t + 1]
                + ssm.B1[i] @ u1[:, t : t + 1]
                + ssm.Q[i] @ np.random.randn(nx, 1)
            )
        y += ssm.R @ np.random.randn(T, ny, 1)

        return y[:, :, 0]

    def simulate_output(
        self,
        ssm: NamedTuple,
        index: np.ndarray,
        u: np.ndarray,
        u1: np.ndarray,
        y: np.ndarray,
        x0: np.ndarray = None,
        P0: np.ndarray = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the filtered output distribution with the Kalman filter

        Args:
            ssm: Discrete state-space model
            index: Index of unique time steps
            u: Input data
            u1: Forward finite difference of the input data
            y: Output data
            x0: Inital state mean different from `ssm`
            P0: Inital state deviation different from `ssm`

        Returns:
            2-element tuple containing
                - **ym**: Filtered output mean
                - **ysd**: Filtered output standard deviation
        """
        T = y.shape[1]
        ny, nx = ssm.C.shape
        ym = np.empty((T, ny, 1))
        ysd = np.empty((T, ny, ny))
        do_update = ~np.isnan(y).any(axis=0)

        if x0 is None:
            x = deepcopy(ssm.x0)
        else:
            if x0.shape != (nx, 1):
                raise ValueError(f"the dimensions of `x0` must be {(nx, 1)}")
            x = x0

        if P0 is None:
            P = deepcopy(ssm.P0)
        else:
            if P0.shape != (nx, nx):
                raise ValueError(f"the dimensions of `P0` must be {(nx, nx)}")
            P = P0

        for t in range(T):
            if do_update[t]:
                x, P, *_ = self.update(
                    ssm.C, ssm.D, ssm.R, x, P, u[:, t : t + 1], y[:, t]
                )

            ym[t, :, :] = ssm.C @ x
            ysd[t, :, :] = np.sqrt(ssm.C @ P.T @ P @ ssm.C.T) + ssm.R

            i = index[t]
            x, P = self.predict(
                ssm.A[i],
                ssm.B0[i],
                ssm.B1[i],
                ssm.Q[i],
                x,
                P,
                u[:, t : t + 1],
                u1[:, t : t + 1],
            )

        return ym, ysd
