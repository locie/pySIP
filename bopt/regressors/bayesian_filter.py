"""Import dependencies"""
import numpy as np


class BayesianFilter:
    """Template for Bayesian filters"""

    def prediction(self, **kwargs):
        """State predictive distribution from time t to t+1"""
        raise NotImplementedError

    def dprediction(self, **kwargs):
        """Derivative state predictive distribution from time t to t+1"""
        raise NotImplementedError

    def update(self, **kwargs):
        """Filtered state distribution at time t"""
        raise NotImplementedError

    def dupdate(self, **kwargs):
        """Derivative filtered state distribution at time t"""
        raise NotImplementedError

    def filtering(self, **kwargs):
        """Compute the filtered state distribution and the residuals"""
        raise NotImplementedError

    def smoothing(self, **kwargs):
        """Compute the smoothed state distribution"""
        raise NotImplementedError

    def log_likelihood(self, **kwargs):
        """Evaluate the negative log-likelihood"""
        raise NotImplementedError

    def dlog_likelihood(self, **kwargs):
        """Evaluate the gradient of the negative log-likelihood"""
        raise NotImplementedError

    # def eval_d2log_likelihood(self):
    #     raise NotImplementedError


class SRKalman(BayesianFilter):
    """Square-root Kalman filter

    Args:
        Np: Number of free parameters
        Nx: Number of states
        Nu: Number of inputs
        Ny: Number of outputs
    """

    def __init__(self, Np, Nx, Nu, Ny):
        self.Np = Np
        self.Nx = Nx
        self.Nu = Nu
        self.Ny = Ny
        self._LOG_2PI = np.log(2.0 * np.pi)
        self._compute_hessian = False
        self._Prep = np.zeros((2 * Nx, Nx))
        self._dPrep = np.zeros((Np, 2 * Nx, Nx))
        self._Preu = np.zeros((Nx + Ny, Nx + Ny))
        self._dPreu = np.zeros((Np, Nx + Ny, Nx + Ny))

    def prediction(self, Ad, B0d, B1d, Qd, x, P, dt, u, u1):
        """State predictive distribution from time t to t+1

        Args:
            Ad: State matrix
            B0d, B1d: Input matrices
            Qd: Process noise covariance matrix
            x: State mean
            P: State deviation
            dt: Time interval
            u: Input array
            u1: Derivative input array

        Returns:
            x: Prior state mean at time t+1
            P: Prior state deviation at time t+1
        """
        self._Prep[self.Nx:, :] = Qd
        self._Prep[:self.Nx, :] = P @ Ad.T
        P = np.linalg.qr(self._Prep, "r")
        x = Ad @ x + B0d @ (dt * u1 + u) - B1d @ u1

        return x, P

    def dprediction(self, Ad, dAd, B0d, dB0d, B1d, dB1d, Qd, dQd,
                    x, dx, P, dP, dt, u, u1):
        """Derivative state predictive distribution from time t to t+1

        Args:
            Ad: State matrix
            dAd: Derivative state matrix
            B0d, B1d: Input matrices
            dB0d, dB1d: Derivative input matrices
            Qd: Process noise covariance matrix
            dQd: Derivative process noise covariance matrix
            x: State mean
            dx: Derivative state mean
            P: State deviation
            dP: Derivative state deviation
            dt: Time interval
            u: Input array
            u1: Derivative input array

        Returns:
            x: Prior state mean at time t+1
            dx: Derivative prior state mean at time t+1
            P: Prior state deviation at time t+1
            dP: Derivative prior state deviation at time t+1
        """
        self._Prep[self.Nx:, :] = Qd
        self._Prep[:self.Nx, :] = P @ Ad.T

        dPrep = self._dPrep[:dP.shape[0], :, :]
        dPrep[:, self.Nx:, :] = dQd
        dPrep[:, :self.Nx, :] = dP @ Ad.T + P @ dAd.swapaxes(1, 2)

        Q, P = np.linalg.qr(self._Prep)
        tmp = np.linalg.solve(P.T, dPrep.swapaxes(1, 2) @ Q).swapaxes(1, 2)
        dP = (np.swapaxes(np.tril(tmp, -1), 1, 2) + np.triu(tmp)) @ P

        dx = dAd @ x + Ad @ dx + dB0d @ (dt * u1 + u) - dB1d @ u1
        x = Ad @ x + B0d @ (dt * u1 + u) - B1d @ u1

        return x, dx, P, dP

    def update(self, C, D, R, x, P, u, y):
        """Filtered state distribution at time t

        Args:
            C: Output matrix
            D: Feedthrough matrix
            R: Measurement deviation matrix
            x: State mean
            P: State deviation
            u: Input array
            y: Measurement array

        Returns:
            x: Filtered state mean
            P: Filtered state deviation
            e: Standardized residulas
            S: Residuals deviation
        """
        self._Preu[:self.Ny, :self.Ny] = R
        self._Preu[self.Ny:, :self.Ny] = P @ C.T
        self._Preu[self.Ny:, self.Ny:] = P
        Post = np.linalg.qr(self._Preu, "r")
        S = Post[:self.Ny, :self.Ny]

        if self.Ny > 1:
            e = np.linalg.solve(S, y - C @ x - D @ u)
        else:
            e = (y - C @ x - D @ u) / S

        x += Post[:self.Ny, self.Ny:].T @ e
        P = Post[self.Ny:, self.Ny:]

        return x, P, e, S

    def dupdate(self, C, dC, D, dD, R, dR, x, dx, P, dP, u, y):
        """Derivative filtered state distribution at time t

        Args:
            C: Output matrix
            dC: Derivative output matrix
            D: Feedthrough matrix
            dD: Derivative feedthrough matrix
            R: Measurement deviation matrix
            dR: Derivative measurement deviation matrix
            x: State mean
            dx: Derivative state mean
            P: State deviation
            dP: Derivative state deviation
            u: Input array
            y: Measurement array

        Returns:
            x: Filtered state mean
            dx: Derivative Filtered state mean
            P: Filtered state deviation
            dP: Derivative Filtered state deviation
            e: Standardized residuals
            de: Derivative standardized residuals
            S: Residuals deviation
            dS: Derivative residuals deviation
        """
        self._Preu[:self.Ny, :self.Ny] = R
        self._Preu[self.Ny:, :self.Ny] = P @ C.T
        self._Preu[self.Ny:, self.Ny:] = P
        dPreu = self._dPreu[:dP.shape[0], :, :]
        dPreu[:, :self.Ny, :self.Ny] = dR
        dPreu[:, self.Ny:, :self.Ny] = dP @ C.T + P @ dC.swapaxes(1, 2)
        dPreu[:, self.Ny:, self.Ny:] = dP

        Q, Post = np.linalg.qr(self._Preu)
        tmp = np.linalg.solve(Post.T, dPreu.swapaxes(1, 2) @ Q).swapaxes(1, 2)
        dPost = (np.swapaxes(np.tril(tmp, -1), 1, 2) + np.triu(tmp)) @ Post

        K = Post[:self.Ny, self.Ny:].T
        S = Post[:self.Ny, :self.Ny]
        dS = dPost[:, :self.Ny, :self.Ny]

        if self.Ny > 1:
            e = np.linalg.solve(S, y - C @ x - D @ u)
            de = np.linalg.solve(-S, dS @ e + dC @ x + C @ dx + dD @ u)
        else:
            e = (y - C @ x - D @ u) / S
            de = -(dS @ e + dC @ x + C @ dx + dD @ u) / S

        x += K @ e
        dx += dPost[:, :self.Ny, self.Ny:].swapaxes(1, 2) @ e + K @ de
        P = Post[self.Ny:, self.Ny:]
        dP = dPost[:, self.Ny:, self.Ny:]

        return x, dx, P, dP, e, de, S, dS

    def log_likelihood(self, idx, Ad, B0d, B1d, Qd, C, D, R,
                       x0, P0, dt, u, u1, y):
        """Evaluate the negative log-likelihood

        Args:
            idx: Index of unique time intervals
            Ad: State matrix
            B0d, B1d: Input matrices
            Qd: Process noise covariance matrix
            C: Output matrix
            D: Feedthrough matrix
            R: Measurement deviation matrix
            x0: Prior state mean at time t0
            P0: Prior state deviation at time t0
            dt: Time interval
            u: Input array
            u1: Derivative input array
            y: Measurement array

        Returns:
            loglik: Negative log-likelihood
        """
        T = y.shape[1]
        x = x0.copy()
        P = P0.copy()
        loglik = 0.5 * T * self.Ny * self._LOG_2PI

        do_update = ~np.isnan(y).any(axis=0)

        for t in range(T):
            if do_update[t]:
                x, P, e, S = self.update(C, D, R, x, P, u[:, t:t + 1], y[:, t])

                if self.Ny > 1:
                    loglik += np.linalg.slogdet(S)[1] + 0.5 * e.T @ e
                else:
                    loglik += np.log(np.abs(S)) + 0.5 * e**2

            i = idx[t]
            x, P = self.prediction(
                Ad[i, :, :], B0d[i, :, :], B1d[i, :, :], Qd[i, :, :],
                x, P, dt[i], u[:, t:t + 1], u1[:, t:t + 1]
            )

        return loglik[0, 0]

    def dlog_likelihood(self, idx, Ad, dAd, B0d, dB0d, B1d, dB1d, Qd, dQd,
                        C, dC, D, dD, R, dR, x0, dx0, P0, dP0, dt, u, u1, y):
        """Evaluate the gradient of the negative log-likelihood

        Args:
            idx: Index of unique time intervals
            Ad: State matrix
            dAd: Derivative state matrix
            B0d, B1d: Input matrices
            dB0d, dB1d: Derivative input matrices
            Qd: Process noise covariance matrix
            dQd: Derivative process noise covariance matrix
            C: Output matrix
            dC: Derivative output matrix
            D: Feedthrough matrix
            dD: Derivative feedthrough matrix
            R: Measurement deviation matrix
            dR: Derivative measurement deviation matrix
            x0: Prior state mean at time t0
            dx0: Derivative prior state mean at time t0
            P0: Prior state deviation at time t0
            dP0: Derivative prior state deviation at time t0
            dt: Time interval array
            u: Input array
            u1: Derivative input array
            y: Measurement array

        Returns:
            loglik: Negative log-likelihood
            gradient: Gradient of the negative log-likelihood
        """
        T = y.shape[1]
        x = x0.copy()
        dx = dx0.copy()
        P = P0.copy()
        dP = dP0.copy()

        loglik = 0.5 * T * self.Ny * self._LOG_2PI
        gradient = np.zeros(dP0.shape[0])
        if self._compute_hessian:
            hessian = np.zeros((dP0.shape[0], dP0.shape[0]))

        do_update = ~np.isnan(y).any(axis=0)

        for t in range(T):
            if do_update[t]:
                x, dx, P, dP, e, de, S, dS = self.dupdate(
                    C, dC, D, dD, R, dR, x, dx, P, dP, u[:, t:t + 1], y[:, t]
                )

                if self.Ny > 1:
                    loglik += np.linalg.slogdet(S)[1] + 0.5 * e.T @ e
                    tmp = np.linalg.solve(S, dS)
                    gradient += tmp.trace(0, 1, 2) + np.squeeze(e.T @ de)
                else:
                    loglik += np.log(np.abs(S)) + 0.5 * e**2
                    gradient += np.squeeze(dS / S + e * de)
                    if self._compute_hessian:
                        hessian += np.outer(dS / S, dS / S) + np.outer(de, de)

            i = idx[t]
            x, dx, P, dP = self.dprediction(
                Ad[i, :, :], dAd[i, :, :, :], B0d[i, :, :], dB0d[i, :, :, :],
                B1d[i, :, :], dB1d[i, :, :, :], Qd[i, :, :], dQd[i, :, :, :],
                x, dx, P, dP, dt[i], u[:, t:t + 1], u1[:, t:t + 1]
            )

        if self._compute_hessian:
            return loglik[0, 0], gradient, hessian
        else:
            return loglik[0, 0], gradient

    def filtering(self, idx, Ad, B0d, B1d, Qd, C, D, R,
                  x0, P0, dt, u, u1, y):
        """Compute the filtered state distribution with the Kalman filter

        Args:
            idx: Index of unique time intervals
            Ad: State matrix
            B0d, B1d: Input matrices
            Qd: Process noise covariance matrix
            C: Output matrix
            D: Feedthrough matrix
            R: Measurement deviation matrix
            x0: Prior state mean at time t0
            P0: Prior state deviation at time t0
            dt: Time interval
            u: Input array
            u1: Derivative input array
            y: Measurement array

        Returns:
            xf: Filtered state distribution mean
            Pf: Filtered state distribution covariance
            res: Standardized residuals
            dev_res: Residuals deviation
        """
        T = y.shape[1]
        x = x0.copy()
        P = P0.copy()

        xf = np.empty((T, self.Nx, 1))
        Pf = np.empty((T, self.Nx, self.Nx))
        res = np.zeros((T, self.Ny, 1))
        dev_res = np.zeros((T, self.Ny, self.Ny))

        do_update = ~np.isnan(y).any(axis=0)

        for t in range(T):
            if do_update[t]:
                x, P, e, S = self.update(C, D, R, x, P, u[:, t:t + 1], y[:, t])
                res[t, :, :] = e
                dev_res[t, :, :] = S

            xf[t, :, :] = x
            Pf[t, :, :] = P.T @ P

            i = idx[t]
            x, P = self.prediction(
                Ad[i, :, :], B0d[i, :, :], B1d[i, :, :], Qd[i, :, :],
                x, P, dt[i], u[:, t:t + 1], u1[:, t:t + 1]
            )

        return xf, Pf, res, dev_res

    def smoothing(self, idx, Ad, B0d, B1d, Qd, C, D, R,
                  x0, P0, dt, u, u1, y):
        """Compute the smoothed state distribution Rauch-Tung-Striebel smoother

        Args:
            idx: Index of unique time intervals
            Ad: State matrix
            B0d, B1d: Input matrices
            Qd: Process noise covariance matrix
            C: Output matrix
            D: Feedthrough matrix
            R: Measurement deviation matrix
            x0: Prior state mean at time t0
            P0: Prior state deviation at time t0
            dt: Time interval
            u: Input array
            u1: Derivative input array
            y: Measurement array

        Returns:
            xs: Smoothed state distribution mean
            Ps: Smoothed state distribution covariance
        """
        T = y.shape[1]
        x = x0.copy()
        P = P0.copy()

        xp = np.empty((T, self.Nx, 1))
        xs = np.empty((T, self.Nx, 1))
        Pp = np.empty((T, self.Nx, self.Nx))
        Ps = np.empty((T, self.Nx, self.Nx))

        do_update = ~np.isnan(y).any(axis=0)

        # Run forward Kalman filter
        for t in range(T):
            # Save prior state distributoin
            xp[t, :, :] = x
            Pp[t, :, :] = P.T @ P

            if do_update[t]:
                x, P, _, _ = self.update(C, D, R, x, P, u[:, t:t + 1], y[:, t])

            # Save filtered state distribution
            xs[t, :, :] = x
            Ps[t, :, :] = P.T @ P

            i = idx[t]
            x, P = self.prediction(
                Ad[i, :, :], B0d[i, :, :], B1d[i, :, :], Qd[i, :, :],
                x, P, dt[i], u[:, t:t + 1], u1[:, t:t + 1]
            )

        # Run backward RTS smoother
        for t in reversed(range(y.shape[1] - 1)):
            # smoother gain (note: Pp and Ps are symmetric)
            G = np.linalg.solve(Pp[t + 1, :, :],
                                Ad[idx[t], :, :] @ Ps[t, :, :]).T

            # smoothed state mean and covariance
            xs[t, :, :] += G @ (xs[t + 1, :, :] - xp[t + 1, :, :])
            Ps[t, :, :] += G @ (Ps[t + 1, :, :] - Pp[t + 1, :, :]) @ G.T

        return xs, Ps
