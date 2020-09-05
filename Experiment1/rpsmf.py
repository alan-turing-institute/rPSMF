# -*- coding: utf-8 -*-

"""PSMF Experiment 1 - Base class for rPSMF (iterative version)

This file is part of the PSMF codebase.
See the LICENSE file for copyright and licensing information.

"""

import autograd.numpy as np

from psmf import PSMFIter


class rPSMFIter(PSMFIter):
    def __init__(
        self, theta0, C0, V0, mu0, P0, Q0, R0, lambda0, nonlinearity,
    ):
        super().__init__(theta0, C0, V0, mu0, P0, Q0, R0, nonlinearity)

        self.Q0 = Q0
        self.R0 = R0
        self.lambda0 = lambda0
        self._lambda = {0: lambda0}

        self._Q = {0: Q0}
        self._R = {0: R0}

    def incremental_ll_factory(self):
        # Returns the incremental likelihood function
        def normnon(theta, mu, V, t):
            return (
                self.nonlinearity(theta, mu, t).T
                @ V
                @ self.nonlinearity(theta, mu, t)
            )

        def incremental_ll(theta, mu, y, C, V, eta_k, lmd, d, t):
            return 0.5 * d * np.log(normnon(theta, mu, V, t) + eta_k) + (
                (d + lmd) / 2.0
            ) * np.log(
                1
                + np.power(
                    np.linalg.norm(y - C @ self.nonlinearity(theta, mu, t)), 2
                )
                / (lmd * (eta_k + normnon(theta, mu, V, t)))
            )

        return incremental_ll

    def step_reset(self):
        super().step_reset()
        self._lambda = {0: self.lambda0}

        self._R = {0: self.R0}
        self._Q = {0: self.Q0}

    def _predictive_covariance(self, i, k):
        F = self._jacfunc(self._mu[k - 1], self._theta[i - 1], k)
        F = F.squeeze()
        return F @ self._P[k - 1] @ F.T + self._Q[k - 1]

    def _compute_eta_k(self, k, P_bar):
        return (
            np.trace(
                self._R[k - 1] + self._C[k - 1] @ P_bar @ self._C[k - 1].T
            )
            / self._d
        )

    def _update_dictionary_covariance(self, k, Nk, mu_bar, yk):
        num = self._V[k - 1] @ mu_bar @ mu_bar.T @ self._V[k - 1]
        phi_k = self._lambda[k - 1] / (self._lambda[k - 1] + self._d) + (
            yk - self._y_pred[k]
        ).T @ (yk - self._y_pred[k]) / ((self._lambda[k - 1] + self._d) * Nk)
        self._V[k] = phi_k * (self._V[k - 1] - num / Nk)

    def _compute_inverse_coefficient_innovation(self, k, mu_bar, P_bar):
        Rbar = self._R[k - 1] + np.kron(
            mu_bar.T @ self._V[k - 1] @ mu_bar, np.eye(self._d)
        )
        Sk = self._C[k - 1] @ P_bar @ self._C[k - 1].T + Rbar
        return np.linalg.inv(Sk)

    def _update_coefficient_covariance(self, k, Skinv, P_bar, yk):
        omega_k = (
            self._lambda[k - 1]
            + (yk - self._y_pred[k]).T @ Skinv @ (yk - self._y_pred[k])
        ) / (self._lambda[k - 1] + self._d)
        self._P[k] = omega_k * (
            P_bar - P_bar @ self._C[k - 1].T @ Skinv @ self._C[k - 1] @ P_bar
        )
        # Updating lambda, Q and R here is a bit of a side-effect, but it's
        # okay for now
        self._Q[k] = omega_k * self._Q[k - 1]
        self._R[k] = omega_k * self._R[k - 1]
        self._lambda[k] = self._lambda[k - 1] + self._d

    def _store_gradient(self, i, k, yk, eta_k):
        self._gradsum += self._gradfunc(
            self._theta[i - 1],
            self._mu[k - 1],
            yk,
            self._C[k - 1],
            self._V[k - 1],
            eta_k,
            self._lambda[k - 1],
            self._d,
            k,
        )


class rPSMFIterMissing(rPSMFIter):
    def incremental_ll_factory(self):
        def normnon(theta, mu, V, t):
            return (
                self.nonlinearity(theta, mu, t).T
                @ V
                @ self.nonlinearity(theta, mu, t)
            )

        def incremental_ll(theta, mu, z, M, C, V, eta_k, lmd, d, t):
            U = normnon(theta, mu, V, t) * M + eta_k * np.eye(d)
            Ui = np.diag(1.0 / np.diag(U))
            diff = z - M @ C @ self.nonlinearity(theta, mu, t)
            return 0.5 * np.trace(np.log(U)) + 0.5 * diff.T @ Ui @ diff

        return incremental_ll

    def step(self, y, m, i, T):
        self.step_reset()
        for k in range(1, T + 1):
            Mk = np.diag(m[k].squeeze())
            self.inner(i, k, y[k], Mk)

    def inner(self, i, k, yk, Mk):
        # NOTE: We assume the missing elements in yk are set to zero, so zk and
        # yk are interchangeable.
        mu_bar = self._predictive_mean(i, k)
        P_bar = self._predictive_covariance(i, k)
        MkC = Mk @ self._C[k - 1]
        self._y_pred[k] = self._predict_measurement(k, mu_bar, MkC)
        eta_k = self._compute_eta_k(k, P_bar, Mk, MkC)
        Nk = self._compute_dictionary_innovation(k, eta_k, mu_bar, P_bar)
        self._update_dictionary_mean(k, yk, Nk, mu_bar)
        self._update_dictionary_covariance(k, Nk, mu_bar, yk, Mk, eta_k)
        Skinv = self._compute_inverse_coefficient_innovation(
            k, mu_bar, P_bar, Mk, MkC
        )
        self._update_coefficient_mean(k, yk, MkC, Skinv, mu_bar, P_bar)
        self._update_coefficient_covariance(k, Skinv, P_bar, yk, MkC)
        self._store_gradient(i, k, yk, eta_k)
        self._prune(k)

    def _predict_measurement(self, k, mu_bar, MkC):
        return MkC @ mu_bar

    def _compute_eta_k(self, k, P_bar, Mk, MkC):
        return (
            np.trace(Mk @ self._R[k - 1] @ Mk.T + MkC @ P_bar @ MkC.T)
            / self._d
        )

    def _update_dictionary_covariance(self, k, Nk, mu_bar, yk, Mk, eta_k):
        num = self._V[k - 1] @ mu_bar @ mu_bar.T @ self._V[k - 1]
        U = mu_bar.T @ self._V[k - 1] @ mu_bar * Mk + eta_k * np.eye(self._d)
        Ui = np.diag(1.0 / np.diag(U))  # faster than np.linalg.inv (tested)
        phi_k = (
            self._lambda[k - 1]
            + (yk - self._y_pred[k]).T @ Ui @ (yk - self._y_pred[k])
        ) / (self._lambda[k - 1] + self._d)
        self._V[k] = phi_k * (self._V[k - 1] - num / Nk)

    def _compute_inverse_coefficient_innovation(
        self, k, mu_bar, P_bar, Mk, MkC
    ):
        Rbar = Mk @ self._R[k - 1] @ Mk.T + np.kron(
            mu_bar.T @ self._V[k - 1] @ mu_bar, Mk
        )
        Sk = MkC @ P_bar @ MkC.T + Rbar
        return np.linalg.inv(Sk)

    def _update_coefficient_mean(self, k, yk, MkC, Skinv, mu_bar, P_bar):
        self._mu[k] = mu_bar + P_bar @ MkC.T @ Skinv @ (yk - self._y_pred[k])

    def _update_coefficient_covariance(self, k, Skinv, P_bar, yk, MkC):
        omega_k = (
            self._lambda[k - 1]
            + (yk - self._y_pred[k]).T @ Skinv @ (yk - self._y_pred[k])
        ) / (self._lambda[k - 1] + self._d)
        self._P[k] = omega_k * (P_bar - P_bar @ MkC.T @ Skinv @ MkC @ P_bar)
        self._Q[k] = omega_k * self._Q[k - 1]
        self._R[k] = omega_k * self._R[k - 1]
        self._lambda[k] = self._lambda[k - 1] + self._d

    def _store_gradient(self, i, k, yk, Mk, eta_k):
        self._gradsum += self._gradfunc(
            self._theta[i - 1],
            self._mu[k - 1],
            yk,
            Mk,
            self._C[k - 1],
            self._V[k - 1],
            eta_k,
            self._lambda[k - 1],
            self._d,
            k,
        )
