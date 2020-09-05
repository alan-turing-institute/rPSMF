# -*- coding: utf-8 -*-

"""PSMF Experiment 1 - Base class for PSMF (iterative version)

This file is part of the PSMF codebase.
See the LICENSE file for copyright and licensing information.

"""

import autograd.numpy as np

from autograd import grad, jacobian


class PSMFIter:

    """Iterative version of PSMF"""

    def __init__(self, theta0, C0, V0, mu0, P0, Qs, Rs, nonlinearity):
        self.nonlinearity = nonlinearity

        self.theta0 = theta0
        self.C0 = C0
        self.V0 = V0
        self.mu0 = mu0
        self.P0 = P0

        # estimated quantities are prefixed with an underscore
        self._d, self._r = C0.shape
        self._C = {}
        self._P = {}
        self._Q = Qs
        self._R = Rs
        self._V = {}
        self._mu = {}
        self._theta = {0: theta0}

        self._gradfunc = grad(self.incremental_ll_factory())
        self._jacfunc = jacobian(self.nonlinearity)

        self._y_pred = {}

    def incremental_ll_factory(self):
        # Returns the incremental likelihood function
        def normnon(theta, mu, V, t):
            return (
                self.nonlinearity(theta, mu, t).T
                @ V
                @ self.nonlinearity(theta, mu, t)
            )

        def incremental_ll(theta, mu, y, C, V, eta_k, d, t):
            return 0.5 * d * np.log(
                normnon(theta, mu, V, t) + eta_k
            ) + 0.5 * np.power(
                np.linalg.norm(y - C @ self.nonlinearity(theta, mu, t)), 2
            ) / (
                eta_k + normnon(theta, mu, V, t)
            )

        return incremental_ll

    def run(self, y, T, n_iter, n_pred):
        self.adam_init()
        for i in range(1, n_iter + 1):
            self.step(y, i, T)
            self.predict(i, T, n_pred)
            self.adam_update(i)

    def step_reset(self):
        # Get the last value from the previous iteration or the initial value
        latest = lambda D, init: D[sorted(D.keys())[-1]] if D else init

        self._C = {0: latest(self._C, self.C0)}
        self._mu = {0: latest(self._mu, self.mu0)}
        self._P = {0: latest(self._P, self.P0)}
        self._V = {0: latest(self._V, self.V0)}
        self._gradsum = np.zeros(self.theta0.shape)

    def step(self, y, i, T):
        self.step_reset()
        for k in range(1, T + 1):
            self.inner(i, k, y[k])

    def inner(self, i, k, yk):
        mu_bar = self._predictive_mean(i, k)
        P_bar = self._predictive_covariance(i, k)
        self._y_pred[k] = self._predict_measurement(k, mu_bar)
        eta_k = self._compute_eta_k(k, P_bar)
        Nk = self._compute_dictionary_innovation(k, eta_k, mu_bar, P_bar)
        self._update_dictionary_mean(k, yk, Nk, mu_bar)
        self._update_dictionary_covariance(k, Nk, mu_bar, yk)
        Skinv = self._compute_inverse_coefficient_innovation(k, mu_bar, P_bar)
        self._update_coefficient_mean(k, yk, Skinv, mu_bar, P_bar)
        self._update_coefficient_covariance(k, Skinv, P_bar, yk)
        self._store_gradient(i, k, yk, eta_k)
        self._prune(k)

    def _predictive_mean(self, i, k):
        return self.nonlinearity(self._theta[i - 1], self._mu[k - 1], k)

    def _predictive_covariance(self, i, k):
        F = self._jacfunc(self._mu[k - 1], self._theta[i - 1], k)
        F = F.squeeze()
        return F @ self._P[k - 1] @ F.T + self._Q[k]

    def _predict_measurement(self, k, mu_bar):
        # Predicted measurement
        return self._C[k - 1] @ mu_bar

    def _compute_eta_k(self, k, P_bar):
        return (
            np.trace(self._R[k] + self._C[k - 1] @ P_bar @ self._C[k - 1].T)
            / self._d
        )

    def _compute_dictionary_innovation(self, k, eta_k, mu_bar, P_bar):
        return mu_bar.T @ self._V[k - 1] @ mu_bar + eta_k

    def _update_dictionary_mean(self, k, yk, Nk, mu_bar):
        # Perform mean update of the dictionary
        num = (yk - self._y_pred[k]) @ mu_bar.T @ self._V[k - 1].T
        self._C[k] = self._C[k - 1] + num / Nk

    def _update_dictionary_covariance(self, k, Nk, mu_bar, yk):
        # Perform covariance update of the dictionary
        num = self._V[k - 1] @ mu_bar @ mu_bar.T @ self._V[k - 1]
        self._V[k] = self._V[k - 1] - num / Nk

    def _compute_inverse_coefficient_innovation(self, k, mu_bar, P_bar):
        Rbar = self._R[k] + np.kron(
            mu_bar.T @ self._V[k - 1] @ mu_bar, np.eye(self._d)
        )
        Sk = self._C[k - 1] @ P_bar @ self._C[k - 1].T + Rbar
        return np.linalg.inv(Sk)

    def _update_coefficient_mean(self, k, yk, Skinv, mu_bar, P_bar):
        # Mean update of x_k
        self._mu[k] = mu_bar + P_bar @ self._C[k - 1].T @ Skinv @ (
            yk - self._y_pred[k]
        )

    def _update_coefficient_covariance(self, k, Skinv, P_bar, yk):
        # Covariance update of x_k
        self._P[k] = (
            P_bar - P_bar @ self._C[k - 1].T @ Skinv @ self._C[k - 1] @ P_bar
        )

    def _store_gradient(self, i, k, yk, eta_k):
        self._gradsum += self._gradfunc(
            self._theta[i - 1],
            self._mu[k - 1],
            yk,
            self._C[k - 1],
            self._V[k - 1],
            eta_k,
            self._d,
            k,
        )

    def _prune(self, k):
        del self._C[k - 1], self._V[k - 1], self._mu[k - 1], self._P[k - 1]

    def predict(self, i, T, n_pred):
        self._mu_pred = {T: self._mu[T]}
        for k in range(T + 1, T + n_pred + 1):
            self._mu_pred[k] = self.nonlinearity(
                self._theta[i - 1], self._mu_pred[k - 1], k
            )
            self._y_pred[k] = self._C[T] @ self._mu_pred[k]

    def adam_init(self, gam=1e-3, b1=0.9, b2=0.999):
        """ Initialize adam """
        self.adam_gam = gam
        self.adam_b1 = b1
        self.adam_b2 = b2

        self.adam_m = np.zeros(self.theta0.shape)
        self.adam_v = np.zeros(self.theta0.shape)

        self.adam_m_hat = np.zeros(self.theta0.shape)
        self.adam_v_hat = np.zeros(self.theta0.shape)

    def adam_update(self, i):
        self.adam_m = (
            self.adam_b1 * self.adam_m + (1 - self.adam_b1) * self._gradsum
        )
        self.adam_v = self.adam_b2 * self.adam_v + (
            1 - self.adam_b2
        ) * np.multiply(self._gradsum, self._gradsum)
        self.adam_m_hat = self.adam_m / (1 - np.power(self.adam_b1, i))
        self.adam_v_hat = self.adam_v / (1 - np.power(self.adam_b2, i))

        pr = np.divide(
            np.ones(self.theta0.shape), np.sqrt(self.adam_v_hat) + 1e-8
        )
        self._theta[i] = np.maximum(
            self._theta[i - 1]
            - self.adam_gam * np.multiply(pr, self.adam_m_hat),
            0,
        )


class PSMFIterMissing(PSMFIter):
    def __init__(*args, **kwargs):
        # WORK IN PROGRESS, DO NOT USE
        raise NotImplementedError

    def incremental_ll_factory(self):
        def normnon(theta, mu, V, t):
            return (
                self.nonlinearity(theta, mu, t).T
                @ V
                @ self.nonlinearity(theta, mu, t)
            )

        def incremental_ll(theta, mu, z, m, C, V, eta_k, d, t):
            # m is the missing mask for y.
            M = np.diag(m)
            U = normnon(theta, mu, V, t) * M + eta_k * np.eye(d)
            Ui = np.diag(1.0 / np.diag(U))
            diff = z - M @ C @ self.nonlinearity(theta, mu, t)
            return 0.5 * np.trace(np.log(U)) + 0.5 * diff.T @ Ui @ diff

        return incremental_ll
