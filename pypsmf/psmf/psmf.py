# -*- coding: utf-8 -*-

"""Base class for PSMF - iterative version
"""

import autograd.numpy as np

from autograd import grad, jacobian

from .learning_rate import BaseLearningRate
from .learning_rate import ConstantLearningRate


class PSMFIter:

    """Iterative version of PSMF"""

    def __init__(
        self, theta0, C0, V0, mu0, P0, Qs, Rs, nonlinearity, optim="adam"
    ):
        self.nonlinearity = nonlinearity
        assert optim in ["adam", "sgd"]
        self.optim = optim

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
        # note: argnum=1 because the derivative is to x, which is expected as
        # the second argument of nonlinearity().
        self._jacfunc = jacobian(self.nonlinearity, argnum=1)

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
        self.optim_init()
        for i in range(1, n_iter + 1):
            self.step(y, i, T)
            self.predict(i, T, n_pred)
            self.optim_update(i)

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
        F = self._jacfunc(self._theta[i - 1], self._mu[k - 1], k)

        # The output of the Jacobian is (r, 1, r, 1), since the input, mu, to
        # the nonlinearity is (r, 1) and it outputs an array of the same size.
        # We need F to be an (r, r) matrix, so we squeeze out the second and
        # fourth axes.
        F = F.squeeze(axis=1).squeeze(axis=2)
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
        if np.all(Rbar == np.diag(np.diagonal(Rbar))) and Rbar.sum() > 0:
            Ri = np.diag(1 / np.diag(Rbar))
            RiC = Ri @ self._C[k - 1]
            Pi = np.linalg.inv(P_bar)
            PiCRiC = Pi + self._C[k - 1].T @ RiC
            Skinv = Ri - RiC @ np.linalg.inv(PiCRiC) @ RiC.T
        else:
            Sk = self._C[k - 1] @ P_bar @ self._C[k - 1].T + Rbar
            Skinv = np.linalg.inv(Sk)
        return Skinv

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
        if isinstance(gam, BaseLearningRate):
            self.adam_gam = gam
        else:
            self.adam_gam = ConstantLearningRate(gam)

        self.adam_b1 = b1
        self.adam_b2 = b2

        self.adam_m = np.zeros(self.theta0.shape)
        self.adam_v = np.zeros(self.theta0.shape)

        self.adam_m_hat = np.zeros(self.theta0.shape)
        self.adam_v_hat = np.zeros(self.theta0.shape)

    def sgd_init(self, gam=1e-3):
        if isinstance(gam, BaseLearningRate):
            self.sgd_gam = gam
        else:
            self.sgd_gam = ConstantLearningRate(gam)

    def optim_init(self, gam=1e-3):
        if self.optim == "adam":
            self.adam_init(gam=gam)
        elif self.optim == "sgd":
            self.sgd_init(gam=gam)

    def optim_update(self, i, project=True):
        if self.optim == "adam":
            return self.adam_update(i, project=project)
        elif self.optim == "sgd":
            return self.sgd_update(i, project=project)

    def adam_update(self, i, project=True):
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
        lr = self.adam_gam.get(i)
        self._theta[i] = self._theta[i - 1] - lr * np.multiply(
            pr, self.adam_m_hat
        )
        if project:
            self._theta[i] = np.maximum(self._theta[i], 0)

    def sgd_update(self, i, project=True):
        lr = self.sgd_gam.get(i)
        self._theta[i] = self._theta[i - 1] - lr * self._gradsum
        if project:
            self._theta[i] = np.maximum(self._theta[i], 0)


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


class PSMFRecursive(PSMFIter):
    def run(self, y, T, n_pred, update_every=1):
        self._update_every = update_every
        self.optim_init()
        self.step(y, T)
        self.predict(T, n_pred)

    def step(self, y, T):
        self.step_reset()
        for k in range(1, T + 1):
            self.inner(k, y[k])

    def inner(self, k, yk):
        mu_bar = self._predictive_mean(k, k)
        P_bar = self._predictive_covariance(k, k)
        self._y_pred[k] = self._predict_measurement(k, mu_bar)
        eta_k = self._compute_eta_k(k, P_bar)
        Nk = self._compute_dictionary_innovation(k, eta_k, mu_bar, P_bar)
        self._update_dictionary_mean(k, yk, Nk, mu_bar)
        self._update_dictionary_covariance(k, Nk, mu_bar, yk)
        Skinv = self._compute_inverse_coefficient_innovation(k, mu_bar, P_bar)
        self._update_coefficient_mean(k, yk, Skinv, mu_bar, P_bar)
        self._update_coefficient_covariance(k, Skinv, P_bar, yk)

        self._store_gradient(k, k, yk, eta_k)
        if k % self._update_every == 0:
            self.optim_update(k)
            self._reset_gradient()
        else:
            self._carry_theta(k)

    def _reset_gradient(self):
        self._gradsum = np.zeros(self.theta0.shape)

    def _carry_theta(self, i):
        self._theta[i] = self._theta[i-1]

    def _set_gradient(self, k, yk, eta_k):
        self._gradsum = self._gradfunc(
            self._theta[k - 1],
            self._mu[k - 1],
            yk,
            self._C[k - 1],
            self._V[k - 1],
            eta_k,
            self._d,
            k,
        )

    def predict(self, T, n_pred):
        last_theta = self._theta[T]
        self._mu_pred = {T: self._mu[T]}
        for k in range(T + 1, T + n_pred + 1):
            self._mu_pred[k] = self.nonlinearity(
                last_theta, self._mu_pred[k - 1], k
            )
            self._y_pred[k] = self._C[T] @ self._mu_pred[k]
