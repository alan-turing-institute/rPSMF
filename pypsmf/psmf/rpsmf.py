# -*- coding: utf-8 -*-

import autograd.numpy as np
import mpmath as mp

from collections import defaultdict

from .psmf import PSMFIter


class rPSMFIter(PSMFIter):
    def __init__(
        self,
        theta0,
        C0,
        V0,
        mu0,
        P0,
        Q0,
        R0,
        lambda0,
        nonlinearity,
        fixed_lambda=False,
        use_scaling=False,
        optim="adam",
    ):
        assert optim in ["adam", "sgd"]
        super().__init__(
            theta0, C0, V0, mu0, P0, Q0, R0, nonlinearity, optim=optim
        )

        self.Q0 = Q0
        self.R0 = R0
        self.lambda0 = lambda0

        self.fixed_lambda = fixed_lambda
        if fixed_lambda:
            self._lambda = defaultdict(lambda: lambda0)
        else:
            self._lambda = {0: lambda0}

        self._Q = {0: Q0}
        self._R = {0: R0}

        self._alpha = 1.0
        self._beta = 1.0
        if use_scaling:
            self._alpha = self.compute_scaling_factor(
                self._r * self._d, self._d
            )
            self._beta = self.compute_scaling_factor(self._r, self._d)

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

    def compute_scaling_factor(self, dim, offset, verbose=False):
        # compute the scaling factor by minimizing the KL-divergence between
        # two multivariate t distributions of dimension `dim` where one has
        # covariance I and dof lambda + offset and the other has covariance
        # alpha * I and dof lambda, against alpha.

        # We use mpmath here because the high dimensionality leads to precision
        # errors in scipy.

        def _integrand(v, alpha, m, lmd, d):
            return (
                mp.power(v / (1 + v), m / 2)
                * 1.0
                / (v * mp.power(1 + v, (lmd + d) / 2.0))
                * mp.log(1 + (lmd + d) / (alpha * lmd) * v)
            )

        def _H(alpha, m, lmd, d):
            H2 = mp.beta(m / 2, (lmd + d) / 2) * mp.log(alpha)
            Q = mp.quad(lambda v: _integrand(v, alpha, m, lmd, d), [0, mp.inf])
            H3 = (1 + lmd / m) * Q
            return H2 + H3

        m = mp.mpf(dim)
        lmd = mp.mpf(self.lambda0)
        d = mp.mpf(offset)
        F = lambda alpha: _H(alpha, m, lmd, d)
        dF = lambda alpha: mp.diff(F, alpha)
        alpha_star = mp.findroot(dF, mp.mpf(1.0), verbose=verbose)
        return float(alpha_star)

    def step_reset(self):
        super().step_reset()
        if self.fixed_lambda:
            self._lambda = defaultdict(lambda: self.lambda0)
        else:
            self._lambda = {0: self.lambda0}

        self._R = {0: self.R0}
        self._Q = {0: self.Q0}

    def _predictive_covariance(self, i, k):
        F = self._jacfunc(self._theta[i - 1], self._mu[k - 1], k)
        # The output of the Jacobian is (r, 1, r, 1), since the input, mu, to
        # the nonlinearity is (r, 1) and it outputs an array of the same size.
        # We need F to be an (r, r) matrix, so we squeeze out the second and
        # fourth axes.
        F = F.squeeze(axis=1).squeeze(axis=2)
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
        self._V[k] = self._alpha * phi_k * (self._V[k - 1] - num / Nk)

    def _compute_inverse_coefficient_innovation(self, k, mu_bar, P_bar):
        Rbar = self._R[k - 1] + np.kron(
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

    def _update_coefficient_covariance(self, k, Skinv, P_bar, yk):
        omega_k = (
            self._lambda[k - 1]
            + (yk - self._y_pred[k]).T @ Skinv @ (yk - self._y_pred[k])
        ) / (self._lambda[k - 1] + self._d)
        self._P[k] = (
            self._beta
            * omega_k
            * (
                P_bar
                - P_bar @ self._C[k - 1].T @ Skinv @ self._C[k - 1] @ P_bar
            )
        )
        self._Q[k] = omega_k * self._Q[k - 1]
        self._R[k] = omega_k * self._R[k - 1]
        if not self.fixed_lambda:
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
        self._V[k] = self._alpha * phi_k * (self._V[k - 1] - num / Nk)

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
        self._P[k] = (
            self._beta
            * omega_k
            * (P_bar - P_bar @ MkC.T @ Skinv @ MkC @ P_bar)
        )
        self._Q[k] = omega_k * self._Q[k - 1]
        self._R[k] = omega_k * self._R[k - 1]
        if not self.fixed_lambda:
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


class rPSMFRecursive(rPSMFIter):
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
        self._theta[i] = self._theta[i - 1]

    def predict(self, T, n_pred):
        last_theta = self._theta[T]
        self._mu_pred = {T: self._mu[T]}
        for k in range(T + 1, T + n_pred + 1):
            self._mu_pred[k] = self.nonlinearity(
                last_theta, self._mu_pred[k - 1], k
            )
            self._y_pred[k] = self._C[T] @ self._mu_pred[k]
