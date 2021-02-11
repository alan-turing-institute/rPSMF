#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import autograd.numpy as np
import sys
import time

from data import generate_normal_data
from psmf import PSMFRecursive
from psmf.tracking import TrackingMixin


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v", "--verbose", help="Enable verbose mode", action="store_true"
    )
    parser.add_argument(
        "-p", "--live-plot", help="Enable live plotting", action="store_true"
    )
    parser.add_argument("-s", "--seed", help="Random seed to use", type=int)
    parser.add_argument(
        "--output-bases",
        help="Output file for bases figure",
        default="bases.pdf",
    )
    parser.add_argument(
        "--output-cost-y",
        help="Output file for the cost (y) figure",
        default="cost_y.pdf",
    )
    parser.add_argument(
        "--output-cost-theta",
        help="Output file for the cost (theta) figure",
        default="cost_theta.pdf",
    )
    parser.add_argument(
        "--output-fit",
        help="Output file for the fit figure",
        default="fit.pdf",
    )
    return parser.parse_args()


class PSMFRecursiveSynthetic(TrackingMixin, PSMFRecursive):
    def run(
        self,
        y,
        y_true,
        y_obs,
        theta_true,
        C_true,
        x_true,
        T,
        n_pred,
        adam_gam=1e-6,
        live_plot=False,
        verbose=True,
    ):
        self.adam_init(gam=adam_gam)
        self.errors_init(y_obs, T, 1, n_pred, theta_true=theta_true)
        self.figures_init(live_plot=live_plot)
        self.log(0, T, 0, verbose=verbose)

        self.__x_true = x_true
        self.__y_true = y_true
        self.__y_obs = y_obs
        self.__theta_true = theta_true
        self.__T = T
        self.__n_pred = n_pred
        self.__live_plot = live_plot

        self.step(y, T)
        self.predict(T, n_pred)

    def inner(self, k, yk):
        t_start = time.time()

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
        self._set_gradient(k, yk, eta_k)
        self.adam_update(k)

        self.predict(k, self.__T + self.__n_pred)

        self.errors_update(
            k,
            self.__y_true,
            self.__T,
            self.__n_pred,
            theta_true=self.__theta_true,
        )
        self.log(k, self.__T, time.time() - t_start, verbose=True)
        if k % 100 == 0:
            self.figures_update(
                self.__y_obs,
                self.__T,
                self.__n_pred,
                live_plot=self.__live_plot,
                x_true=self.__x_true,
            )

    ### Algorithmic simplifications

    def step_reset(self):
        # TODO: This has no effect
        super().step_reset()
        # We re-initialize V at every iteration for this experiment.
        self._V = {0: self.V0}

    def _predictive_covariance(self, i, k):
        return self._P[k - 1]

    def _compute_eta_k(self, k, P_bar):
        return np.trace(self._R[k - 1]) / self._d

    def _compute_inverse_coefficient_innovation(self, k, mu_bar, P_bar):
        # P is assumed 0
        pass

    def _update_coefficient_mean(self, k, yk, Skinv, mu_bar, P_bar):
        self._mu[k] = mu_bar

    def _update_coefficient_covariance(self, k, Skinv, P_bar, yk):
        self._P[k] = P_bar

    def _prune(self, k):
        del self._C[k - 1], self._V[k - 1], self._P[k - 1]

    def _update_dictionary_covariance(self, k, Nk, mu_bar, yk):
        super()._update_dictionary_covariance(k, Nk, mu_bar, yk)

    ### End algorithmic simplifications


def nonlinearity(theta, x, t):
    return np.cos(2 * np.pi * theta * t + x)


def main():
    args = parse_args()
    seed = args.seed or np.random.randint(10000)
    print("Using seed: %r" % seed)
    np.random.seed(seed)

    d = 10
    r = 3
    T = 4000
    n_pred = 800
    var = 0.01

    data = generate_normal_data(
        nonlinearity, d=d, T=T, n_pred=n_pred, r=r, var=var
    )

    C0 = 0.1 * np.random.randn(d, r)
    theta0 = 1e-3 * np.random.rand(r, 1)
    v0 = 0.1
    V0 = np.kron(v0, np.eye(r))
    mu0 = np.zeros([r, 1])
    P0 = np.zeros([r, r])
    Qs = {k: 0 * np.identity(r) for k in range(T + 1)}
    Rs = {k: np.identity(d) for k in range(T + 1)}

    psmf = PSMFRecursiveSynthetic(
        theta0, C0, V0, mu0, P0, Qs, Rs, nonlinearity
    )
    psmf.run(
        data["y_train"],
        data["y_true"],
        data["y_obs"],
        data["theta_true"],
        data["C_true"],
        data["x_true"],
        T,
        n_pred,
        adam_gam=1e-6,
        live_plot=args.live_plot,
        verbose=args.verbose,
    )
    output_files = dict(
        fit=args.output_fit,
        bases=args.output_bases,
        cost_y=args.output_cost_y,
        cost_theta=args.output_cost_theta,
    )
    psmf.figures_save(
        data["y_obs"],
        n_pred,
        T,
        x_true=data["x_true"],
        output_files=output_files,
        fit_figsize=(8, 2)
    )
    psmf.figures_close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(1)
