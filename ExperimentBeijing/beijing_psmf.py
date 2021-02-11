#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import autograd.numpy as np
import matplotlib.pyplot as plt
import time

from psmf import PSMFIter
from psmf.nonlinearities import RandomWalk, FourierBasis
from psmf.tracking import TrackingMixin

plt.rcParams["figure.raise_window"] = False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input", help="Input data file (csv)", required=True
    )
    parser.add_argument(
        "--figure",
        help="Figure to generate",
        choices=[
            "periodic",
            "random_walk",
        ],
        required=True,
    )
    parser.add_argument(
        "--output-bases",
        help="Output file for bases figure",
        default="bases.pdf",
    )
    parser.add_argument(
        "--output-cost",
        help="Output file for the cost figure",
        default="cost.pdf",
    )
    parser.add_argument(
        "--output-fit",
        help="Output file for the fit figure",
        default="fit.pdf",
    )
    parser.add_argument(
        "-v", "--verbose", help="Enable verbose mode", action="store_true"
    )
    parser.add_argument(
        "-l", "--live-plot", help="Enable live plotting", action="store_true"
    )
    return parser.parse_args()


class PSMF(TrackingMixin, PSMFIter):
    def run(
        self,
        y_full,
        y_train,
        T,
        n_iter,
        n_pred,
        adam_gam=1e-3,
        live_plot=False,
        verbose=True,
    ):
        self.adam_init(gam=adam_gam)
        self.errors_init(y_full, T, n_iter, n_pred)
        self.log(0, n_iter, 0, verbose=verbose)
        self.figures_init(live_plot=live_plot)

        for i in range(1, n_iter + 1):
            t_start = time.time()
            self.step(y_train, i, T)
            self.predict(i, T, n_pred)
            self.adam_update(i, project=True)

            self.errors_update(i, y_full, T, n_pred)
            self.log(i, n_iter, time.time() - t_start, verbose=verbose)
            if i % 10 == 0:
                self.figures_update(y_full, T, n_pred, live_plot=live_plot)

    def step_reset(self):
        super().step_reset()
        # We re-initialize V at every iteration for this experiment.
        self._V = {0: self.V0}

    def _prune(self, k):
        del self._C[k - 1], self._V[k - 1], self._P[k - 1]


def load_data(filename, sample=True):
    Y = np.genfromtxt(filename, delimiter=",")
    Y = ((Y.T - Y.T.mean(axis=0)) / (Y.T.std(axis=0))).T
    return Y[:, ::100] if sample else Y


def main():
    args = parse_args()

    # Hyperparameters
    seed = 2151  # fixed for reproducibility
    rank = 1
    learning_rate = 1e-3
    theta_scale = 0.1
    c_scale = v_scale = 5
    p_scale = q_scale = r_scale = 1
    n_iter = 100

    if args.figure == "periodic":
        nonlinearity = FourierBasis(rank=rank, N=1)
    elif args.figure == "random_walk":
        nonlinearity = RandomWalk()
    else:
        raise ValueError("Unknown figure request: {args.figure}")

    np.random.seed(seed)
    Y = load_data(args.input, sample=True)

    d, T = Y.shape
    n_pred = int(0.2 * T)

    y = {k + 1: Y[:, k, None] for k in range(T)}
    y_train = {k: y[k] for k in range(1, T - n_pred + 1)}
    T = len(y_train)
    y_full = y

    C0 = c_scale * np.random.randn(d, rank)
    theta0 = theta_scale * np.random.rand(nonlinearity.n_params, 1)
    V0 = np.kron(v_scale, np.eye(rank))
    mu0 = np.zeros([rank, 1])
    P0 = p_scale * np.eye(rank)
    Q0 = q_scale * np.eye(rank)
    R0 = r_scale * np.eye(d)
    Qs = {k: Q0 for k in range(T + 1)}
    Rs = {k: R0 for k in range(T + 1)}

    psmf = PSMF(theta0, C0, V0, mu0, P0, Qs, Rs, nonlinearity)
    psmf.run(
        y_full,
        y_train,
        T,
        n_iter,
        n_pred,
        adam_gam=learning_rate,
        live_plot=args.live_plot,
        verbose=args.verbose,
    )

    output_files = {
        "bases": args.output_bases,
        "cost_y": args.output_cost,
        "fit": args.output_fit,
    }
    psmf.figures_save(
        y, n_pred, T, output_files=output_files, fit_figsize=(8, 2)
    )


if __name__ == "__main__":
    main()
