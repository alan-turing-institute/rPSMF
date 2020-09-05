# -*- coding: utf-8 -*-

"""PSMF Experiment 1 - Tracking mixin

This file is part of the PSMF codebase.
See the LICENSE file for copyright and licensing information.

"""

import autograd.numpy as np
import matplotlib
import matplotlib.pyplot as plt


plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{sansmath}\sansmath'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Helvetica'


class TrackingMixin:
    """Methods for tracking and plotting various quantities during runs"""

    def log(self, i, n_iter, delta_t, verbose=True, prefix=""):
        if not verbose:
            return
        s = "[%s%03i/%i]: ||y* - Cx||^2 = %.8f, ||θ* - θ||^2 = %.8f, Δt = %.3f"
        d = (
            prefix,
            i,
            n_iter,
            self._E_y[i],
            self._E_theta[i],
            delta_t,
        )
        l = s % d
        if not hasattr(self, "_logs"):
            self._logs = []
        self._logs.append(l)
        if not verbose:
            return
        print(s % d, flush=True)

    def logs_save(self, filename):
        if not hasattr(self, "_logs"):
            return
        with open(filename, "w") as fp:
            fp.write("\n".join(self._logs))

    def errors_init(self, y_true, theta_true, T, n_iter, n_pred):
        self._E_y = {}
        self._E_theta = {}

        Y_pred = self.C0 @ np.zeros((self._r, T + n_pred))
        Y_real = np.array([y_true[k] for k in range(1, T + n_pred + 1)])
        Y_real = Y_real.squeeze().T

        self._E_y[0] = np.linalg.norm(Y_pred - Y_real).item()
        self._E_theta[0] = np.linalg.norm(self.theta0 - theta_true).item()

    def errors_update(self, i, y_true, theta_true, T, n_pred):
        Y_pred = np.array([self._y_pred[k] for k in range(1, T + n_pred + 1)])
        Y_true = np.array([y_true[k] for k in range(1, T + n_pred + 1)])
        self._E_y[i] = np.linalg.norm(Y_pred - Y_true).item()
        self._E_theta[i] = np.linalg.norm(self._theta[i] - theta_true).item()

    def figures_init(self, live_plot=False):
        if not live_plot:
            matplotlib.use("Agg")
        plt.ion()
        self._fig_fit = plt.figure()
        self._fig_subspace = plt.figure()
        self._fig_error = plt.figure()

    def figures_close(self):
        plt.close(self._fig_fit)
        plt.close(self._fig_subspace)
        plt.close(self._fig_error)

    def figures_update(self, x_true, y_obs, T, n_pred, live_plot=False):
        if not live_plot:
            return
        self.figure_fit_update(y_obs, n_pred, T)
        self.figure_subspace_update(x_true, T)
        self.figure_error_update()

    def figure_fit_update(self, y_obs, n_pred, T):
        self._fig_fit.clf()

        Y = np.asarray([y_obs[t] for t in range(1, T + n_pred + 1)])
        Y = Y.squeeze().T
        Y_pred = np.asarray(
            [self._y_pred[t] for t in range(1, T + n_pred + 1)]
        )
        Y_pred = Y_pred.squeeze().T

        # Colors come from the "high-contrast" color scheme of
        # https://personal.sron.nl/~pault/data/colourschemes.pdf
        for l in range(self._d):
            ax = self._fig_fit.add_subplot(4, 5, l + 1)
            ax.plot(Y[l, :T], color="#004488", alpha=0.5)
            ax.plot(
                range(T, T + n_pred), Y[l, T:], color="#ddaa33", alpha=0.5,
            )
            ax.plot(Y_pred[l, :], color="#bb5566", linestyle="dashed")
            ax.axis("off")

        plt.pause(0.01)

    def figure_subspace_update(self, x_true, T):
        self._fig_subspace.clf()

        X_true = np.array([x_true[k] for k in range(0, T + 1)])
        X_true = X_true.squeeze().T

        X_pred = np.array([self._mu[k] for k in range(0, T + 1)])
        X_pred = X_pred.squeeze().T

        used = set()
        for l in range(self._r):
            ax = self._fig_subspace.add_subplot(
                2, int((self._r + 1) / 2), l + 1
            )
            pl = np.argmin(
                [
                    np.linalg.norm(X_true[l] - X_pred[m])
                    if not m in used
                    else np.inf
                    for m in range(self._r)
                ]
            )
            ax.plot(np.squeeze(X_true[l, :]), color="#004488", alpha=0.7)
            ax.plot(np.squeeze(X_pred[pl, :]), "--", color="#bb5566")
            ax.axis("off")
            used.add(pl)

        plt.pause(0.01)

    def figure_error_update(self):
        idx = sorted(self._E_y.keys())
        E_y = np.array([self._E_y[i] for i in idx])
        E_theta = np.array([self._E_theta[i] for i in idx])

        err = [E_y, E_theta]
        titles = ["||y* - Cx||^2", "||θ* - θ||^2"]

        self._fig_error.clf()
        for l in range(len(err)):
            ax = self._fig_error.add_subplot(len(err), 1, l + 1)
            ax.plot(idx, err[l])
            ax.title.set_text(titles[l])

        plt.pause(0.01)

    def figure_save(
        self, name, filename, y_obs=None, x_true=None, n_pred=None, T=None
    ):
        assert name in ["fit", "bases", "cost"]

        if name == "cost":
            idx = sorted(self._E_y.keys())
            E_y = np.array([self._E_y[i] for i in idx])
            plt.close(self._fig_error)
            self._fig_error = plt.figure(figsize=(7.5, 2))
            plt.semilogx(E_y, color="#bb5566")
            plt.xlabel("Iterations", fontsize=14)
            plt.ylabel("Frobenius norm", fontsize=14)
            fig = self._fig_error
        elif name == "fit":
            self.figure_fit_update(y_obs, n_pred, T)
            fig = self._fig_fit
        elif name == "bases":
            self.figure_subspace_update(x_true, T)
            fig = self._fig_subspace

        fig.savefig(
            filename,
            dpi=None,
            facecolor="w",
            edgecolor="w",
            orientation="portrait",
            format="pdf",
            bbox_inches="tight",
            pad_inches=0,
            metadata=None,
        )

    def figures_save(
        self, y_obs, x_true, n_pred, T, output_fit, output_bases, output_cost
    ):
        self.figure_save("fit", output_fit, y_obs=y_obs, n_pred=n_pred, T=T)
        self.figure_save("bases", output_bases, x_true=x_true, T=T)
        self.figure_save("cost", output_cost)
