# -*- coding: utf-8 -*-

"""Mixin for logging and plotting in some of the experiments

"""

import autograd.numpy as np
import matplotlib
import matplotlib.pyplot as plt


plt.rcParams["text.usetex"] = True
plt.rcParams["text.latex.preamble"] = "\\usepackage{{sansmath}}\\sansmath"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = "Helvetica"
plt.rcParams["figure.raise_window"] = False


class TrackingMixin:
    def log(self, i, n_iter, delta_t, verbose=True, prefix=""):
        c = {
            "iter": "[%s%03i/%i]" % (prefix, i, n_iter),
            "full": "||y - Cx||^2 = %.5f" % self._E_y[i],
            "train": "||y - Cx||^2 (train) = %.5f" % self._E_train[i],
            "pred": "||y - Cx||^2 (pred) = %.5f" % self._E_pred[i],
            "time": "Δt = %.3f" % delta_t,
        }
        if self._E_theta is None:
            l = "{iter} {full}, {train}, {pred}, {time}".format(**c)
        else:
            c["theta"] = "||θ* - θ||^2 = %.5f" % self._E_theta[i]
            l = "{iter} {full}, {train}, {pred}, {theta}, {time}".format(**c)

        if not hasattr(self, "_logs"):
            self._logs = []
        self._logs.append(l)
        if not verbose:
            return
        print(l, flush=True)

    def logs_save(self, filename):
        if not hasattr(self, "_logs"):
            return
        with open(filename, "w") as fp:
            fp.write("\n".join(self._logs))

    def errors_init(self, y, T, n_iter, n_pred, theta_true=None):
        self._E_y = {}
        self._E_train = {}
        self._E_pred = {}
        self._E_theta = None if theta_true is None else {}

        Y_pred = self.C0 @ np.zeros((self._r, T + n_pred))
        Y_real = np.array([y[k] for k in range(1, T + n_pred + 1)])
        Y_real = Y_real.squeeze().T

        self._E_y[0] = np.linalg.norm(Y_pred - Y_real).item()
        self._E_train[0] = np.linalg.norm(Y_pred[:, :T] - Y_real[:, :T]).item()
        self._E_pred[0] = np.linalg.norm(Y_pred[:, T:] - Y_real[:, T:]).item()
        if not theta_true is None:
            self._E_theta[0] = np.linalg.norm(self.theta0 - theta_true).item()

    def errors_update(self, i, y, T, n_pred, theta_true=None):
        Y_pred = np.array([self._y_pred[k] for k in range(1, T + n_pred + 1)])
        Y_true = np.array([y[k] for k in range(1, T + n_pred + 1)])

        Y_pred = Y_pred.squeeze().T
        Y_true = Y_true.squeeze().T

        self._E_y[i] = np.linalg.norm(Y_pred - Y_true).item()
        self._E_train[i] = np.linalg.norm(Y_pred[:, :T] - Y_true[:, :T]).item()
        self._E_pred[i] = np.linalg.norm(Y_pred[:, T:] - Y_true[:, T:]).item()
        if not theta_true is None:
            self._E_theta[i] = np.linalg.norm(
                self._theta[i] - theta_true
            ).item()

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

    def figures_update(self, y_obs, T, n_pred, live_plot=False, x_true=None):
        if not live_plot:
            return
        self.figure_fit_update(y_obs, n_pred, T)
        self.figure_subspace_update(T, x_true=x_true)
        self.figure_error_update()

    def figure_fit_update(self, y_obs, n_pred, T):
        self._fig_fit.clf()

        Y = np.asarray([y_obs[t] for t in range(1, T + n_pred + 1)])
        Y = Y.squeeze().T
        Y_pred = np.asarray(
            [self._y_pred[t] for t in range(1, T + n_pred + 1)]
        )
        Y_pred = Y_pred.squeeze().T

        U, V = sorted(self._plot_dims(self._d))

        # Colors come from the "high-contrast" color scheme of
        # https://personal.sron.nl/~pault/data/colourschemes.pdf
        # There'll be a little gap in the plot for Y, but that's just in the
        # plot, not in the data.
        for l in range(self._d):
            ax = self._fig_fit.add_subplot(U, V, l + 1)
            ax.plot(range(1, T + 1), Y[l, :T], color="#004488", alpha=0.5)
            ax.plot(
                range(T + 1, T + n_pred + 1),
                Y[l, T:],
                color="#ddaa33",
                alpha=0.5,
            )
            ax.plot(
                range(1, T + n_pred + 1),
                Y_pred[l, :],
                color="#bb5566",
                linestyle="dashed",
            )
            ax.axis("off")

        plt.pause(0.01)

    def figure_subspace_update(self, T, x_true=None):
        self._fig_subspace.clf()

        if x_true:
            X_true = np.array([x_true[k] for k in range(0, T + 1)])
            X_true = X_true.squeeze().T

        X_pred = np.array([self._mu[k] for k in range(0, T + 1)])
        X_pred = X_pred.squeeze().T
        X_pred = X_pred.reshape(self._r, T + 1)

        used = set()
        for l in range(self._r):
            ax = self._fig_subspace.add_subplot(
                2, int((self._r + 1) / 2), l + 1
            )
            if x_true:
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
                used.add(pl)
            else:
                ax.plot(np.squeeze(X_pred[l, :]), "--", color="#bb5566")
            ax.axis("off")

        plt.pause(0.01)

    def figure_error_update(self):
        idx = sorted(self._E_train.keys())
        E_train = np.array([self._E_train[i] for i in idx])
        E_pred = np.array([self._E_pred[i] for i in idx])
        if not self._E_theta is None:
            E_theta = np.array([self._E_theta[i] for i in idx])
            err = [E_train, E_pred, E_theta]
            titles = [
                "$||y^* - Cx||^2$ (train)",
                "$||y^* - Cx||^2$ (pred)",
                "$||\\theta^* - \\theta||^2$",
            ]
        else:
            err = [E_train, E_pred]
            titles = ["$||y^* - Cx||^2$ (train)", "$||y^* - Cx||^2$ (pred)"]

        self._fig_error.clf()
        for l in range(len(err)):
            ax = self._fig_error.add_subplot(len(err), 1, l + 1)
            ax.plot(idx, err[l])
            ax.title.set_text(titles[l])

        plt.pause(0.01)

    def figure_save(
        self,
        name,
        filename,
        y_obs=None,
        x_true=None,
        n_pred=None,
        T=None,
        fit_figsize=None,
    ):
        assert name in ["fit", "bases", "cost_y", "cost_theta"]

        if name == "cost_y":
            idx = sorted(self._E_y.keys())
            E_y = np.array([self._E_y[i] for i in idx])
            plt.close(self._fig_error)
            self._fig_error = plt.figure(figsize=(7.5, 2))
            plt.plot(E_y, color="#bb5566")
            plt.xlabel("Iterations", fontsize=14)
            plt.ylabel("Frobenius norm", fontsize=14)
            fig = self._fig_error
        elif name == "cost_theta":
            idx = sorted(self._E_theta.keys())
            E_theta = np.array([self._E_theta[i] for i in idx])
            plt.close(self._fig_error)
            self._fig_error = plt.figure(figsize=(7.5, 2))
            plt.plot(E_theta, color="#bb5566")
            plt.xlabel("Iterations", fontsize=14)
            plt.ylabel("Frobenius norm", fontsize=14)
            fig = self._fig_error
        elif name == "fit":
            self._fig_fit = plt.figure(figsize=fit_figsize)
            self.figure_fit_update(y_obs, n_pred, T)
            fig = self._fig_fit
        elif name == "bases":
            self.figure_subspace_update(T, x_true=x_true)
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
        self,
        y_obs,
        n_pred,
        T,
        x_true=None,
        output_files=None,
        fit_figsize=None,
    ):
        output_files = output_files or {}
        fname = lambda k: output_files.get(k, k + ".pdf")
        self.figure_save(
            "fit",
            fname("fit"),
            y_obs=y_obs,
            n_pred=n_pred,
            T=T,
            fit_figsize=fit_figsize,
        )
        self.figure_save("bases", fname("bases"), T=T, x_true=x_true)
        self.figure_save("cost_y", fname("cost_y"))
        if not self._E_theta is None:
            self.figure_save("cost_theta", fname("cost_theta"))

    def _plot_dims(self, n):
        """Given n plots, find a pleasing 2-D arrangement"""
        factors = []
        for i in reversed(range(1, n + 1)):
            if n % i == 0:
                factors.append(i)

        diffs = {}
        for i in range(len(factors)):
            a = factors[i]
            for j in range(i + 1, len(factors)):
                b = factors[j]
                if a * b == n:
                    diffs[(a, b)] = abs(a - b)
        return min(diffs, key=diffs.get)
