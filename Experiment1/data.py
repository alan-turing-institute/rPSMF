# -*- coding: utf-8 -*-

"""PSMF Experiment 1 - Generate data

This file is part of the PSMF codebase.
See the LICENSE file for copyright and licensing information.

"""

import autograd.numpy as np


def generate_normal_data(
    nonlinearity=None, d=20, T=500, n_pred=250, r=6, var=0.1, **kwargs
):
    C_true = np.random.randn(d, r)
    theta_true = np.asarray([[th * 1e-3] for th in range(1, r + 1)])

    x_true = {0: np.random.randn(r, 1)}

    y_true = {}
    y_obs = {}

    for t in range(1, T + n_pred + 1):
        x_true[t] = nonlinearity(theta_true, x_true[t - 1], t)
        y_true[t] = C_true @ x_true[t]
        y_obs[t] = C_true @ x_true[t] + np.sqrt(var) * np.random.randn(d, 1)

    y_train = {t: y_obs[t] for t in range(1, T + 1)}

    return dict(
        C_true=C_true,
        theta_true=theta_true,
        x_true=x_true,
        y_true=y_true,
        y_obs=y_obs,
        y_train=y_train,
    )


def generate_t_data(
    nonlinearity, d=20, T=500, n_pred=250, r=6, var=0.1, dof=3, **kwargs
):
    C_true = np.random.randn(d, r)
    theta_true = np.asarray([[th * 1e-3] for th in range(1, r + 1)])

    x_true = {0: np.random.randn(r, 1)}
    y_true = {}
    y_obs = {}

    for t in range(1, T + n_pred + 1):
        x_true[t] = nonlinearity(theta_true, x_true[t - 1], t)
        y_true[t] = C_true @ x_true[t]
        y_obs[t] = C_true @ x_true[t] + np.sqrt(var) * np.random.standard_t(
            dof, (d, 1)
        )

    y_train = {t: y_obs[t] for t in range(1, T + 1)}

    return dict(
        C_true=C_true,
        theta_true=theta_true,
        x_true=x_true,
        y_true=y_true,
        y_obs=y_obs,
        y_train=y_train,
    )
