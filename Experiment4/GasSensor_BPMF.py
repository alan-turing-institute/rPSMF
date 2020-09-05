# -*- coding: utf-8 -*-

"""PSMF Experiment 4 - BPMF

Implementation of Bayesian Probabilistic Matrix Factorization

This is based on Ruslan Salakhutdinov's original Matlab code, available here: 
http://www.utstat.toronto.edu/~rsalakhu/BPMF.html

In BPMF the authors solve R = U.T @ V, so we translate this as

  R -> Y
  U.T -> C
  V -> X

this means that their N "users" is our d, their M "movies" is our n, and their 
D "rank" is our r.

This file is part of the PSMF codebase.
See the LICENSE file for copyright and licensing information.

"""

import time
import numpy as np

from tqdm import trange
from scipy.stats import wishart

from LondonAir_common import (
    RMSEM,
    matrix_hash,
    parse_args,
    prepare_missing,
    prepare_output,
    dump_output,
)


def pmf(
    train_vec,
    C,
    X,
    num_p,
    num_m,
    num_feat,
    epochs=50,
    epsilon=50,
    lmd=0.01,
    momentum=0.8,
    num_batches=9,
):
    """Only to be used from BPMF! Assumes data is standardized """

    # NOTE: This is not the exact same initialization as the other algorithms,
    # but otherwise PMF doesn't converge
    w1_M1 = 0.1 * X.T
    w1_P1 = 0.1 * C
    w1_M1_inc = np.zeros_like(w1_M1)
    w1_P1_inc = np.zeros_like(w1_P1)

    mean_rating = train_vec[:, 2].mean()

    for epoch in range(epochs):
        train_vec = np.random.permutation(train_vec)

        for batch in range(num_batches):
            # print(f"epoch {epoch} batch {batch}", end="\r")

            batch_arr = np.array_split(train_vec, num_batches)[batch]
            N = batch_arr.shape[0]

            aa_p = batch_arr[:, 0].astype(int)
            aa_m = batch_arr[:, 1].astype(int)
            rating = batch_arr[:, 2]

            # Default prediction is the mean rating
            rating = rating - mean_rating

            # Compute predictions
            pred_out = np.sum(w1_M1[aa_m, :] * w1_P1[aa_p, :], axis=1)

            # Compute gradients
            IO = np.tile(2 * (pred_out - rating).reshape(N, 1), (1, num_feat))
            Ix_m = IO * w1_P1[aa_p, :] + lmd * w1_M1[aa_m, :]
            Ix_p = IO * w1_M1[aa_m, :] + lmd * w1_P1[aa_p, :]

            dw1_M1 = np.zeros((num_m, num_feat))
            dw1_P1 = np.zeros((num_p, num_feat))

            for ii in range(N):
                dw1_M1[aa_m[ii], :] = dw1_M1[aa_m[ii], :] + Ix_m[ii, :]
                dw1_P1[aa_p[ii], :] = dw1_P1[aa_p[ii], :] + Ix_p[ii, :]

            # Update movie and user features
            w1_M1_inc = momentum * w1_M1_inc + epsilon * dw1_M1 / N
            w1_M1 = w1_M1 - w1_M1_inc

            w1_P1_inc = momentum * w1_P1_inc + epsilon * dw1_P1 / N
            w1_P1 = w1_P1 - w1_P1_inc

    return w1_M1, w1_P1


def pred(w1_M1_sample, w1_P1_sample, probe_vec, mean_rating):
    # Make predictions on the test data
    aa_p = probe_vec[:, 0].astype(int)
    aa_m = probe_vec[:, 1].astype(int)

    pred_out = (
        np.sum(w1_M1_sample[aa_m, :] * w1_P1_sample[aa_p, :], axis=1)
        + mean_rating
    )
    return pred_out


def bpmf(
    Y,
    C,
    X,
    num_p,
    num_m,
    num_feat,
    M,
    Mmiss,
    YorigInt,
    Einit,
    epochs=50,
    beta=2,
):

    assert (num_p, num_m) == Y.shape

    Epred = np.zeros([1, epochs + 1])
    Efull = np.copy(Epred)
    Epred[:, 0] = Einit
    Efull[:, 0] = Einit
    RunTime = np.zeros([1, epochs + 1])

    # Initialize hierarchical priors
    mu_u = np.zeros((num_feat, 1))
    mu_m = np.zeros((num_feat, 1))
    alpha_u = np.eye(num_feat)
    alpha_m = np.eye(num_feat)

    # parameters of Inv-Wishart distribution
    WI_u = np.eye(num_feat)
    b0_u = 2
    df_u = num_feat
    mu0_u = np.zeros((num_feat, 1))

    WI_m = np.eye(num_feat)
    b0_m = 2
    df_m = num_feat
    mu0_m = np.zeros((num_feat, 1))

    triplets_tr = []  # {dimension_idx, time_idx, measurement}
    triplets_pr = []
    for i in range(num_p):
        for j in range(num_m):
            if M[i, j] == 1:  # M = 1 means present
                triplets_tr.append((i, j, Y[i, j]))
            if 0.95 < Mmiss[i, j] < 1.05:
                triplets_pr.append((i, j, YorigInt[i, j]))

    train_vec = np.array(triplets_tr)
    probe_vec = np.array(triplets_pr)

    # NOTE we standardize and center the measurements, otherwise the algorithm
    # can diverge. We undo this during prediction, of course.
    old_mean = np.mean(train_vec[:, 2])
    old_std = np.mean(train_vec[:, 2])
    train_vec[:, 2] = (train_vec[:, 2] - old_mean) / old_std
    probe_vec[:, 2] = (probe_vec[:, 2] - old_mean) / old_std

    RunTimeStart = time.time()

    mean_rating = train_vec[:, 2].mean()

    count = np.zeros((num_p, num_m))
    aa_p = train_vec[:, 0].astype(int)
    aa_m = train_vec[:, 1].astype(int)
    for ii in range(train_vec.shape[0]):
        count[aa_p[ii], aa_m[ii]] = train_vec[ii, 2]

    # Use the PMF solution as initial. We use the same number of epochs as
    # given to this function.
    w1_M1_sample, w1_P1_sample = pmf(
        train_vec, C, X, num_p, num_m, num_feat, epochs=epochs
    )

    # Do simple fit
    mu_u = np.mean(w1_P1_sample).T
    alpha_u = np.linalg.inv(np.cov(w1_P1_sample))

    mu_m = np.mean(w1_M1_sample).T
    alpha_m = np.linalg.inv(np.cov(w1_P1_sample))

    count = count.T
    probe_rat_all = pred(w1_M1_sample, w1_P1_sample, probe_vec, mean_rating)
    counter_prob = 1

    for epoch in range(epochs):
        # Sample from movie hyperparams
        N = w1_M1_sample.shape[0]

        x_bar = np.mean(w1_M1_sample, axis=0).reshape(num_feat, 1)
        S_bar = np.cov(w1_M1_sample, rowvar=False)

        WI_post = np.linalg.inv(
            np.linalg.inv(WI_m)
            + N / 1 * S_bar
            + N * b0_m * (mu0_m - x_bar) @ (mu0_m - x_bar).T / (1 * (b0_m + N))
        )
        WI_post = (WI_post + WI_post.T) / 2
        df_mpost = df_m + N
        alpha_m = wishart.rvs(df_mpost, WI_post)
        mu_temp = (b0_m * mu0_m + N * x_bar) / (b0_m + N)
        lam = np.linalg.cholesky(np.linalg.inv((b0_m + N) * alpha_m))
        lam = lam.T
        mu_m = lam @ np.random.randn(num_feat, 1) + mu_temp

        # Sample from user hyperparams
        N = w1_P1_sample.shape[0]
        x_bar = np.mean(w1_P1_sample, axis=0).reshape(num_feat, 1)
        S_bar = np.cov(w1_P1_sample, rowvar=False)

        WI_post = np.linalg.inv(
            np.linalg.inv(WI_u)
            + N / 1 * S_bar
            + N * b0_u * (mu0_u - x_bar) @ (mu0_u - x_bar).T / (1 * (b0_u + N))
        )
        WI_post = (WI_post + WI_post.T) / 2
        df_upost = df_u + N
        alpha_u = wishart.rvs(df_upost, WI_post)
        mu_temp = (b0_u * mu0_u + N * x_bar) / (b0_u + N)
        lam = np.linalg.cholesky(np.linalg.inv((b0_u + N) * alpha_u))
        lam = lam.T
        mu_u = lam @ np.random.randn(num_feat, 1) + mu_temp

        for gibbs in range(2):
            count = count.T

            # Infer posterior distribution over all movie feature vectors
            for mm in range(num_m):
                ff = M[:, mm] > 0  # select those that are present
                MM = w1_P1_sample[ff, :]
                rr = count[ff, mm] - mean_rating
                rr = np.expand_dims(rr, 1)  # make it column vec.

                covar = np.linalg.inv(alpha_m + beta * MM.T @ MM)
                mean_m = covar @ (beta * MM.T @ rr + alpha_m @ mu_m)

                lam = np.linalg.cholesky(covar).T

                w1_M1_sample[mm, :] = (
                    lam @ np.random.randn(num_feat, 1) + mean_m
                ).squeeze()

            # Infer posterior distribution over all user feature vectors
            count = count.T
            for uu in range(num_p):
                ff = M[uu, :] > 0
                MM = w1_M1_sample[ff, :]
                rr = count[ff, uu] - mean_rating
                rr = np.expand_dims(rr, 1)  # make it column vec

                covar = np.linalg.inv(alpha_u + beta * MM.T @ MM)
                mean_u = covar @ (beta * MM.T @ rr + alpha_u @ mu_u)

                lam = np.linalg.cholesky(covar).T

                w1_P1_sample[uu, :] = (
                    lam @ np.random.randn(num_feat, 1) + mean_u
                ).squeeze()

        probe_rat = pred(w1_M1_sample, w1_P1_sample, probe_vec, mean_rating)
        probe_rat_all = (counter_prob * probe_rat_all + probe_rat) / (
            counter_prob + 1
        )
        counter_prob += 1

        # Reconstruct Yrec2. Note that the original BPMF code uses
        # probe_rat_all as the predictor, so we do the same
        pred_out = probe_rat_all * old_std + old_mean
        Yrec2 = np.zeros_like(Y)
        aa_p = probe_vec[:, 0].astype(int)
        aa_m = probe_vec[:, 1].astype(int)
        NN = probe_vec.shape[0]

        for ii in range(NN):
            Yrec2[aa_p[ii], aa_m[ii]] = pred_out[ii]

        Epred[:, epoch + 1] = np.nan  # BPMF doesn't do online predictions
        Efull[:, epoch + 1] = RMSEM(Yrec2, YorigInt, Mmiss)
        RunTime[:, epoch + 1] = time.time() - RunTimeStart

    return Epred, Efull, RunTime


def main():
    args = parse_args()

    # Set the seed if given, otherwise draw one and print it out
    seed = args.seed or np.random.randint(10000)
    print("Using seed: %r" % seed)
    np.random.seed(seed)

    # Load the data
    Yorig = np.genfromtxt(args.input, delimiter=",")

    # Create a copy with missings set to zero
    YorigInt = np.copy(Yorig)
    YorigInt[np.isnan(YorigInt)] = 0

    _range = trange if args.fancy else range
    log = print if not args.fancy else lambda *a, **kw: None

    # Initialize arrays to keep track of quantaties of interest
    errors_predict = []
    errors_full = []
    runtimes = []
    Y_hashes = []
    C_hashes = []
    X_hashes = []

    # Extract dimensions and set latent dimensionality
    d, n = Yorig.shape
    r = 10
    Iter = 2

    for i in _range(args.repeats):
        # Create the missing mask (missMask) and its inverse (M)
        Ymiss = np.copy(Yorig)
        missRatio, missMask = prepare_missing(Ymiss, args.percentage / 100)
        M = np.array(np.invert(np.isnan(Ymiss)), dtype=int)

        # In the data we work with, set missing to 0
        Y = np.copy(Ymiss)
        Y[np.isnan(Y)] = 0

        log("[%04i/%04i]" % (i + 1, args.repeats))
        C = np.random.rand(d, r)
        X = np.random.rand(r, n)

        # store hash of matrices; used to ensure they're the same between
        # scripts
        Y_hashes.append(matrix_hash(Y))
        C_hashes.append(matrix_hash(C))
        X_hashes.append(matrix_hash(X))

        YrecInit = C @ X
        Einit = RMSEM(YrecInit, YorigInt, missMask)

        # Since BPMF uses random numbers internally, we wouldn't get the same
        # C,X as the other methods without saving/resetting state.
        rand_state = np.random.get_state()

        [ep, ef, rt] = bpmf(
            Y, C, X, d, n, r, M, missMask, YorigInt, Einit, epochs=Iter
        )

        np.random.set_state(rand_state)

        errors_predict.append(ep[:, Iter].item())
        errors_full.append(ef[:, Iter].item())
        runtimes.append(rt[:, Iter].item())

    params = {"Iter": Iter, "r": r}
    hashes = {"Y": Y_hashes, "C": C_hashes, "X": X_hashes}
    results = {
        "error_predict": errors_predict,
        "error_full": errors_full,
        "runtime": runtimes,
        "inside_sig": None,  # Not returned in the model
    }
    output = prepare_output(
        args.input,
        __file__,
        params,
        hashes,
        results,
        seed,
        args.percentage,
        missRatio,
        "BPMF",
    )
    dump_output(output, args.output)


if __name__ == "__main__":
    main()
