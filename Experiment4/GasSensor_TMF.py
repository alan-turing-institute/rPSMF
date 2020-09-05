# -*- coding: utf-8 -*-

"""PSMF Experiment 4 - TMF

This file is part of the PSMF codebase.
See the LICENSE file for copyright and licensing information.

"""

import numpy as np
import time

from tqdm import trange

from LondonAir_common import (
    RMSEM,
    dump_output,
    matrix_hash,
    parse_args,
    prepare_missing,
    prepare_output,
)


def temporalRegularizedMF(
    Y, C, X, d, n, r, M, Mmiss, lam, R, Iter, YorgInt, Einit
):

    Epred = np.zeros([1, Iter + 1])
    Efull = np.copy(Epred)
    Epred[:, 0] = Einit
    Efull[:, 0] = Einit

    RunTime = np.zeros([1, Iter + 1])

    Yrec = np.zeros([d, n])

    A = np.identity(r)

    RunTimeStart = time.time()

    for i in range(0, Iter):

        X0 = X[:, [n - 1]]

        gam1 = 1e-6
        nu = 2
        gam = gam1 / ((i + 1) ** 0.7)

        t = 0

        MC = np.diag(
            M[:, t]
        )  # You need to write t, but not [t], as np.diag should take a one-dimensional array.
        CM = MC @ C

        Xp = A @ X0

        Yrec[:, [t]] = C @ Xp

        CPinv = np.linalg.inv(CM.T @ CM + nu * np.identity(r))
        X[:, [t]] = CPinv @ (nu * Xp + CM.T @ Y[:, [t]])

        C = C + gam * (MC.T @ (Y[:, [t]] - CM @ Xp) @ Xp.T)

        for t in range(1, n):

            MC = np.diag(
                M[:, t]
            )  # You need to write t, but not [t], np.diag should take a one-dimensional array.
            CM = MC @ C

            Xp = A @ X[:, [t - 1]]

            Yrec[:, [t]] = C @ Xp

            CPinv = np.linalg.inv(CM.T @ CM + nu * np.identity(r))
            X[:, [t]] = CPinv @ (nu * Xp + CM.T @ Y[:, [t]])

            C = C + gam * (MC.T @ (Y[:, [t]] - CM @ Xp) @ Xp.T)

        Yrec2 = C @ X
        Epred[:, i + 1] = RMSEM(Yrec, YorgInt, Mmiss)
        Efull[:, i + 1] = RMSEM(Yrec2, YorgInt, Mmiss)

        RunTime[:, i + 1] = time.time() - RunTimeStart

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

    # Extract dimensions and set latent dimensionality
    d, n = Yorig.shape
    r = 10
    Iter = 2
    rho = 10
    R = rho * np.eye(d)

    # Initialize arrays to keep track of quantaties of interest
    errors_predict = []
    errors_full = []
    runtimes = []
    Y_hashes = []
    C_hashes = []
    X_hashes = []

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

        [ep, ef, rt] = temporalRegularizedMF(
            Y, C, X, d, n, r, M, missMask, rho, R, Iter, YorigInt, Einit
        )

        errors_predict.append(ep[:, Iter].item())
        errors_full.append(ef[:, Iter].item())
        runtimes.append(rt[:, Iter].item())

    params = {"r": r, "rho": rho, "Iter": Iter}
    hashes = {"Y": Y_hashes, "C": C_hashes, "X": X_hashes}
    results = {
        "error_predict": errors_predict,
        "error_full": errors_full,
        "runtime": runtimes,
        "inside_sig": None,
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
        "TMF",
    )
    dump_output(output, args.output)


if __name__ == "__main__":
    main()
