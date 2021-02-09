# -*- coding: utf-8 -*-

"""PSMF Experiment 3 - PSMF

This file is part of the PSMF codebase.
See the LICENSE file for copyright and licensing information.

"""

import numpy as np
import time
import datetime as dt

from tqdm import trange
from joblib import Memory

from common import (
    RMSEM,
    compute_number_inside_bars,
    dump_output,
    matrix_hash,
    parse_args,
    prepare_missing,
    prepare_output,
)

MEMORY = Memory("./cache", verbose=0)


def compute_Sinv(CM, PP, Rbar):
    # woodbury, using inv, assuming Rbar diagonal
    Ri = np.diag(1 / np.diag(Rbar))
    RiC = Ri @ CM
    Pi = np.linalg.inv(PP)
    PiCRiC = Pi + CM.T @ RiC
    return Ri - RiC @ np.linalg.inv(PiCRiC) @ RiC.T


@MEMORY.cache
def ProbabilisticSequentialMatrixFactorizer(
    Y, C, X, d, n, r, M, Mmiss, lam, V, Q, R, P, sig, Iter, YorgInt, Einit
):

    Epred = np.zeros([1, Iter + 1])
    Efull = np.copy(Epred)
    Epred[:, 0] = Einit
    Efull[:, 0] = Einit

    RunTime = np.zeros([1, Iter + 1])

    Yrec = np.zeros([d, n])
    YrecL = np.copy(Yrec)
    YrecH = np.copy(Yrec)

    Id = np.eye(d)

    RunTimeStart = time.time()

    for i in range(0, Iter):
        for t in range(n):

            Mk = np.diag(M[:, t])
            CM = Mk @ C

            Xp = X[:, [n - 1]] if t == 0 else X[:, [t - 1]]
            PP = P + Q

            Yrec[:, [t]] = C @ Xp

            # Assumes R is diagonal. Otherwise: MRM = Mk @ R @ Mk.T
            MRM = np.diag(np.diag(Mk) * np.diag(R))
            Rbar = MRM + Xp.T @ V @ Xp * Id
            CPinv = compute_Sinv(CM, PP, Rbar)
            X[:, [t]] = Xp + PP @ CM.T @ CPinv @ (Y[:, [t]] - CM @ Xp)
            P = PP - PP @ CM.T @ CPinv @ CM @ PP

            eta_k = np.trace(CM @ PP @ CM.T + MRM) / d
            Nt = Xp.T @ V @ Xp + eta_k

            C = C + ((Y[:, [t]] - CM @ Xp) @ Xp.T @ V) / (Nt)
            V = V - (V @ Xp @ Xp.T @ V) / (Nt)

            YrecL[:, [t]] = Yrec[:, [t]] - sig * np.sqrt(Nt)
            YrecH[:, [t]] = Yrec[:, [t]] + sig * np.sqrt(Nt)

        Yrec2 = C @ X

        Epred[:, i + 1] = RMSEM(Yrec, YorgInt, Mmiss)
        Efull[:, i + 1] = RMSEM(Yrec2, YorgInt, Mmiss)

        RunTime[:, i + 1] = time.time() - RunTimeStart

    InsideBars = compute_number_inside_bars(Mmiss, d, n, YorgInt, YrecL, YrecH)

    return Epred, Efull, RunTime, InsideBars


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

    # Initialize dimensions, hyperparameters, and noise covariances
    d, n = Yorig.shape
    r = 10
    sig = 2
    Iter = 2
    rho = 10
    v = 2
    q = 0.1
    p = 1.0

    V = v * np.eye(r)
    Q = q * np.eye(r)
    R = rho * np.eye(d)
    P = p * np.eye(r)

    # Initialize arrays to keep track of quantaties of interest
    errors_predict = []
    errors_full = []
    runtimes = []
    inside_bars = []
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

        C = np.random.rand(d, r)
        X = np.random.rand(r, n)

        # store hash of matrices; used to ensure they're the same between
        # scripts
        Y_hashes.append(matrix_hash(Y))
        C_hashes.append(matrix_hash(C))
        X_hashes.append(matrix_hash(X))

        YrecInit = C @ X
        Einit = RMSEM(YrecInit, YorigInt, missMask)

        [ep, ef, rt, ib] = ProbabilisticSequentialMatrixFactorizer(
            Y,
            C,
            X,
            d,
            n,
            r,
            M,
            missMask,
            rho,
            V,
            Q,
            R,
            P,
            sig,
            Iter,
            YorigInt,
            Einit,
        )

        errors_predict.append(ep[:, Iter].item())
        errors_full.append(ef[:, Iter].item())
        runtimes.append(rt[:, Iter].item())
        inside_bars.append(ib)

        log(
            "[%s] Finished step %04i of %04i"
            % (dt.datetime.now().strftime("%c"), i + 1, args.repeats)
        )

    params = {
        "r": r,
        "sig": sig,
        "rho": rho,
        "v": v,
        "q": q,
        "p": p,
        "Iter": Iter,
    }
    hashes = {"Y": Y_hashes, "C": C_hashes, "X": X_hashes}
    results = {
        "error_predict": errors_predict,
        "error_full": errors_full,
        "runtime": runtimes,
        "inside_sig": inside_bars,
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
        "PSMF",
    )
    dump_output(output, args.output)


if __name__ == "__main__":
    main()
