# -*- coding: utf-8 -*-

"""PSMF Experiment 3 - rPSMF

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
def robust_PSMF(
    Y,
    C,
    X,
    d,
    n,
    r,
    M,
    Mmiss,
    V,
    Q0,
    R0,
    P,
    lambda0,
    sig,
    Iter,
    YorigInt,
    Einit,
):

    Epred = np.zeros([1, Iter + 1])
    Efull = np.zeros([1, Iter + 1])
    Epred[:, 0] = Einit
    Efull[:, 0] = Einit

    RunTime = np.zeros([1, Iter + 1])

    Yrec = np.zeros([d, n])
    YrecL = np.zeros([d, n])
    YrecH = np.zeros([d, n])

    Id = np.identity(d)

    RunTimeStart = time.time()

    for i in range(0, Iter):

        Q = Q0
        R = R0
        lmd = lambda0

        for t in range(n):

            Mk = np.diag(M[:, t])
            CM = Mk @ C

            Xp = X[:, [n - 1]] if t == 0 else X[:, [t - 1]]
            PP = P + Q

            Yrec[:, [t]] = C @ Xp

            # Assumes R is diagonal. Otherwise: MRM = Mk @ R @ Mk.T
            MRM = np.diag(np.diag(Mk) * np.diag(R))
            XVX = Xp.T @ V @ Xp
            PPCM = PP @ CM.T
            CMPPCM = CM @ PPCM

            # Replace Mk in Rbar = MRM + XVX * Mk by Id, otherwise singular
            Rbar = MRM + XVX * Id
            CPinv = compute_Sinv(CM, PP, Rbar)

            diff = Y[:, [t]] - CM @ Xp
            PPCMCPinv = PPCM @ CPinv

            X[:, [t]] = Xp + PPCMCPinv @ diff
            omega = (lmd + diff.T @ CPinv @ diff) / (lmd + d)
            P = omega * (PP - PPCMCPinv @ CM @ PP)

            eta_k = np.trace(MRM + CMPPCM) / d
            Nk = XVX + eta_k

            C = C + diff @ Xp.T @ V.T / Nk
            U = XVX * Mk + eta_k * Id
            Ui = np.diag(1.0 / np.diag(U))
            phi = (lmd + diff.T @ Ui @ diff) / (lmd + d)
            V = phi * (V - (V @ Xp @ Xp.T @ V) / Nk)

            # We approximate the boundaries of the interval using (-sig*std, 
            # +sig*std), based on the normal distribution. This can be shown to 
            # have a negligle effect on performance compared to the exact 
            # version (kept below for reference), but is significantly faster.
            sqU = np.sqrt(np.diag(U)).reshape(d, 1)
            YrecL[:, [t]] = Yrec[:, [t]] - sig * sqU
            YrecH[:, [t]] = Yrec[:, [t]] + sig * sqU
            # AREA can be precomputed
            # AREA = scipy.stats.norm().cdf(sig) - scipy.stats.norm().cdf(-sig)
            # YrecL[:, [t]], YrecH[:, [t]] = scipy.stats.t.interval(
            #     AREA,
            #     lmd,
            #     loc=Yrec[:, [t]],
            #     scale=np.sqrt(np.diag(U)).reshape(d, 1),
            # )

            Q = omega * Q
            R = omega * R
            lmd = lmd + d

        Yrec2 = C @ X

        Epred[:, i + 1] = RMSEM(Yrec, YorigInt, Mmiss)
        Efull[:, i + 1] = RMSEM(Yrec2, YorigInt, Mmiss)

        RunTime[:, i + 1] = time.time() - RunTimeStart

    InsideBars = compute_number_inside_bars(
        Mmiss, d, n, YorigInt, YrecL, YrecH
    )

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

    # Extract dimensions and set latent dimensionality
    d, T = Yorig.shape
    r = 10
    sig = 2
    Iter = 2
    rho = 10
    v = 2
    q = 0.1
    p = 1.0
    lambda0 = 1.8

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
        X = np.random.rand(r, T)

        # store hash of matrices; used to ensure they're the same between
        # scripts
        Y_hashes.append(matrix_hash(Y))
        C_hashes.append(matrix_hash(C))
        X_hashes.append(matrix_hash(X))

        YrecInit = C @ X
        Einit = RMSEM(YrecInit, YorigInt, missMask)

        [ep, ef, rt, ib] = robust_PSMF(
            Y,
            C,
            X,
            d,
            T,
            r,
            M,
            missMask,
            V,
            Q,
            R,
            P,
            lambda0,
            sig,
            Iter,
            YorigInt,
            Einit,
        )

        error_pred = ep[:, Iter].item()
        error_full = ef[:, Iter].item()
        if np.isnan(error_pred) or np.isnan(error_full):
            runtime = float('nan')
            inside_bar = float('nan')
        else:
            runtime = rt[:, Iter].item()
            inside_bar = ib

        errors_predict.append(error_pred)
        errors_full.append(error_full)
        runtimes.append(runtime)
        inside_bars.append(inside_bar)

        log(
            "[%s] Finished step %04i of %04i"
            % (dt.datetime.now().strftime("%c"), i + 1, args.repeats)
        )

    params = {
        "r": r,
        "sig": sig,
        "rho": rho,
        "q": q,
        "p": p,
        "Iter": Iter,
        "lambda0": lambda0,
        "v": v,
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
        "rPSMF",
    )
    dump_output(output, args.output)


if __name__ == "__main__":
    main()
