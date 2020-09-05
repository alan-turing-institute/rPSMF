# -*- coding: utf-8 -*-

"""PSMF Experiment 3 - Shared functionality

This file is part of the PSMF codebase.
See the LICENSE file for copyright and licensing information.

"""

import argparse
import hashlib
import json
import numpy as np
import os
import safer
import socket


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input", help="Input CSV file with pollutants", required=True
    )
    parser.add_argument(
        "-o", "--output", help="Output JSON file", required=True
    )
    parser.add_argument(
        "-p",
        "--percentage",
        help="Missing percentage (0-100)",
        type=int,
        required=True,
    )
    parser.add_argument(
        "-s", "--seed", help="Random seed", type=int, default=42
    )
    parser.add_argument(
        "-r", "--repeats", help="Number of MC repeats", type=int, default=1000
    )
    parser.add_argument(
        "-f",
        "--fancy",
        help="Show fancy progress bar (suppress other printing)",
        action="store_true",
    )

    return parser.parse_args()


def prepare_missing(Ymiss, missRatio, misSeg=20):
    """Prepare the data by generating a random missing mask

    Ymiss : (copy of) the input data (d x n)
    missRatio : fraction in [0, 1] of the amount of missing data we want
    misSeg : length of consecutive missing data segments

    Returns

    ratio : actually achieved ratio of missingness (close to desired)
    Mmiss : missing mask (missing set to 1)

    """
    d, n = Ymiss.shape
    NumMissDefault = np.sum(np.isnan(Ymiss))
    Mmiss = np.zeros_like(Ymiss)
    ratio = NumMissDefault / (d * n)
    while ratio < missRatio:
        for i in range(0, d):
            mStart = np.random.randint(1, n - misSeg)
            for j in range(mStart, mStart + misSeg):
                if not np.isnan(Ymiss[i, j]):
                    Ymiss[i, j] = np.nan
                    Mmiss[i, j] = 1
        ratio = (NumMissDefault + np.sum(Mmiss)) / (d * n)

    return ratio, Mmiss


def RMSEM(Y1, Y2, M):
    nMissing = np.sum(M)
    Y1m = np.multiply(Y1, M)
    Y2m = np.multiply(Y2, M)
    R2 = (1 / nMissing) * np.sum(np.power(Y1m - Y2m, 2))
    return np.sqrt(R2)


def compute_number_inside_bars(Mmiss, m, n, Yorg, YrecL, YrecH):
    TotalNum = 0
    for i in range(m):
        for j in range(n):
            if Mmiss[i, j] == 1:
                if Yorg[i, j] < YrecH[i, j] and YrecL[i, j] < Yorg[i, j]:
                    TotalNum += 1
    return TotalNum / np.sum(Mmiss)


def sha1sum(filename):
    blocksize = 1 << 16
    hasher = hashlib.sha1()
    with open(filename, "rb") as fp:
        buf = fp.read(blocksize)
        while len(buf) > 0:
            hasher.update(buf)
            buf = fp.read(blocksize)
    return hasher.hexdigest()


def matrix_hash(A):
    b2b = hashlib.blake2b(digest_size=16)
    b2b.update(A.tobytes())
    return b2b.hexdigest()


def prepare_output(
    input_filename,
    script_filename,
    params,
    hashes,
    results,
    seed,
    percentage,
    missRatio,
    method,
):
    out = {}

    out["method"] = method
    out["seed"] = seed
    out["missing_percentage"] = percentage
    out["missing_ratio"] = missRatio
    out["parameters"] = params
    out['hashes'] = hashes
    out["results"] = results

    # hostname of the machine that created the result
    out["hostname"] = socket.gethostname()

    # save the full path of the script and its sha1
    out["script"] = script_filename
    out["script_sha1"] = sha1sum(script_filename)

    # save the full path of the data file and its sha1
    dataset_filename = os.path.abspath(os.path.realpath(input_filename))
    out["dataset"] = dataset_filename
    out["dataset_sha1"] = sha1sum(dataset_filename)

    return out


def dump_output(output, filename):
    with safer.open(filename, "w") as fp:
        json.dump(output, fp, indent="\t", sort_keys=True)
