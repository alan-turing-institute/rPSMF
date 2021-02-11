#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""PSMF Experiment 3 - Generate imputation table

This file is part of the PSMF codebase.
See the LICENSE file for copyright and licensing information.

"""

import argparse
import json
import numpy as np
import os
import sys

from table_common import (
    METHODS,
    PERCENTAGES_APP,
    PERCENTAGE_MAIN,
    DATASETS,
    DATASET_NAMES,
    PREAMBLE,
    EPILOGUE,
    make_filepath,
)

FILE_LENGTHS = set()
FILES_MISSING = set()
FILES_NAN = set()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input-dir",
        help="Input directory with result files",
        required=True,
    )
    parser.add_argument(
        "-o", "--output-file", help="Output file for the table", required=True
    )
    parser.add_argument(
        "-p",
        "--percentage",
        help="Missing percentage to ues",
        required=True,
        type=int,
        choices=[20, 30, 40],
    )
    parser.add_argument(
        "-s",
        "--standalone",
        help="Build a standalone latex document",
        action="store_true",
    )
    return parser.parse_args()


def warn_missing(path):
    if path in FILES_MISSING:
        return
    print(f"Warning: result file {path} not found.", file=sys.stderr)
    FILES_MISSING.add(path)


def warn_nan(path, count):
    if path in FILES_NAN:
        return
    print(
        f"Warning: result file {path} contains {count} NaN value(s)",
        file=sys.stderr,
    )
    FILES_NAN.add(path)


def get_mean_error(filename, method):
    with open(filename, "r") as fp:
        data = json.load(fp)

    FILE_LENGTHS.add(len(data["hashes"]["C"]))
    assert len(FILE_LENGTHS) == 1

    assert data["method"] == method
    values = data["results"]["error_full"]

    if np.any(np.isnan(values)):
        warn_nan(filename, np.sum(np.isnan(values)))
        return np.nanmean(values)
    return np.mean(values)


def get_cell_value(method, dataset, perc, result_dir):
    path = make_filepath(method, dataset, perc, result_dir)
    if not os.path.exists(path):
        warn_missing(path)
        return "TODO"

    avg_err = get_mean_error(path, method)

    others = {}
    for other_method in METHODS:
        if other_method == method:
            continue

        other_path = make_filepath(other_method, dataset, perc, result_dir)
        if not os.path.exists(other_path):
            continue

        others[other_method] = get_mean_error(other_path, other_method)

    min_err = min(list(others.values()) + [avg_err])
    if avg_err == min_err:
        return "\\textbf{%.2f}" % avg_err
    return "%.2f" % avg_err


def get_value_nohighlight(
    method, dataset, perc, result_dir, key, fmt=None, reduce="mean"
):

    fmt = (
        (lambda s: s if isinstance(s, str) else "%.2f" % s)
        if fmt is None
        else fmt
    )

    path = make_filepath(method, dataset, perc, result_dir)
    if not os.path.exists(path):
        warn_missing(path)
        return fmt("(TODO)")

    with open(path, "r") as fp:
        data = json.load(fp)

    FILE_LENGTHS.add(len(data["hashes"]["C"]))

    assert len(FILE_LENGTHS) == 1
    assert data["method"] == method
    assert data["missing_percentage"] == perc

    values = data["results"][key]

    if np.any(np.isnan(values)):
        warn_nan(path, np.sum(np.isnan(values)))
        func = {"mean": np.nanmean, "std": np.nanstd}[reduce]
    else:
        func = {"mean": np.mean, "std": np.std}[reduce]
    return fmt(func(values))


def get_cell_std(method, dataset, perc, result_dir):
    # padding is overengineering to right-align the stds
    key = "error_full"
    pad = lambda v: "" if v < 10 else (2 * "\;" if v < 100 else 3 * "\;")
    safepad = lambda s: s if isinstance(s, str) else pad(s)
    v = get_value_nohighlight(
        method,
        dataset,
        perc,
        result_dir,
        key=key,
        fmt=lambda v: v,
        reduce="mean",
    )
    numberfmt = lambda s: s if isinstance(s, str) else "%.2f" % s
    fmt = lambda s: "{\\scriptscriptstyle %s(%s)}" % (safepad(v), numberfmt(s))
    return get_value_nohighlight(
        method,
        dataset,
        perc,
        result_dir,
        key,
        fmt,
        # lambda s: "{\\scriptscriptstyle (%.2f)}" % s,
        reduce="std",
    )


def get_runtime(method, dataset, perc, result_dir):
    return get_value_nohighlight(method, dataset, perc, result_dir, "runtime")


def get_runtime_std(method, dataset, perc, result_dir):
    return get_value_nohighlight(
        method,
        dataset,
        perc,
        result_dir,
        "runtime",
        lambda s: "{\\tiny (%.2f)}" % s,
        reduce="std",
    )


def build_table(result_dir, perc):
    tex = []
    tex.append("%% DO NOT EDIT - AUTOMATICALLY GENERATED FROM RESULTS!")
    tex.append("%% This table requires booktabs, amsmath, and multirow!")

    names = {"MLESMF": "MLE-SMF", "PMF": "PMF*", "BPMF": "BPMF*"}
    methname = lambda m: names.get(m, m)
    dataset_name = lambda d: DATASET_NAMES.get(d, d)

    fmt = "l" + "r" * len(DATASETS) + "" + "r" * len(DATASETS)

    tex.append("\\begin{tabular}{%s}" % fmt)
    tex.append("\\toprule")
    superheader = (
        " & ".join(
            [""]
            + [
                "\\multicolumn{%i}{c}{Imputation RMSE}" % len(DATASETS),
                "\\multicolumn{%i}{c}{Runtime (s)}" % len(DATASETS),
            ]
        )
        + " \\\\"
        + "\\cmidrule(lr){2-6} \\cmidrule(lr){7-11}"
    )
    tex.append(superheader)
    header = (
        " & ".join([""] + [dataset_name(dset) for dset in DATASETS] * 2)
        + " \\\\"
        + " \\cmidrule(lr){2-6} \\cmidrule(lr){7-11}"
    )
    tex.append(header)

    for method in METHODS:
        vrow = [methname(method)]
        for dataset in DATASETS:
            value = get_cell_value(method, dataset, perc, result_dir)
            std = get_cell_std(method, dataset, perc, result_dir)
            vrow.append(f"$\\underset{{{std}}}{{{value}}}$")

        for dataset in DATASETS:
            vrow.append(get_runtime(method, dataset, perc, result_dir))
        tex.append(" & ".join(vrow) + "\\\\")

    tex.append("\\bottomrule")
    tex.append("\\end{tabular}")
    return tex


def main():
    args = parse_args()
    tex = build_table(args.input_dir, args.percentage)
    tex = PREAMBLE + tex + EPILOGUE if args.standalone else tex
    with open(args.output_file, "w") as fp:
        fp.write("\n".join(tex))


if __name__ == "__main__":
    main()
