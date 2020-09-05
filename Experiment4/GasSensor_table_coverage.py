#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""PSMF Experiment 4 - Generate coverage table

This file is part of the PSMF codebase.
See the LICENSE file for copyright and licensing information.

"""

import json
import numpy as np
import os
import sys

from LondonAir_table_common import (
        PROB_METHODS,
    POLLUTANTS,
    PERCENTAGES,
    parse_args,
    make_filepath,
)


def get_mean_coverage(path, method):
    with open(path, "r") as fp:
        data = json.load(fp)
    assert data["method"] == method
    return np.mean(data["results"]["inside_sig"])


def get_coverage(method, pol, perc, result_dir):
    path = make_filepath(method, pol, perc, result_dir)
    if not os.path.exists(path):
        print("Warning: result file '%s' not found." % path, file=sys.stderr)
        return "TODO"

    cov = get_mean_coverage(path, method)

    others = {}
    for other_method in PROB_METHODS:
        if other_method == method:
            continue

        other_path = make_filepath(other_method, pol, perc, result_dir)
        if not os.path.exists(other_path):
            continue

        others[other_method] = get_mean_coverage(other_path, other_method)

    max_cov = max(list(others.values()) + [cov])
    if cov == max_cov:
        return "\\textbf{%.2f}" % cov
    return "%.2f" % cov


def build_tex_table(result_dir):
    tex = []
    tex.append("%% DO NOT EDIT - AUTOMATICALLY GENERATED FROM RESULTS!")
    tex.append("%% This table requires booktabs and multirow!")

    methname = lambda m: "MLE-SMF" if m == "MLESMF" else m
    polname = lambda p: "NO$_2$" if p == "NO2" else p
    mc3 = lambda s: "\\multicolumn{3}{c}{%s}" % s

    tex.append("\\begin{tabular}{lccccccccc}")
    tex.append("\\toprule")
    superheader = (
        " & ".join([""] + [mc3(polname(pol)) for pol in POLLUTANTS])
        + "\\\\ \n"
        + "\\cmidrule(lr){2-4} \\cmidrule(lr){5-7} \\cmidrule(lr){8-10}"
    )
    tex.append(superheader)
    subheader = (
        " & ".join([""] + ["%i\%%" % i for i in PERCENTAGES] * 3)
        + " \\\\ "
        + "\n"
        + "\\cmidrule(lr){2-4} \\cmidrule(lr){5-7} \\cmidrule(lr){8-10}"
    )
    tex.append(subheader)

    for method in PROB_METHODS:
        row = [methname(method)]
        for pol in POLLUTANTS:
            for perc in PERCENTAGES:
                row.append(get_coverage(method, pol, perc, result_dir))

        tex.append(" & ".join(row) + "\\\\")

    tex.append("\\bottomrule")
    tex.append("\\end{tabular}")

    return tex


def main():
    args = parse_args()
    tex = build_tex_table(args.input_dir)
    with open(args.output_file, "w") as fp:
        fp.write("\n".join(tex))


if __name__ == "__main__":
    main()
