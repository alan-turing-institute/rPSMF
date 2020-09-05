#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""PSMF Experiment 4 - Generate imputation table

This file is part of the PSMF codebase.
See the LICENSE file for copyright and licensing information.

"""

import json
import numpy as np
import os
import sys

from LondonAir_table_common import (
    METHODS,
    POLLUTANTS,
    PERCENTAGES,
    parse_args,
    make_filepath,
)


def get_mean_error(filename, method):
    with open(filename, "r") as fp:
        data = json.load(fp)

    assert data["method"] == method
    return np.mean(data["results"]["error_full"])


def get_cell_value(method, pol, perc, result_dir):
    path = make_filepath(method, pol, perc, result_dir)
    if not os.path.exists(path):
        print("Warning: result file '%s' not found." % path, file=sys.stderr)
        return "TODO"

    avg_err = get_mean_error(path, method)

    others = {}
    for other_method in METHODS:
        if other_method == method:
            continue

        other_path = make_filepath(other_method, pol, perc, result_dir)
        if not os.path.exists(other_path):
            continue

        others[other_method] = get_mean_error(other_path, other_method)

    min_err = min(list(others.values()) + [avg_err])
    if avg_err == min_err:
        return "\\textbf{%.2f}" % avg_err
    return "%.2f" % avg_err


def get_cell_std(method, pol, perc, result_dir):
    path = make_filepath(method, pol, perc, result_dir)
    if not os.path.exists(path):
        print("Warning: result file '%s' not found." % path, file=sys.stderr)
        return "(TODO)"

    with open(path, "r") as fp:
        data = json.load(fp)
    assert data["method"] == method
    assert data["missing_percentage"] == perc
    return "(%.2f)" % np.std(data["results"]["error_full"])


def get_run_time(method, pol, result_dir):
    path = make_filepath(method, pol, 30, result_dir)
    if not os.path.exists(path):
        print("Warning: result file '%s' not found." % path, file=sys.stderr)
        return "TODO"

    with open(path, "r") as fp:
        data = json.load(fp)

    assert data["method"] == method
    return "%.2f" % np.mean(data["results"]["runtime"])


def build_tex_table(result_dir):
    tex = []
    tex.append("%% DO NOT EDIT - AUTOMATICALLY GENERATED FROM RESULTS!")
    tex.append("%% This table requires booktabs and multirow!")

    mr2 = lambda s: "\\multirow{2}{*}{%s}" % s
    names = {"MLESMF": "MLE-SMF", "PMF": "PMF*", "BPMF": "BPMF*"}
    methname = lambda m: names.get(m, m)
    polname = lambda p: "NO$_2$" if p == "NO2" else p
    mc3 = lambda s: "\\multicolumn{3}{c}{%s}" % s

    tex.append("\\begin{tabular}{lcccccccccc}")
    tex.append("\\toprule")
    superheader = (
        " & ".join(
            [""]
            + [mc3(polname(pol)) for pol in POLLUTANTS]
            + [
                "\\multirow{3}{*}{\shortstack[l]{CPU time (sec)\\\\ on NO$_2$ dataset}} \\\\",
            ]
        )
        + " \\cmidrule(lr){2-4} \\cmidrule(lr){5-7} \\cmidrule(lr){8-10}"
    )
    tex.append(superheader)
    subheader = (
        " & ".join([""] + ["%i\%%" % i for i in PERCENTAGES] * 3)
        + " \\\\ "
        + "\\cmidrule(lr){2-4} \\cmidrule(lr){5-7} \\cmidrule(lr){8-10}"
    )
    tex.append(subheader)

    for method in METHODS:
        vrow = [mr2(methname(method))]
        srow = [""]
        for pol in POLLUTANTS:
            for perc in PERCENTAGES:
                vrow.append(get_cell_value(method, pol, perc, result_dir))
                srow.append(get_cell_std(method, pol, perc, result_dir))
        vrow.append(mr2(get_run_time(method, "20160930_203718", result_dir)))
        srow.append("")

        tex.append(" & ".join(vrow) + "\\\\")
        tex.append(" & ".join(srow) + "\\\\")

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
