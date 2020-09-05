# -*- coding: utf-8 -*-

"""PSMF Experiment 4 - Shared code for the table generation scripts

This file is part of the PSMF codebase.
See the LICENSE file for copyright and licensing information.

"""

import argparse
import os

METHODS = ["PSMF", "rPSMF", "MLESMF", "TMF", "PMF", "BPMF"]
PROB_METHODS = ["PSMF", "rPSMF", "MLESMF"]
POLLUTANTS = ["20160930_203718"]
PERCENTAGES = [30]


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
    return parser.parse_args()


def make_filepath(method, pol, perc, result_dir):
    filename = "%s_%i_%s.json" % (pol, perc, method)
    return os.path.join(result_dir, filename)
