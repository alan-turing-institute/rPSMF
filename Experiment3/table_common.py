# -*- coding: utf-8 -*-

"""PSMF Experiment 3 - Shared code for the table generation scripts

This file is part of the PSMF codebase.
See the LICENSE file for copyright and licensing information.

"""

import argparse
import os

METHODS = ["PSMF", "rPSMF", "MLESMF", "TMF", "PMF", "BPMF"]
PROB_METHODS = ["PSMF", "rPSMF", "MLESMF"]
DATASETS = [
    "LondonAir_NO2",
    "LondonAir_PM10",
    "LondonAir_PM25",
    "sp500_closing_prices",
    "GasSensor_20160930_203718",
]
PERCENTAGES_APP = [20, 40]
PERCENTAGE_MAIN = 30

DATASET_NAMES = {
    "LondonAir_NO2": "NO$_2$",
    "LondonAir_PM10": "PM10",
    "LondonAir_PM25": "PM25",
    "sp500_closing_prices": "S\&P500",
    "GasSensor_20160930_203718": "Gas",
}

PREAMBLE = [
    "\\documentclass[11pt, preview=true]{standalone}",
    "\\usepackage{booktabs}",
    "\\usepackage{multirow}",
    "\\usepackage{amsmath}",
    "\\begin{document}",
]
EPILOGUE = ["\\end{document}"]

def make_filepath(method, dataset, perc, result_dir):
    filename = "%s_%i_%s.json" % (dataset, perc, method)
    return os.path.join(result_dir, filename)
