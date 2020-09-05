# Probabilistic Sequential Matrix Factorization

This repository contains the code to reproduce the experiments in:

[**Akyildiz, van den Burg, Damoulas, Steel - Probabilistic Sequential Matrix 
Factorization (2019)**](https://arxiv.org/abs/1910.03906)

Work that uses the methods described in the paper or the code in this 
repository should cite the paper, for instance using the following BibTeX 
entry:

```bib
@article{akyildiz2019probabilistic,
    title={Probabilistic sequential matrix factorization},
    author={{\"O}mer Deniz Akyildiz and Gerrit J. J. {van den Burg} and Theodoros Damoulas and Mark F. J. Steel},
    year={2019},
    journal={arXiv preprint arXiv:1910.03906},
}
```

If you encounter a problem when using this repository or simply want to ask a 
question, please don't hesitate to [open an issue on 
GitHub](https://github.com/alan-turing-institute/rPSMF) or send an email to 
``odakyildiz at turing dot ac dot uk`` and/or ``gvandenburg at turing dot ac 
dot uk``.

## Introduction

Our Probabilistic Sequential Matrix Factorization (PSMF) method allows you to 
model high-dimensional timeseries data that exhibits non-stationary dynamics. 
We also propose a robust variant of the model, called rPSMF, that handles 
model misspecification and outliers.

See [the paper](https://arxiv.org/abs/1910.03906) for further details.

<p align="center">
  <img width="40%" src="./.github/rpsmf.png" alt="Illustration of fitting 
  rPSMF to a multidimensional time series that contains outliers">
  <br>
  <br>
  <span><i>Illustration of using rPSMF to model a 20-dimensional time series 
  with non-linear dynamics and t-distributed outliers. Blue lines are the 
  observed data and yellow lines are unobserved future data. The red dashed 
  line shows the predictions from our model.
  </i></span>
</p>

## Usage

The code in this repository is organized by experiment, where the experiments 
are numbered as they occur in the paper. The [Convergence](./Convergence) 
directory contains the code for the convergence experiment in the appendix of 
the paper. Experiment 4 is an additional test on real-world [gas sensor 
data](https://archive.ics.uci.edu/ml/datasets/Gas+sensor+array+temperature+modulation)
that confirms the competitive performance of the proposed method on larger 
datasets.

The accompanying Makefile can be used to reproduce Experiments 1, 3, and 4, by 
simply running:

```bash
$ make Experiment1        # or Experiment3, or Experiment4
```

These experiments will be run through a Python virtual environment that will 
be automatically created with the required dependencies. For Experiments 3 and 
4 the results are captured in LaTeX tables that are automatically generated as 
well.

The code for Experiment 2 and the Convergence experiment are written in 
Matlab, so require a Matlab installation to reproduce. See the readme files in 
the corresponding directories for more information.

## Notes

The code is licensed under the MIT license unless otherwise noted, see the 
[LICENSE](./LICENSE) file for further details.
