# Experiment 2 - Change Point Detection

This directory contains the code to reproduce the experiments on change point 
detection in the [PSMF paper](https://arxiv.org/abs/1910.03906).

To reproduce, open Matlab and run:

```matlab
> main
```

This experiment makes use of the following publically-available source code. 
These files are not licensed under the MIT license that covers the rest of the 
PSMF codebase and are not copyrighted by us. Copyright belongs to the original 
authors, we merely distribute it here for completeness.

*From Ilaria Lauzana's [changepoint detection 
codebase](https://github.com/epfl-lasa/changepoint-detection)*:

 - ``MBOCPD.m``
 - ``MVBOCPD.m``
 - ``constant_hazard.m``
 - ``studentpdf_multi.m``

*From [Arno Solin](https://users.aalto.fi/~asolin/) and the [IHGP 
codebase](https://github.com/AaltoML/IHGP)*:

 - ``cf_matern32_to_ss.m``. This file is licensed under the GPL (v3).
