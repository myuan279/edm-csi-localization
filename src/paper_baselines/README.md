# Paper baselines

The modules here contain model implementations, not the older paper-named
import shims. `knn.py` implements one Train-RP prototype per reference point,
Euclidean distance and k=1. `mlp.py` defaults to the paper's 312D RI input.

`ddpm_loc.py` is the 312D CSI adaptation of DDPM-Loc; `ddpm.py` supplies its
diffusion schedule and sampler. The one-step code is in
`ict/` for iCT and `meanflow/` for MeanFlow. These method-named
packages retain additional research objectives from their historical source;
the paper's frozen Test record binds iCT to the historical v2 and MeanFlow to
the historical v1 implementation. EDM-Loc is
the proposed method and is exposed separately by `../edm_loc/`.

All modules accept already prepared tensors or arrays. The paper used a
Train-only normalization and frozen RP split; feeding raw or differently
normalized CSI changes the method.
