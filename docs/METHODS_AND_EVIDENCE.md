# Methods and evidence map

The manuscript PDF supplied on 2026-09-29 is the scope for this release.
Table 1 evaluates nine methods on the 399-RP Test partition. All methods use
the paper's Train-only RI preprocessing; raw amplitude/phase is converted to
`[A cos(phi), A sin(phi)]` before feature normalization.

| Manuscript method | Public implementation | Paper evidence note |
|---|---|---|
| EDM-Loc | `src/edm_loc/conditional_diffusion.py`, `csi_representation.py` | Proposed method; 20 trajectories, 32 positive noise levels, 63 NFE default |
| DDPM-Loc | `src/paper_baselines/ddpm_loc.py`, `ddpm.py` | 312D CSI adaptation; Table 1 uses ancestral 1000 NFE, Fig. 2 separately uses respaced/DDIM budgets |
| iCT | `src/paper_baselines/ict/models.py`, `objectives/ict.py` | One-step model; keep its checkpoint-bound resolved config with any new run |
| MeanFlow | `src/paper_baselines/meanflow/models.py`, `objectives/meanflow.py` | One-step model; the paper's three frozen checkpoints used the historical v1 implementation |
| FullCov | `src/paper_baselines/gaussian.py` | Input-dependent Gaussian mean and covariance |
| GlobalCov | `src/paper_baselines/gaussian.py` | Input-dependent mean and shared covariance |
| MLP | `src/paper_baselines/mlp.py` | 312D RI input for this paper; model class default corrected from older 8D project setting |
| CNN | `src/paper_baselines/cnn.py` | RI tensor reshaped to `(2,39,4)` after Train-only normalization |
| KNN | `src/paper_baselines/knn.py` | Train-RP mean prototypes, squared Euclidean distance, `k=1`; replaces unrelated snapshot-level `k=5` shim |

The older `archives/csi-edm-pre-replacement-20261015/CURRENT_PAPER_20260906`
archive describes several baseline
Test evaluations as post-hoc reuse of the held-out set. They are descriptive
comparisons, not new untouched-Test confirmation. The PDF's Figure 2 draws
from later completed formal-Test NFE sweeps. Do not substitute the ancestral
1000-NFE DDPM-Loc Table 1 point for its 50-NFE curve point. The frozen Test
record binds iCT to v2 and MeanFlow to v1. Their checkpoint identities and
resolved configurations remain in the local research archive; this source-only
release does not include the weights, so exact numerical reproduction is
unverified.

Figures 1–3 are distributed as final PNG exports only. The original Test
arrays, derived plotting tables, paired samples and calibration model outputs
are absent from this release. Figure 3's historical pairing and calibration
cannot be independently reconstructed from the PNG alone.

## Boundaries of the portable scripts

`scripts/train_edm_loc.py` trains a new model and writes a new checkpoint. It
does not execute the historical formal runner, validate its original data
access ledger, or guarantee Table 1 numbers. `scripts/evaluate_edm_loc.py` only
evaluates an explicitly supplied NPZ with a public-format checkpoint. The
manuscript's calibration, figures and table values remain tied to their
frozen original experiments and cannot be inferred from a successful smoke
check or from the exported images.
