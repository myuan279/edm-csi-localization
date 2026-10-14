# EDM-Loc: probabilistic localization from LTE CSI

Source release for the manuscript *EDM-Loc: More Accurate Probabilistic
Localization with Fewer Denoiser Evaluations* (ICASSP 2027 submission).

## Overview

EDM-Loc estimates a distribution of 2D positions from LTE channel state
information (CSI). The proposed conditional EDM model uses a 312-value
real/imaginary CSI representation and generates 20 position samples per input
at the paper's default 63 denoiser evaluations. The included comparisons are
the eight methods in Table 1; they use the same paper-specific CSI setting.

![EDM-Loc method overview](figures/figure_1_edm_loc_overview.png)

**Release status.** The model definitions, portable EDM-Loc Train/Development
retraining and explicit evaluation entry points, and the three manuscript PNG
figures are included. Original OpenCSI arrays, frozen paper checkpoints,
calibration artifacts, full prediction arrays and figure source data are **not**
redistributed here. Running `scripts/train_edm_loc.py` creates a new run; it
does not recreate or overwrite the paper's formal Test evidence.

## Methods and repository layout

| Path | Purpose |
|---|---|
| `src/edm_loc/` | Proposed EDM-Loc, RI conversion, conditional FiLM denoiser, EDM loss and Heun sampler |
| `src/paper_baselines/` | All eight Table 1 comparison methods, including DDPM-Loc, iCT, MeanFlow, FullCov, GlobalCov, MLP, CNN and paper-matched RP-prototype KNN |
| `src/paper_baselines/ict/` | Paper iCT model, objective, training and evaluation source |
| `src/paper_baselines/meanflow/` | Paper MeanFlow model, objective, training and evaluation source |
| `scripts/train_edm_loc.py`, `scripts/evaluate_edm_loc.py` | Portable new-run EDM utilities for explicit NPZ inputs |
| `scripts/smoke_test.py` | Data-free model and interface checks |
| `figures/` | The three manuscript PNG figures |
| `docs/METHODS_AND_EVIDENCE.md` | Method-to-source map and evidence boundaries |

The proposed method is **EDM-Loc**. The eight comparison methods in Table 1
are DDPM-Loc, iCT, MeanFlow, FullCov, GlobalCov, MLP, CNN and KNN. The public
KNN uses one mean fingerprint per Train RP, squared Euclidean distance and
`k=1`. The public MLP defaults to 312 RI inputs. The iCT and MeanFlow
directories have method-based names; their historical protocol IDs remain in
the source so the original experiment records stay traceable.

| Role | Method | Implementation |
|---|---|---|
| Proposed | EDM-Loc | `src/edm_loc/` |
| Diffusion comparison | DDPM-Loc | `src/paper_baselines/ddpm_loc.py` |
| One-step comparisons | iCT, MeanFlow | `src/paper_baselines/ict/`, `src/paper_baselines/meanflow/` |
| Probabilistic regression | FullCov, GlobalCov | `src/paper_baselines/gaussian.py` |
| Point regression | MLP, CNN, KNN | `src/paper_baselines/mlp.py`, `cnn.py`, `knn.py` |

## Quick start

Use Python 3.11 or a compatible Python 3 version. Install a PyTorch build for
your CPU or CUDA environment, then:

```bash
python -m pip install -r requirements.txt
python scripts/smoke_test.py
python scripts/verify_release.py
```

The `figures/*.png` files are the manuscript art. Figure source data and the
editable design source for Figure 1 are not included.

## Input contract for new EDM runs

Prepare **separate** Train and Development `.npz` files, each with:

| Array | Shape | Meaning |
|---|---|---|
| `raw_ap` | `[N,2,39,4]` | OpenCSI amplitude and phase, in that order |
| `coordinates_cm` | `[N,2]` | Physical coordinates in centimeters |
| `rp_ids` | `[N]` | Reference-point IDs |
| `global_uids` | `[N]` | Unique snapshot IDs |

RP IDs and UIDs must be disjoint across Train and Development. The training
script fits CSI and coordinate normalization on Train only, then applies it to
Development. To compare with the paper, use its frozen 2,509/398/279/399 RP
Train/Development/Calibration/Test partition and the same selected input
subsets. Supplying another split or feature normalization defines a new
experiment.

```bash
python scripts/train_edm_loc.py \
  --train-npz data/train.npz \
  --development-npz data/development.npz \
  --output-dir outputs/edm_seed2031 \
  --seed 2031

python scripts/evaluate_edm_loc.py \
  --checkpoint outputs/edm_seed2031/best_checkpoint.pt \
  --input-npz data/evaluation.npz \
  --output-dir outputs/evaluation_seed2031
```

The training defaults match the recorded 200 epochs, batch size 1,024,
AdamW learning rate `3e-4`, weight decay `1e-4`, EMA `0.9999` and selection
every ten epochs. It selects on Development ES, then pooled Mean, matching the
recorded checkpoint key. The evaluation utility reports RP-macro Mean/P90 and unbiased ES with 20 samples
at 63 actual NFE. For three-seed paper-style reporting, run seeds 2031, 2032
and 2033 separately and average their metrics. The script does not fit a
calibrator or report prediction-region coverage; those require the separate
Calibration panel and frozen calibrator artifacts.

## Paper result provenance

The manuscript's Table 1 and Figures 2–3 are historical results, not outputs
claimed by the new portable scripts. Only final PNG figure exports are
included; the plotted Test values and paired sample data are not distributed.
See [method and evidence map](docs/METHODS_AND_EVIDENCE.md) before comparing
new results with the manuscript.

| Figure | Manuscript image |
|---|---|
| 1: model overview | [PNG](figures/figure_1_edm_loc_overview.png) |
| 2: sampling budget | [PNG](figures/figure_2_sampling_budget.png) |
| 3: calibrated regions | [PNG](figures/figure_3_calibrated_regions.png) |

## Data, checkpoints and reuse

OpenCSI must be obtained under its own terms from its dataset provider. This
release does not contain original measurements or the manuscript's trained
weights. [Source provenance](docs/SOURCE_PROVENANCE.md) records where adapted
implementation files came from in the local research workspace, and
`SOURCE_MANIFEST.tsv` records release-file hashes. No open-source license has
been selected for this package; the authors should settle code ownership and
third-party attribution before publishing it as an openly reusable repository.

## Citation

If you use this implementation, cite the associated EDM-Loc manuscript. A
final BibTeX entry should be added when the publication details are confirmed.
