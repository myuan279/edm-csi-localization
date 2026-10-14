# Source provenance

This release was curated from the local `csi-ddpm` workspace on 2026-09-29
and placed at `csi-edm/` on 2026-10-15. The historical `csi-edm` tree is
retained at `archives/csi-edm-pre-replacement-20261015/` relative to the
workspace root. The source paths below identify that historical tree.
`SOURCE_MANIFEST.tsv` records the bytes and SHA-256 of every distributed file.

| Release path | Local source and change |
|---|---|
| `src/edm_loc/`, `src/opencsi_data.py`, `src/probabilistic_rng.py`, `src/probabilistic_metrics.py` | Adapted from the historical archive's `ICASSP2027_EDM-Loc_20260929/source/methods/core/`; former `csi_ddpm_v2` core and paper-facing `edm_loc` wrapper were consolidated, modules were renamed for their role, unused historical framework modules were removed, and the package initializer was simplified. Frozen protocol IDs and RNG stream names remain unchanged. |
| `src/paper_baselines/mlp.py`, `cnn.py`, `gaussian.py`, `ddpm.py` | Copied from that bundle's `source/methods/baselines/`; MLP default input changed from 8 to the paper's 312 and the unrelated older snapshot KNN class was removed. |
| `src/paper_baselines/ddpm_loc.py` | Copied from workspace `ddpm_loc_baseline.py`; its import changed to the packaged `ddpm.py`. This is the 312D CSI adaptation, not the earlier simulated-ranging DDPM-Loc experiment. |
| `src/paper_baselines/knn.py` | New importable extraction of the Train-RP prototype and nearest-prototype computation in the historical archive's `experiment/KNN_Evaluation.py`; it is not the older `k=5` snapshot KNN class. |
| `src/paper_baselines/ict/` | Selected model, objective, data adapter, train and evaluate source from the historical archive's `ICASSP2027_EDM-Loc_20260929/source/research_baselines/ri_onestep_v2/`; used for paper iCT. The public directory was renamed; protocol identifiers remain historical. |
| `src/paper_baselines/meanflow/` | Same selected source from the corresponding historical v1 folder; the frozen Test record binds paper MeanFlow to v1. The public directory was renamed; protocol identifiers remain historical. |
| `figures/figure_*.png` | Existing manuscript image exports from the historical archive's `figures/`. |

The new public CLI scripts are release adaptations, not exact byte copies of
historical formal runners. Figure source data, the redraw utility and the
paper protocol summary were excluded from the GitHub release and retained at
`archives/csi-edm-release-excluded-20261015/` relative to the workspace root.
The historical archive remains at
`archives/csi-edm-pre-replacement-20261015/` for audit and was not rewritten
to match this release.
