# Publishing handoff

Use the **contents of this `csi-edm/` directory** as the GitHub repository root.
The earlier research archive is retained separately at
`../archives/csi-edm-pre-replacement-20261015/`; it contains historical
protocols, duplicate versions and large evidence bundles and is outside this
release tree.

Before making the repository public:

1. Confirm code ownership with all contributors and choose a license. Add its
   full text as `LICENSE`; until then the package has no grant of reuse rights.
2. Review whether any copied DDPM-Loc or other third-party implementation
   needs a copyright notice or upstream license. Record the source commit and
   modifications in `NOTICE` if applicable.
3. Verify that the released source and manuscript PNG figures are permitted
   for public sharing under the dataset and collaboration terms.
4. Run `python scripts/smoke_test.py`, then run
   `python scripts/verify_release.py` in a clean checkout. No protected data
   or checkpoint should be present in that checkout.
5. Review the Git staged file list before pushing. `data/`, `outputs/`, model
   checkpoints and local caches are ignored by `.gitignore`.

The published repository may truthfully call itself a **source release**.
It cannot claim an exact end-to-end reproduction until the frozen split,
checkpoints, calibrators, resolved one-step configurations, environment and
complete run commands are shareable and verified in a clean environment.
