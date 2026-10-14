"""Small, data-free checks of the public model definitions and CSI contract."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from edm_loc.contracts import DatasetBatch
from edm_loc.conditional_diffusion import build_registered_denoiser, registered_edm_config
from edm_loc.csi_representation import transform_csi
from paper_baselines import (
    DDPMLocDenoiseNet,
    FullCovarianceGaussianMLP,
    HomoscedasticGaussianMLP,
    MLPRegressor,
    OpenCSICNNRegressor,
    TrainRPPrototypeKNN,
)
from paper_baselines.ict.models import ConsistencyModel
from paper_baselines.meanflow.models import MeanFlowModel


def main() -> None:
    torch.set_num_threads(1)
    raw = np.zeros((2, 2, 39, 4), dtype=np.float32)
    raw[:, 0] = 1.0
    ri = transform_csi(raw, "R1_FREQ_RI")
    assert ri.shape == (2, 2, 39, 4)
    assert np.allclose(ri[:, 0], 1.0) and np.allclose(ri[:, 1], 0.0)
    flat = torch.from_numpy(ri.reshape(2, 312))
    batch = DatasetBatch(
        csi=torch.from_numpy(ri),
        coordinates_cm=torch.zeros(2, 2),
        rp_ids=torch.tensor([1, 2]),
        global_uids=torch.tensor([10, 11]),
        subcarrier_mask=torch.ones(2, 39, dtype=torch.bool),
        port_mask=torch.ones(2, 4, dtype=torch.bool),
        dataset_id="synthetic-smoke",
    )
    batch.validate()
    edm = build_registered_denoiser(2031).eval()
    with torch.no_grad():
        edm_out = edm(torch.zeros(2, 2), torch.zeros(2), batch)
        ddpm_out = DDPMLocDenoiseNet().eval()(torch.zeros(2, 2), torch.zeros(2, dtype=torch.long), flat)
        mlp_out = MLPRegressor().eval()(flat)
        cnn_out = OpenCSICNNRegressor().eval()(flat)
        full_out = FullCovarianceGaussianMLP().eval()(flat)
        global_out = HomoscedasticGaussianMLP().eval()(flat)
        ict_out = ConsistencyModel().eval()(torch.zeros(2, 2), torch.ones(2), flat)
        meanflow_out = MeanFlowModel().eval()(torch.zeros(2, 2), torch.zeros(2), torch.ones(2), flat)
    for name, value in {
        "EDM-Loc": edm_out, "DDPM-Loc": ddpm_out, "MLP": mlp_out,
        "CNN": cnn_out, "FullCov": full_out, "GlobalCov": global_out,
        "iCT": ict_out, "MeanFlow": meanflow_out,
    }.items():
        assert value.shape[0] == 2 and torch.isfinite(value).all(), name
    assert MLPRegressor().net[0].in_features == 312
    assert registered_edm_config().expected_heun_nfe == 63
    x = np.zeros((4, 312), dtype=np.float32)
    x[2:, 0] = 10.0
    y = np.array([[0, 0], [2, 0], [10, 0], [12, 0]], dtype=np.float32)
    knn = TrainRPPrototypeKNN().fit(x, y, np.array([1, 1, 2, 2]))
    assert np.allclose(knn.predict(x[[0, 3]]), [[1, 0], [11, 0]])
    print("PASS: nine paper methods, RI shape, 312D MLP, RP-prototype KNN, 63-NFE EDM config")


if __name__ == "__main__":
    main()
