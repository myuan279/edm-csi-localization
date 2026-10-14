"""Train-RP prototype KNN used for the paper's RI-CSI comparison.

Inputs must already have the paper's Train-only feature normalization.  Each
RP contributes one mean feature vector and one mean physical coordinate.
The nearest prototype uses float64 squared Euclidean distance; sorted RP IDs
give the historical ascending-RP tie rule.
"""

from __future__ import annotations

import numpy as np


class TrainRPPrototypeKNN:
    def __init__(self) -> None:
        self.rp_ids: np.ndarray | None = None
        self.prototypes: np.ndarray | None = None
        self.coordinates_cm: np.ndarray | None = None

    def fit(self, features: np.ndarray, coordinates_cm: np.ndarray, rp_ids: np.ndarray) -> "TrainRPPrototypeKNN":
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(coordinates_cm, dtype=np.float64)
        rp = np.asarray(rp_ids, dtype=np.int64)
        if x.ndim != 2 or x.shape[1] != 312 or y.shape != (len(x), 2) or rp.shape != (len(x),):
            raise ValueError("expected normalized [N,312] features, [N,2] cm positions and [N] RP IDs")
        if len(x) == 0 or not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("training arrays must be nonempty and finite")
        keys, inverse = np.unique(rp, return_inverse=True)
        counts = np.bincount(inverse)
        prototypes = np.zeros((len(keys), 312), dtype=np.float64)
        positions = np.zeros((len(keys), 2), dtype=np.float64)
        np.add.at(prototypes, inverse, x)
        np.add.at(positions, inverse, y)
        self.rp_ids = keys
        self.prototypes = prototypes / counts[:, None]
        self.coordinates_cm = positions / counts[:, None]
        return self

    def predict(self, features: np.ndarray, *, batch_size: int = 32) -> np.ndarray:
        if self.prototypes is None or self.coordinates_cm is None:
            raise RuntimeError("call fit before predict")
        queries = np.asarray(features, dtype=np.float64)
        if queries.ndim != 2 or queries.shape[1] != 312 or not np.isfinite(queries).all():
            raise ValueError("expected finite normalized [N,312] features")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        out = np.empty((len(queries), 2), dtype=np.float32)
        for start in range(0, len(queries), batch_size):
            stop = min(start + batch_size, len(queries))
            distance = ((queries[start:stop, None] - self.prototypes[None]) ** 2).sum(axis=2)
            out[start:stop] = self.coordinates_cm[distance.argmin(axis=1)].astype(np.float32)
        return out
