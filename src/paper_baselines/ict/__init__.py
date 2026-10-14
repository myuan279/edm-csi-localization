"""RI-312D acceleration and one-step localization baselines.

The package is intentionally independent from the frozen paper entrypoints.
It exposes only Development-scoped data access and requires explicit resolved
configuration binding before any formal run.
"""

PROTOCOL_ID = "ri_acceleration_onestep_v2"
MODEL_SCHEMA_VERSION = "ri_onestep_v2_schema_1"
SEEDS = (2031, 2032, 2033)
ALLOWED_SUBSETS = frozenset({"development_train_300k", "selection_validation_rp10"})
TRAINING_UPDATES = 58_600
