"""Topology- and dimension-controlled continuous-memory benchmark data."""

from .generator import (
    BANK_SCHEMA_VERSION,
    DELTA_T,
    GENERATOR_VERSION,
    GP_GRID_SPACING,
    GP_LENGTH_SCALE,
    GP_STD,
    TRAINING_HORIZON,
    ConditionSpec,
    ManifoldBatch,
    ParentBank,
    ParentSpec,
    derive_s1,
    derive_s2,
    derive_torus,
    make_parent_bank,
)

__all__ = [
    "BANK_SCHEMA_VERSION",
    "DELTA_T",
    "GENERATOR_VERSION",
    "GP_GRID_SPACING",
    "GP_LENGTH_SCALE",
    "GP_STD",
    "TRAINING_HORIZON",
    "ConditionSpec",
    "ManifoldBatch",
    "ParentBank",
    "ParentSpec",
    "derive_s1",
    "derive_s2",
    "derive_torus",
    "make_parent_bank",
]
