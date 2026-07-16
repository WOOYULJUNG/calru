"""Model-free exact and statistical audits for manifold benchmark banks."""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from .generator import ManifoldBatch, ParentBank


def _circular_delta(value: np.ndarray) -> np.ndarray:
    return np.remainder(value + math.pi, 2.0 * math.pi) - math.pi


def geodesic_steps(batch: ManifoldBatch) -> np.ndarray:
    topology = str(batch.metadata["topology"])
    if topology == "S2":
        dot = np.sum(batch.latent_path[:-1] * batch.latent_path[1:], axis=-1)
        return np.arccos(np.clip(dot, -1.0, 1.0))
    delta = _circular_delta(batch.latent_path[1:] - batch.latent_path[:-1])
    return np.linalg.norm(delta, axis=-1)


def endpoint_displacement(batch: ManifoldBatch) -> np.ndarray:
    topology = str(batch.metadata["topology"])
    if topology == "S2":
        dot = np.sum(batch.latent_path[0] * batch.latent_path[-1], axis=-1)
        return np.arccos(np.clip(dot, -1.0, 1.0))
    delta = _circular_delta(batch.latent_path[-1] - batch.latent_path[0])
    return np.linalg.norm(delta, axis=-1)


def topology_statistics(batch: ManifoldBatch) -> dict[str, Any]:
    steps = geodesic_steps(batch)
    path_length = np.sum(steps, axis=0)
    endpoint = endpoint_displacement(batch)
    active = batch.dwell_mask[..., 0] > 0.5
    effective_norm = np.linalg.norm(batch.effective_velocity, axis=-1)
    initial_norm = np.linalg.norm(batch.initial_memory, axis=-1)
    return {
        "topology": str(batch.metadata["topology"]),
        "trajectories": batch.trajectories,
        "horizon": batch.horizon,
        "active_step_fraction": float(np.mean(active)),
        "model_input_component_rms": float(np.sqrt(np.mean(batch.inputs**2))),
        "effective_tangent_rms": float(np.sqrt(np.mean(effective_norm**2))),
        "effective_tangent_rms_active": (
            float(np.sqrt(np.mean(effective_norm[active] ** 2)))
            if np.any(active)
            else 0.0
        ),
        "geodesic_step_mean": float(np.mean(steps)),
        "geodesic_step_median": float(np.median(steps)),
        "geodesic_step_p95": float(np.quantile(steps, 0.95)),
        "geodesic_step_p99": float(np.quantile(steps, 0.99)),
        "cumulative_path_length_mean": float(np.mean(path_length)),
        "cumulative_path_length_p95": float(np.quantile(path_length, 0.95)),
        "endpoint_displacement_mean": float(np.mean(endpoint)),
        "target_power_per_component": float(np.mean(batch.output_targets**2)),
        "initial_memory_norm_mean": float(np.mean(initial_norm)),
        "initial_memory_norm_std": float(np.std(initial_norm)),
    }


def _relative_difference(left: float, right: float) -> float:
    denominator = max(abs(float(left)), np.finfo(np.float64).tiny)
    return abs(float(left) - float(right)) / denominator


def topology_calibration(
    s1: ManifoldBatch, t2: ManifoldBatch, s2: ManifoldBatch
) -> dict[str, Any]:
    rows = {
        "S1": topology_statistics(s1),
        "T2": topology_statistics(t2),
        "S2": topology_statistics(s2),
    }
    comparison = {
        "geodesic_step_mean_relative_difference": _relative_difference(
            rows["T2"]["geodesic_step_mean"], rows["S2"]["geodesic_step_mean"]
        ),
        "geodesic_step_p95_relative_difference": _relative_difference(
            rows["T2"]["geodesic_step_p95"], rows["S2"]["geodesic_step_p95"]
        ),
        "cumulative_path_length_mean_relative_difference": _relative_difference(
            rows["T2"]["cumulative_path_length_mean"],
            rows["S2"]["cumulative_path_length_mean"],
        ),
        "active_step_fraction_absolute_difference": abs(
            rows["T2"]["active_step_fraction"]
            - rows["S2"]["active_step_fraction"]
        ),
    }
    thresholds = {
        "geodesic_step_mean_relative_difference": 0.03,
        "geodesic_step_p95_relative_difference": 0.05,
        "cumulative_path_length_mean_relative_difference": 0.05,
        "active_step_fraction_absolute_difference": 0.0,
    }
    checks = {
        name: comparison[name] <= threshold
        for name, threshold in thresholds.items()
    }
    return {
        "pass": bool(all(checks.values())),
        "statistics": rows,
        "T2_vs_S2": comparison,
        "thresholds": thresholds,
        "checks": checks,
    }


def exact_bank_audit(
    banks: Mapping[str, ManifoldBatch],
    *,
    float_tolerance: float,
) -> dict[str, Any]:
    if not banks:
        raise ValueError("at least one bank is required")
    checks: dict[str, bool] = {}
    first = next(iter(banks.values()))
    for name, bank in banks.items():
        arrays = (
            bank.initial_memory,
            bank.inputs,
            bank.output_targets,
            bank.latent_targets,
            bank.latent_path,
            bank.base_drive,
            bank.effective_velocity,
            bank.dwell_mask,
            bank.mask,
        )
        checks[f"{name}.finite"] = bool(all(np.all(np.isfinite(value)) for value in arrays))
        checks[f"{name}.target_path_alignment"] = bool(
            np.array_equal(bank.latent_targets, bank.latent_path[1:])
        )
        zero = bank.dwell_mask[..., 0] == 0
        checks[f"{name}.zero_dwell_input"] = bool(np.all(bank.inputs[zero] == 0))
        checks[f"{name}.zero_dwell_effective_velocity"] = bool(
            np.all(bank.effective_velocity[zero] == 0)
        )
        checks[f"{name}.trajectory_ids_unique"] = bool(
            len(np.unique(bank.trajectory_id)) == bank.trajectories
        )
        topology = str(bank.metadata["topology"])
        if topology == "S2":
            norm_error = np.max(np.abs(np.linalg.norm(bank.latent_path, axis=-1) - 1.0))
            tangency = np.max(
                np.abs(
                    np.sum(
                        bank.latent_path[:-1] * bank.effective_velocity, axis=-1
                    )
                )
            )
            checks[f"{name}.sphere_unit_norm"] = bool(norm_error <= float_tolerance)
            checks[f"{name}.sphere_tangency"] = bool(tangency <= float_tolerance)
            checks[f"{name}.initial_memory_q0"] = bool(
                np.array_equal(bank.initial_memory, bank.latent_path[0])
            )
            checks[f"{name}.output_target_q1_onward"] = bool(
                np.array_equal(bank.output_targets, bank.latent_path[1:])
            )
        else:
            pairs = bank.output_targets.reshape(
                *bank.output_targets.shape[:2], -1, 2
            )
            pair_error = np.max(np.abs(np.sum(pairs**2, axis=-1) - 1.0))
            checks[f"{name}.torus_pair_unit_norm"] = bool(
                pair_error <= float_tolerance
            )
            embedded_q0 = np.empty_like(bank.initial_memory)
            embedded_q0[:, 0::2] = np.cos(bank.latent_path[0])
            embedded_q0[:, 1::2] = np.sin(bank.latent_path[0])
            checks[f"{name}.initial_memory_q0"] = bool(
                np.max(np.abs(bank.initial_memory - embedded_q0)) <= float_tolerance
            )
    for name, bank in banks.items():
        checks[f"{name}.shared_trajectory_id"] = bool(
            np.array_equal(bank.trajectory_id, first.trajectory_id)
        )
        checks[f"{name}.shared_global_dwell"] = bool(
            np.array_equal(bank.dwell_mask, first.dwell_mask)
        )
    return {"pass": bool(all(checks.values())), "checks": checks}


def prefix_pairing_audit(
    short: Mapping[str, ManifoldBatch], long: Mapping[str, ManifoldBatch]
) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    for name, short_bank in short.items():
        long_bank = long[name]
        horizon = short_bank.horizon
        for field in (
            "inputs",
            "output_targets",
            "latent_targets",
            "base_drive",
            "effective_velocity",
            "dwell_mask",
            "mask",
        ):
            checks[f"{name}.{field}"] = bool(
                np.array_equal(getattr(short_bank, field), getattr(long_bank, field)[:horizon])
            )
        checks[f"{name}.latent_path"] = bool(
            np.array_equal(short_bank.latent_path, long_bank.latent_path[: horizon + 1])
        )
        if short_bank.latent_unwrapped is not None:
            checks[f"{name}.latent_unwrapped"] = bool(
                np.array_equal(
                    short_bank.latent_unwrapped,
                    long_bank.latent_unwrapped[: horizon + 1],
                )
            )
    return {"pass": bool(all(checks.values())), "checks": checks}


def dimension_nesting_audit(s1: ManifoldBatch, t8: ManifoldBatch) -> dict[str, Any]:
    checks = {
        "input_first_coordinate": np.array_equal(s1.inputs[..., 0], t8.inputs[..., 0]),
        "latent_first_coordinate": np.array_equal(
            s1.latent_path[..., 0], t8.latent_path[..., 0]
        ),
        "output_first_pair": np.array_equal(s1.output_targets, t8.output_targets[..., :2]),
        "initial_memory_first_pair": np.array_equal(
            s1.initial_memory, t8.initial_memory[..., :2]
        ),
    }
    return {"pass": bool(all(checks.values())), "checks": checks}


def parent_distribution_audit(parent: ParentBank) -> dict[str, Any]:
    spec = parent.metadata["parent_spec"]
    trajectories = int(spec["trajectories"])
    angles = parent.q0_angles
    fourier = {
        f"k{k}": float(np.max(np.abs(np.mean(np.exp(1j * k * angles), axis=0))))
        for k in range(1, 5)
    }
    sphere = parent.q0_sphere_gaussian
    sphere = sphere / np.linalg.norm(sphere, axis=1, keepdims=True)
    sphere_mean = np.mean(sphere, axis=0)
    second_moment = sphere.T @ sphere / trajectories
    second_eigenvalues = np.linalg.eigvalsh(second_moment)

    drive = parent.id_base_drive
    drive_mean = float(np.mean(drive))
    drive_variance = float(np.var(drive))
    expected_variance = float(spec["gp_std"]) ** 2 + float(
        spec["gp_cholesky_jitter"]
    )
    lags = [lag for lag in (1, 4, 16, 32, 64) if lag < parent.max_horizon]
    autocovariance: dict[str, dict[str, float]] = {}
    for lag in lags:
        empirical = float(np.mean(drive[:-lag] * drive[lag:]))
        expected = float(spec["gp_std"]) ** 2 * math.exp(
            -(
                lag
                * float(spec["gp_grid_spacing"])
                / float(spec["gp_length_scale"])
            )
            ** 2
            / 2.0
        )
        autocovariance[str(lag)] = {
            "empirical": empirical,
            "expected": expected,
            "relative_error": abs(empirical - expected) / expected,
        }
    active_probability = np.minimum(1.0, parent.sparsity_parameters)
    dwell = parent.mask_uniform_randoms < active_probability[None, :, None]
    thresholds = {
        "fourier_moment": 4.0 / math.sqrt(trajectories),
        "sphere_mean_abs": 4.0 / math.sqrt(3.0 * trajectories),
        "sphere_second_moment_eigen_abs_error": 0.06,
        "gp_mean_abs": 0.05,
        "gp_variance_relative_error": 0.05,
        "gp_autocovariance_relative_error": 0.05,
        "active_fraction_abs_error_from_0p75": 0.04,
    }
    checks = {
        "angle_fourier_uniformity": max(fourier.values())
        <= thresholds["fourier_moment"],
        "sphere_mean": float(np.max(np.abs(sphere_mean)))
        <= thresholds["sphere_mean_abs"],
        "sphere_second_moment": float(
            np.max(np.abs(second_eigenvalues - 1.0 / 3.0))
        )
        <= thresholds["sphere_second_moment_eigen_abs_error"],
        "gp_mean": abs(drive_mean) <= thresholds["gp_mean_abs"],
        "gp_variance": abs(drive_variance - expected_variance) / expected_variance
        <= thresholds["gp_variance_relative_error"],
        "gp_autocovariance": all(
            row["relative_error"] <= thresholds["gp_autocovariance_relative_error"]
            for row in autocovariance.values()
        ),
        "active_fraction": abs(float(np.mean(dwell)) - 0.75)
        <= thresholds["active_fraction_abs_error_from_0p75"],
    }
    return {
        "pass": bool(all(checks.values())),
        "checks": checks,
        "thresholds": thresholds,
        "angle_fourier_moments_max_over_coordinates": fourier,
        "sphere_mean": sphere_mean.tolist(),
        "sphere_second_moment_eigenvalues": second_eigenvalues.tolist(),
        "gp_mean": drive_mean,
        "gp_variance": drive_variance,
        "gp_expected_variance_including_jitter": expected_variance,
        "gp_autocovariance": autocovariance,
        "active_step_fraction": float(np.mean(dwell)),
    }


def linear_input_leakage_probe(
    batch: ManifoldBatch,
    *,
    seed: int = 0,
    maximum_rows: int = 200_000,
) -> dict[str, Any]:
    """Diagnostic linear probe of current latent from the current input token.

    The split is by trajectory.  A shuffled-training-input control uses the
    identical targets and test set.  This is a data diagnostic, not a neural
    model dependency.
    """

    rng = np.random.default_rng(seed)
    batch_count = batch.trajectories
    permutation = rng.permutation(batch_count)
    split = max(1, int(0.7 * batch_count))
    train_ids = permutation[:split]
    test_ids = permutation[split:]
    if test_ids.size == 0:
        raise ValueError("leakage probe requires at least two trajectories")
    current_latent = batch.latent_path[:-1]

    def flatten(ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = np.transpose(batch.inputs[:, ids], (1, 0, 2)).reshape(-1, batch.inputs.shape[-1])
        y = np.transpose(current_latent[:, ids], (1, 0, 2)).reshape(
            -1, current_latent.shape[-1]
        )
        if x.shape[0] > maximum_rows:
            rows = rng.choice(x.shape[0], size=maximum_rows, replace=False)
            x, y = x[rows], y[rows]
        return x.astype(np.float64), y.astype(np.float64)

    x_train, y_train = flatten(train_ids)
    x_test, y_test = flatten(test_ids)

    def fit_predict(x: np.ndarray, y: np.ndarray, test: np.ndarray) -> np.ndarray:
        design = np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)
        test_design = np.concatenate((test, np.ones((test.shape[0], 1))), axis=1)
        gram = design.T @ design
        gram.flat[:: gram.shape[0] + 1] += 1e-6
        weights = np.linalg.solve(gram, design.T @ y)
        return test_design @ weights

    prediction = fit_predict(x_train, y_train, x_test)
    shuffled_x = x_train[rng.permutation(x_train.shape[0])]
    shuffled_prediction = fit_predict(shuffled_x, y_train, x_test)
    denominator = float(np.sum((y_test - np.mean(y_test, axis=0)) ** 2))
    real_r2 = 1.0 - float(np.sum((y_test - prediction) ** 2)) / denominator
    shuffled_r2 = 1.0 - float(np.sum((y_test - shuffled_prediction) ** 2)) / denominator
    improvement = real_r2 - shuffled_r2
    return {
        "pass": bool(improvement < 0.01),
        "gate": "real_minus_shuffled_R2_below_0p01",
        "real_R2": real_r2,
        "shuffled_R2": shuffled_r2,
        "improvement": improvement,
        "training_trajectories": int(train_ids.size),
        "test_trajectories": int(test_ids.size),
    }


__all__ = [
    "dimension_nesting_audit",
    "endpoint_displacement",
    "exact_bank_audit",
    "geodesic_steps",
    "linear_input_leakage_probe",
    "parent_distribution_audit",
    "prefix_pairing_audit",
    "topology_calibration",
    "topology_statistics",
]
