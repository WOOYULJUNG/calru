"""Phase-gated, provenance-locked GPU launcher for the frozen ring pilot.

Every child runs in a unique attempt directory.  Only a successful child with
a receipt matching this campaign is atomically published under ``runs/`` or
``analysis/``.  Failed and interrupted attempts are retained under
``attempts/`` so a partial directory can never masquerade as a completed run
or permanently block a safe resume.
"""

from __future__ import annotations

import argparse
import fcntl
import functools
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, IO, Sequence

import numpy as np

from .artifacts import (
    RECEIPT_IDENTITY_ENV,
    atomic_bytes,
    atomic_json,
    canonical_hash,
    sha256_file,
    strict_json_load,
    strict_json_loads,
    verify_completion_receipt,
)
from .aggregation import (
    expected_aggregation_bytes,
    aggregation_specification,
)
from .config import (
    DEFAULT_PROTOCOL_PATH,
    expand_phase1_runs,
    load_protocol,
    protocol_fingerprint,
)
from .tasks import load_fixed_bank, sample_angular_integration, save_fixed_bank


ROOT_MARKER = ".calru_sagodi_protocol_root_v1"
OUTPUT_DIR_TOKEN = "__CALRU_ATTEMPT_OUTPUT_DIR__"


@dataclass(frozen=True)
class Job:
    job_id: str
    stage: str
    output_dir: Path
    command: tuple[str, ...]
    receipt_job_id: str


@dataclass
class ActiveJob:
    job: Job
    process: subprocess.Popen[Any]
    handle: IO[bytes]
    gpu: int
    attempt_dir: Path
    log_path: Path
    process_identity: dict[str, Any] | None = None


def _process_identity(pid: int) -> dict[str, Any] | None:
    """Return Linux process-start identity robust to PID reuse."""

    proc = Path("/proc") / str(int(pid))
    try:
        stat = (proc / "stat").read_text()
        end = stat.rfind(")")
        if end < 0:
            return None
        fields_after_comm = stat[end + 2 :].split()
        # fields_after_comm[0] is field 3 (state), so field 22 is index 19.
        start_ticks = int(fields_after_comm[19])
        cmdline = (proc / "cmdline").read_bytes()
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except (OSError, ValueError, IndexError):
        return None
    return {
        "pid": int(pid),
        "boot_id": boot_id,
        "proc_start_ticks": start_ticks,
        "cmdline_sha256": hashlib.sha256(cmdline).hexdigest(),
    }


@dataclass
class CampaignLock:
    path: Path
    token: str
    guard_path: Path
    guard_descriptor: int | None

    def release(self) -> None:
        descriptor = self.guard_descriptor
        if descriptor is None:
            return
        try:
            try:
                payload = strict_json_load(self.path)
            except FileNotFoundError:
                pass
            except (OSError, ValueError, TypeError):
                # Never delete a payload whose ownership can no longer be proven.
                pass
            else:
                if payload.get("token") == self.token:
                    self.path.unlink(missing_ok=True)
        finally:
            # The persistent guard inode is deliberately never unlinked: removing
            # it would reintroduce an inode-replacement race between flock users.
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)
            self.guard_descriptor = None


def _lock_owner_is_live(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    expected = payload.get("owner_process_identity")
    if not isinstance(expected, dict):
        return False
    try:
        pid = int(expected["pid"])
    except (KeyError, TypeError, ValueError):
        return False
    current = _process_identity(pid)
    return current is not None and current == expected


def _acquire_campaign_lock(root: Path) -> CampaignLock:
    """Acquire the sole orchestrator lock without an empty-payload race.

    A persistent, never-unlinked guard inode is locked first and held for the
    campaign lifetime.  The human-readable token payload is written only while
    that guard is held.  Therefore a second owner can never classify a
    just-created-but-not-yet-written payload as stale.
    """

    lock_path = root / ".orchestrator.lock"
    guard_path = root / ".orchestrator.guard"
    owner = _process_identity(os.getpid())
    if owner is None:
        raise RuntimeError("cannot establish current orchestrator process identity")
    descriptor = os.open(guard_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            try:
                existing = strict_json_load(lock_path)
            except (OSError, ValueError, TypeError):
                raise RuntimeError(
                    "campaign already has a live orchestrator; its guarded owner "
                    "payload is not yet visible or is being written"
                ) from exc
            raise RuntimeError(
                "campaign already has a live orchestrator: "
                f"{existing.get('owner_process_identity')}"
            ) from exc

        if lock_path.exists():
            try:
                existing = strict_json_load(lock_path)
            except (OSError, ValueError, TypeError):
                existing = None
            if _lock_owner_is_live(existing):
                raise RuntimeError(
                    "campaign lock payload names a live orchestrator even though "
                    f"the guard was free: {existing['owner_process_identity']}"
                )
            _preserve_invalid_output(
                root, "campaign_setup", "orchestrator_lock", lock_path
            )

        token = uuid.uuid4().hex
        payload = {
            "schema_version": 2,
            "token": token,
            "owner_process_identity": owner,
            "guard_inode": int(os.fstat(descriptor).st_ino),
            "acquired_at": time.time(),
        }
        try:
            atomic_json(lock_path, payload)
        except BaseException:
            lock_path.unlink(missing_ok=True)
            raise
        return CampaignLock(lock_path, token, guard_path, descriptor)
    except BaseException:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)
        raise


def _validated_gpu_ids(gpus: Sequence[int]) -> tuple[int, ...]:
    values = tuple(int(value) for value in gpus)
    if not values:
        raise ValueError("at least one GPU is required")
    if any(value < 0 for value in values):
        raise ValueError(f"GPU ids must be non-negative: {values}")
    if len(set(values)) != len(values):
        raise ValueError(f"duplicate GPU ids are forbidden: {values}")
    return values


def _single_orchestrator_locked(function: Any) -> Any:
    @functools.wraps(function)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        kwargs["gpus"] = _validated_gpu_ids(kwargs["gpus"])
        root = Path(kwargs["artifact_root"]).resolve()
        _prepare_root(root)
        lock = _acquire_campaign_lock(root)
        previous_term = signal.getsignal(signal.SIGTERM)
        previous_int = signal.getsignal(signal.SIGINT)

        def interrupt(signum: int, _frame: Any) -> None:
            raise KeyboardInterrupt(f"campaign interrupted by signal {signum}")

        signal.signal(signal.SIGTERM, interrupt)
        signal.signal(signal.SIGINT, interrupt)
        try:
            return function(*args, **kwargs)
        finally:
            lock.release()
            signal.signal(signal.SIGTERM, previous_term)
            signal.signal(signal.SIGINT, previous_int)

    return wrapped


def _source_hashes(repo_root: Path, package_dir: Path) -> dict[str, str]:
    paths = list(package_dir.glob("*.py"))
    paths.extend(
        repo_root / relative
        for relative in (
            "repro/legacy_code/exp71_pan_block_pulse_hold.py",
            "repro/legacy_code/exp72_structured_attractor_tasks.py",
            "repro/legacy_code/pan_block.py",
            "repro/legacy_code/plru_regularizers.py",
        )
    )
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"campaign source files are missing: {missing}")
    return {
        str(path.resolve().relative_to(repo_root)): sha256_file(path)
        for path in sorted(paths)
    }


def _prepare_root(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    marker = root / ROOT_MARKER
    if marker.exists():
        if not marker.is_file():
            raise RuntimeError(f"campaign root marker is not a file: {marker}")
        return
    if any(root.iterdir()):
        raise RuntimeError(f"artifact root is nonempty and has no safety marker: {root}")
    atomic_bytes(marker, b"CA-LRU Sagodi protocol artifacts; do not mix with legacy runs.\n")


def _write_or_check_manifest(root: Path, manifest: dict[str, Any]) -> None:
    path = root / "manifest.json"
    if path.exists():
        existing = strict_json_load(path)
        if existing != manifest:
            raise RuntimeError("campaign manifest differs; choose a new artifact root")
    else:
        atomic_json(path, manifest)


def _git_state(repo_root: Path) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    ).stdout.strip()
    porcelain = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    ).stdout
    return {"code_commit": commit, "worktree_dirty": bool(porcelain.strip())}


def _resolve_python(python: str) -> str:
    candidate = Path(python).expanduser()
    if candidate.is_absolute() or candidate.parent != Path("."):
        resolved = candidate.resolve()
    else:
        found = shutil.which(str(candidate))
        if found is None:
            raise FileNotFoundError(f"Python executable not found: {python}")
        resolved = Path(found).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Python executable not found: {resolved}")
    return str(resolved)


def _environment_fingerprint(python: str, selected_gpus: Sequence[int]) -> dict[str, Any]:
    script = r'''
import json, platform, sys
import numpy as np
import torch
devices = []
if torch.cuda.is_available():
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        devices.append({
            "index": index,
            "name": props.name,
            "total_memory": int(props.total_memory),
            "compute_capability": [int(props.major), int(props.minor)],
            "multi_processor_count": int(props.multi_processor_count),
        })
print(json.dumps({
    "python": platform.python_version(),
    "python_implementation": platform.python_implementation(),
    "python_executable": sys.executable,
    "platform": platform.platform(),
    "machine": platform.machine(),
    "numpy": np.__version__,
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cudnn": torch.backends.cudnn.version(),
    "cuda_available": torch.cuda.is_available(),
    "cuda_devices": devices,
}, sort_keys=True, allow_nan=False))
'''
    result = subprocess.run(
        [python, "-c", script],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
        env={**os.environ, "PYTHONNOUSERSITE": "1"},
    )
    payload = strict_json_loads(result.stdout)
    payload["python_executable_sha256"] = sha256_file(python)
    payload["selected_gpu_indices"] = [int(value) for value in selected_gpus]
    available = {int(item["index"]): item for item in payload["cuda_devices"]}
    missing = [int(value) for value in selected_gpus if int(value) not in available]
    if missing:
        raise ValueError(f"selected GPU indices are unavailable: {missing}")
    payload["selected_gpu_fingerprint"] = [available[int(value)] for value in selected_gpus]
    return payload


def _unique_attempt_path(root: Path, stage: str, job_id: str, label: str = "attempt") -> Path:
    parent = root / "attempts" / stage / job_id
    parent.mkdir(parents=True, exist_ok=True)
    for _ in range(100):
        token = f"{time.time_ns()}-pid{os.getpid()}-{uuid.uuid4().hex[:8]}"
        candidate = parent / f"{label}-{token}"
        try:
            candidate.mkdir()
        except FileExistsError:
            continue
        return candidate
    raise RuntimeError(f"could not allocate unique attempt directory below {parent}")


def _preserve_invalid_output(root: Path, stage: str, job_id: str, output: Path) -> Path:
    """Move a non-publishable final directory aside without deleting anything."""

    if not output.exists():
        raise FileNotFoundError(output)
    destination = _unique_attempt_path(root, stage, job_id, label="recovered")
    destination.rmdir()
    os.replace(output, destination)
    return destination


def _materialize_evaluation_bank(root: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    phase = protocol["phase1_ring_pilot"]
    task = phase["task"]
    seeds = protocol["seed_policy"]
    count = int(protocol["evaluation"]["id_test_trials"])
    horizon = int(task["sequence_steps"])
    bank_dir = root / "evaluation_bank"
    bank_path = bank_dir / "angular_integration_id.npz"
    sidecar = Path(f"{bank_path}.sha256")

    def validate() -> str:
        batch = load_fixed_bank(bank_path)
        if batch.batch_size != count or batch.time_steps != horizon:
            raise ValueError(
                f"evaluation bank shape mismatch: expected T={horizon}, B={count}, "
                f"got T={batch.time_steps}, B={batch.batch_size}"
            )
        metadata = batch.metadata
        if int(metadata.get("task_seed", -1)) != int(seeds["task_seed"]):
            raise ValueError("evaluation bank task_seed mismatch")
        if int(metadata.get("sample_seed", -1)) != int(seeds["evaluation_bank_seed"]):
            raise ValueError("evaluation bank sample_seed mismatch")
        return sha256_file(bank_path)

    if bank_dir.exists():
        try:
            digest = validate()
        except Exception:
            if (root / "manifest.json").exists():
                raise RuntimeError("manifested evaluation bank is missing, corrupt, or incompatible")
            _preserve_invalid_output(root, "campaign_setup", "evaluation_bank", bank_dir)
        else:
            return {
                "path": str(bank_path.relative_to(root)),
                "sha256": digest,
                "sidecar_sha256": sha256_file(sidecar),
                "trials": count,
                "horizon": horizon,
                "task_seed": int(seeds["task_seed"]),
                "sample_seed": int(seeds["evaluation_bank_seed"]),
            }

    attempt = _unique_attempt_path(root, "campaign_setup", "evaluation_bank")
    staged_bank = attempt / bank_path.name
    batch = sample_angular_integration(
        count,
        horizon,
        int(seeds["task_seed"]),
        int(seeds["evaluation_bank_seed"]),
        "hidden-init",
    )
    save_fixed_bank(staged_bank, batch)
    load_fixed_bank(staged_bank)
    bank_dir.parent.mkdir(parents=True, exist_ok=True)
    os.replace(attempt, bank_dir)
    digest = validate()
    return {
        "path": str(bank_path.relative_to(root)),
        "sha256": digest,
        "sidecar_sha256": sha256_file(sidecar),
        "trials": count,
        "horizon": horizon,
        "task_seed": int(seeds["task_seed"]),
        "sample_seed": int(seeds["evaluation_bank_seed"]),
    }


def _materialize_perturbation_bank(root: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    """Freeze common raw Gaussian directions before model-specific projection."""

    seed = int(protocol["seed_policy"]["perturbation_bank_seed"])
    evaluation = protocol["evaluation"]
    width = int(protocol["phase1_ring_pilot"]["training"]["width"])
    finite_count = int(evaluation["finite_kick_anchor_count"])
    jacobian_count = int(evaluation["jacobian_anchor_count"])
    directions = int(evaluation["ambient_random_directions_per_anchor"])
    bank_dir = root / "perturbation_bank"
    bank_path = bank_dir / f"phase1_ring_seed{seed}.npz"
    sidecar = Path(f"{bank_path}.sha256")
    expected_shapes = {
        "finite_raw": (finite_count, directions, width),
        "jacobian_raw": (jacobian_count, directions, width),
    }
    metadata = {
        "schema_version": 1,
        "seed": seed,
        "generator": "numpy.random.Generator(numpy.random.PCG64)",
        "dtype": "float32",
        "finite_raw_shape": list(expected_shapes["finite_raw"]),
        "jacobian_raw_shape": list(expected_shapes["jacobian_raw"]),
        "semantics": (
            "common raw ambient Gaussian vectors; each analysis projects onto "
            "the checkpoint-local normal space and normalizes"
        ),
        "smoke_rule": "use deterministic prefix slices",
    }

    def validate() -> str:
        if not sidecar.is_file():
            raise FileNotFoundError(f"perturbation-bank sidecar missing: {sidecar}")
        fields = sidecar.read_text(encoding="ascii").strip().split()
        if len(fields) != 2 or fields[1] != bank_path.name:
            raise ValueError("malformed perturbation-bank sidecar")
        digest = sha256_file(bank_path)
        if fields[0] != digest:
            raise ValueError("perturbation-bank SHA-256 mismatch")
        with np.load(bank_path, allow_pickle=False) as archive:
            if set(archive.files) != {"finite_raw", "jacobian_raw", "metadata_json"}:
                raise ValueError("unexpected perturbation-bank arrays")
            for name, shape in expected_shapes.items():
                array = archive[name]
                if array.shape != shape or array.dtype != np.float32:
                    raise ValueError(f"perturbation-bank shape/dtype mismatch for {name}")
            decoded = strict_json_loads(
                np.asarray(archive["metadata_json"], dtype=np.uint8).tobytes().decode("utf-8")
            )
            if decoded != metadata:
                raise ValueError("perturbation-bank metadata mismatch")
        return digest

    if bank_dir.exists():
        try:
            digest = validate()
        except Exception:
            if (root / "manifest.json").exists():
                raise RuntimeError("manifested perturbation bank is missing or corrupt")
            _preserve_invalid_output(root, "campaign_setup", "perturbation_bank", bank_dir)
        else:
            return {
                "path": str(bank_path.relative_to(root)),
                "sha256": digest,
                "sidecar_sha256": sha256_file(sidecar),
                "specification": metadata,
                "specification_sha256": canonical_hash(metadata),
            }

    attempt = _unique_attempt_path(root, "campaign_setup", "perturbation_bank")
    staged_bank = attempt / bank_path.name
    generator = np.random.Generator(np.random.PCG64(seed))
    arrays = {
        "finite_raw": generator.standard_normal(
            expected_shapes["finite_raw"], dtype=np.float32
        ),
        "jacobian_raw": generator.standard_normal(
            expected_shapes["jacobian_raw"], dtype=np.float32
        ),
    }
    metadata_bytes = json.dumps(
        metadata,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    descriptor, raw = tempfile.mkstemp(prefix=f".{bank_path.name}.", dir=attempt)
    os.close(descriptor)
    temporary = Path(raw)
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                **arrays,
                metadata_json=np.frombuffer(metadata_bytes, dtype=np.uint8),
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, staged_bank)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    digest = sha256_file(staged_bank)
    atomic_bytes(
        Path(f"{staged_bank}.sha256"),
        f"{digest}  {staged_bank.name}\n".encode("ascii"),
    )
    os.replace(attempt, bank_dir)
    digest = validate()
    return {
        "path": str(bank_path.relative_to(root)),
        "sha256": digest,
        "sidecar_sha256": sha256_file(sidecar),
        "specification": metadata,
        "specification_sha256": canonical_hash(metadata),
    }


def _receipt_expectations(runs: Sequence[dict[str, Any]]) -> dict[str, dict[str, str]]:
    result = {
        "phase0": {
            "job_id": "phase0_state_audit",
            "stage": "phase0",
            "run_id": "phase0",
            "output": "phase0",
        }
    }
    for run in runs:
        run_id = str(run["run_id"])
        train_key = f"training:{run_id}"
        analysis_key = f"analysis:{run_id}"
        result[train_key] = {
            "job_id": (
                f"{run['model']['id']}-seed{int(run['model_seed'])}-"
                f"lr{float(run['learning_rate']):g}"
            ),
            "stage": "training",
            "run_id": run_id,
            "output": f"runs/{run_id}",
        }
        result[analysis_key] = {
            "job_id": f"phase1-analysis-{run_id}",
            "stage": "analysis",
            "run_id": run_id,
            "output": f"analysis/{run_id}",
        }
    return result


def _receipt_metadata(manifest: dict[str, Any], stage: str, run_id: str) -> dict[str, str]:
    return {
        "campaign_scientific_identity": str(manifest["scientific_identity"]),
        "protocol_fingerprint": str(manifest["protocol_canonical_fingerprint"]),
        "run_id": str(run_id),
        "stage": str(stage),
    }


def _training_checkpoint_binding(
    root: Path,
    manifest: dict[str, Any],
    run_id: str,
    *,
    training_output: Path | None = None,
) -> tuple[bool, str, str | None]:
    """Bind a training receipt and manifest to the checkpoint bytes on disk."""

    expectation = manifest["receipt_expectations"].get(f"training:{run_id}")
    if not isinstance(expectation, dict):
        return False, f"missing training receipt expectation for {run_id}", None
    output = (
        Path(training_output)
        if training_output is not None
        else root / str(expectation["output"])
    )
    receipt_path = output / "completion_receipt.json"
    valid, reason = verify_completion_receipt(
        receipt_path,
        expected_job_id=expectation["job_id"],
        expected_metadata=_receipt_metadata(manifest, "training", run_id),
    )
    if not valid:
        return False, f"training receipt invalid: {reason}", None
    checkpoint = output / "checkpoint.pt"
    training_manifest_path = output / "manifest.json"
    if not checkpoint.is_file() or not training_manifest_path.is_file():
        return False, "training checkpoint or manifest is missing", None
    try:
        digest = sha256_file(checkpoint)
        training_manifest = strict_json_load(training_manifest_path)
        receipt = strict_json_load(receipt_path)
    except (OSError, ValueError, TypeError) as exc:
        return False, f"cannot read training checkpoint provenance: {exc}", None
    if training_manifest.get("checkpoint_sha256") != digest:
        return False, "training manifest checkpoint_sha256 mismatch", None
    if training_manifest.get("campaign_identity") != manifest["scientific_identity"]:
        return False, "training manifest campaign_identity mismatch", None
    if (
        training_manifest.get("protocol_canonical_fingerprint")
        != manifest["protocol_canonical_fingerprint"]
    ):
        return False, "training manifest protocol fingerprint mismatch", None
    if int(receipt.get("schema_version", -1)) != 2:
        return False, "campaign training receipt must use relocatable schema 2", None
    artifacts = receipt.get("artifacts", {})
    if artifacts.get("checkpoint.pt") != digest:
        return False, "training receipt does not bind checkpoint.pt to current bytes", None
    if artifacts.get("manifest.json") != sha256_file(training_manifest_path):
        return False, "training receipt does not bind the current training manifest", None
    return True, "training receipt/manifest/checkpoint binding verified", digest


def _verify_campaign_inputs(
    root: Path, manifest: dict[str, Any], repo_root: Path
) -> None:
    payload = manifest.get("scientific_identity_payload")
    if not isinstance(payload, dict) or canonical_hash(payload) != manifest.get("scientific_identity"):
        raise RuntimeError("campaign scientific identity does not match its manifest payload")
    for key, expected in payload.items():
        if manifest.get(key) != expected:
            raise RuntimeError(
                f"manifest field {key!r} differs from the signed scientific payload"
            )
    protocol_path = Path(manifest["protocol_file"]).resolve()
    if sha256_file(protocol_path) != manifest["protocol_file_sha256"]:
        raise RuntimeError("frozen protocol file changed after campaign creation")
    protocol = load_protocol(protocol_path)
    if protocol_fingerprint(protocol) != manifest["protocol_canonical_fingerprint"]:
        raise RuntimeError("canonical protocol fingerprint changed after campaign creation")
    current_sources = _source_hashes(repo_root, Path(__file__).resolve().parent)
    if current_sources != manifest["source_hashes"]:
        raise RuntimeError("campaign source code changed after campaign creation")
    if _git_state(repo_root) != manifest["code"]:
        raise RuntimeError("git commit/dirty state changed after campaign creation")
    source_note = repo_root / str(protocol["source_protocol"]["path"])
    if not source_note.is_file() or sha256_file(source_note) != manifest["source_protocol_sha256"]:
        raise RuntimeError("normative source protocol note is missing or changed")
    bank = manifest["evaluation_bank"]
    bank_path = root / bank["path"]
    if sha256_file(bank_path) != bank["sha256"]:
        raise RuntimeError("fixed evaluation bank changed after campaign creation")
    sidecar = Path(f"{bank_path}.sha256")
    if sha256_file(sidecar) != bank["sidecar_sha256"]:
        raise RuntimeError("fixed evaluation bank sidecar changed after campaign creation")
    load_fixed_bank(bank_path)
    perturbation = manifest["perturbation_bank"]
    perturbation_path = root / perturbation["path"]
    if sha256_file(perturbation_path) != perturbation["sha256"]:
        raise RuntimeError("fixed perturbation bank changed after campaign creation")
    perturbation_sidecar = Path(f"{perturbation_path}.sha256")
    if sha256_file(perturbation_sidecar) != perturbation["sidecar_sha256"]:
        raise RuntimeError("fixed perturbation bank sidecar changed after campaign creation")
    if canonical_hash(perturbation["specification"]) != perturbation["specification_sha256"]:
        raise RuntimeError("perturbation-bank specification hash mismatch")
    if manifest.get("pilot_aggregation") != aggregation_specification():
        raise RuntimeError("pilot aggregation specification differs from frozen implementation")


def _training_jobs(
    protocol: dict[str, Any],
    *,
    root: Path,
    python: str,
    protocol_path: Path,
    evaluation_bank: Path,
    campaign_identity: str,
    smoke: bool,
) -> list[Job]:
    jobs: list[Job] = []
    for run in expand_phase1_runs(protocol):
        output = root / "runs" / run["run_id"]
        command = [
            python,
            "-m",
            "repro.sagodi_protocol.train",
            "--protocol",
            str(protocol_path),
            "--model",
            run["model"]["id"],
            "--model-seed",
            str(run["model_seed"]),
            "--learning-rate",
            str(run["learning_rate"]),
            "--output-dir",
            OUTPUT_DIR_TOKEN,
            "--evaluation-bank",
            str(evaluation_bank),
            "--campaign-identity",
            str(campaign_identity),
            "--device",
            "cuda:0",
        ]
        if smoke:
            command.append("--smoke")
        else:
            command.extend(
                [
                    "--state-spec",
                    str(
                        root
                        / "phase0"
                        / f"model={run['model']['id']}"
                        / "state_spec.json"
                    ),
                ]
            )
        receipt_job_id = (
            f"{run['model']['id']}-seed{int(run['model_seed'])}-"
            f"lr{float(run['learning_rate']):g}"
        )
        jobs.append(Job(run["run_id"], "training", output, tuple(command), receipt_job_id))
    return jobs


def _analysis_jobs(
    training_jobs: Sequence[Job],
    *,
    root: Path,
    python: str,
    protocol_path: Path,
    evaluation_bank: Path,
    perturbation_bank: Path,
    campaign_identity: str,
    smoke: bool,
) -> list[Job]:
    jobs: list[Job] = []
    for training in training_jobs:
        output = root / "analysis" / training.job_id
        command = [
            python,
            "-m",
            "repro.sagodi_protocol.phase1_analysis",
            "--protocol",
            str(protocol_path),
            "--run-dir",
            str(training.output_dir),
            "--output-dir",
            OUTPUT_DIR_TOKEN,
            "--evaluation-bank",
            str(evaluation_bank),
            "--perturbation-bank",
            str(perturbation_bank),
            "--campaign-identity",
            str(campaign_identity),
            "--device",
            "cuda:0",
        ]
        if smoke:
            command.append("--smoke")
        jobs.append(
            Job(
                training.job_id,
                "analysis",
                output,
                tuple(command),
                f"phase1-analysis-{training.job_id}",
            )
        )
    return jobs


def _verify_job_receipt(
    receipt: Path, job: Job, manifest: dict[str, Any]
) -> tuple[bool, str]:
    valid, reason = verify_completion_receipt(
        receipt,
        expected_job_id=job.receipt_job_id,
        expected_metadata=_receipt_metadata(manifest, job.stage, job.job_id),
    )
    if not valid:
        return valid, reason
    root = job.output_dir.parent.parent
    if job.stage == "training":
        bound, binding_reason, _ = _training_checkpoint_binding(
            root,
            manifest,
            job.job_id,
            training_output=receipt.parent,
        )
        return bound, binding_reason
    if job.stage == "analysis":
        bound, binding_reason, checkpoint_sha256 = _training_checkpoint_binding(
            root, manifest, job.job_id
        )
        if not bound:
            return False, binding_reason
        valid, reason = verify_completion_receipt(
            receipt,
            expected_job_id=job.receipt_job_id,
            expected_metadata={
                **_receipt_metadata(manifest, job.stage, job.job_id),
                "checkpoint_sha256": checkpoint_sha256,
            },
        )
        if not valid:
            return False, f"analysis/checkpoint binding invalid: {reason}"
        return True, "analysis receipt bound to current verified training checkpoint"
    return True, "verified campaign receipt"


def verify_campaign_output_receipt(
    root: Path, manifest: dict[str, Any], key: str
) -> tuple[bool, str]:
    expectation = manifest["receipt_expectations"][key]
    receipt = root / expectation["output"] / "completion_receipt.json"
    if expectation["stage"] == "phase0":
        return verify_completion_receipt(
            receipt,
            expected_job_id=expectation["job_id"],
            expected_metadata=_receipt_metadata(manifest, "phase0", "phase0"),
        )
    job = Job(
        str(expectation["run_id"]),
        str(expectation["stage"]),
        root / str(expectation["output"]),
        (),
        str(expectation["job_id"]),
    )
    return _verify_job_receipt(receipt, job, manifest)


def _job_state(job: Job, manifest: dict[str, Any]) -> tuple[str, str]:
    valid, reason = _verify_job_receipt(
        job.output_dir / "completion_receipt.json", job, manifest
    )
    if valid:
        return "complete", "verified campaign receipt"
    if job.output_dir.exists():
        return "invalid", reason
    return "pending", reason


def _render_command(job: Job, attempt_dir: Path) -> list[str]:
    rendered = [str(attempt_dir) if value == OUTPUT_DIR_TOKEN else value for value in job.command]
    if rendered.count(str(attempt_dir)) != 1:
        raise RuntimeError(f"job command must contain exactly one output token: {job.job_id}")
    return rendered


def _child_environment(
    manifest: dict[str, Any], *, stage: str, run_id: str, gpu: int | None = None
) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(int(gpu))
    values = _receipt_metadata(manifest, stage, run_id)
    for metadata_key, environment_key in RECEIPT_IDENTITY_ENV.items():
        env[environment_key] = values[metadata_key]
    return env


def _publish_attempt(
    root: Path, job: Job, attempt_dir: Path, manifest: dict[str, Any]
) -> None:
    valid, reason = _verify_job_receipt(
        attempt_dir / "completion_receipt.json", job, manifest
    )
    if not valid:
        raise RuntimeError(f"cannot publish invalid attempt {attempt_dir}: {reason}")
    if job.output_dir.exists():
        existing, existing_reason = _job_state(job, manifest)
        if existing == "complete":
            raise RuntimeError(f"concurrent publication detected for {job.output_dir}")
        _preserve_invalid_output(root, job.stage, job.job_id, job.output_dir)
        print(
            f"[preserve invalid {job.stage}] {job.job_id}: {existing_reason}",
            flush=True,
        )
    job.output_dir.parent.mkdir(parents=True, exist_ok=True)
    os.replace(attempt_dir, job.output_dir)
    published, published_reason = _job_state(job, manifest)
    if published != "complete":
        raise RuntimeError(f"published receipt failed verification: {published_reason}")


def _terminate_active(
    active: dict[int, ActiveJob],
    *,
    status: dict[str, Any],
    root: Path,
    reason: str,
    grace: float = 10.0,
    manifest: dict[str, Any] | None = None,
    repo_root: Path | None = None,
) -> None:
    initial_exit_codes = {
        gpu: item.process.poll() for gpu, item in active.items()
    }
    for item in active.values():
        if item.process.poll() is None:
            try:
                os.killpg(item.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.time() + float(grace)
    while time.time() < deadline and any(
        item.process.poll() is None for item in active.values()
    ):
        time.sleep(0.1)
    for item in active.values():
        if item.process.poll() is None:
            try:
                os.killpg(item.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    for gpu, item in list(active.items()):
        try:
            exit_code = item.process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            exit_code = item.process.poll()
        if not item.handle.closed:
            item.handle.close()
        child_reason = reason
        if manifest is not None and repo_root is not None:
            try:
                _verify_campaign_inputs(root, manifest, repo_root)
            except Exception as exc:
                child_reason = f"{reason}; post-child campaign integrity failed: {exc}"
        state = "terminated" if initial_exit_codes[gpu] is None else "abandoned"
        state_reason = (
            child_reason
            if state == "terminated"
            else f"{child_reason}; child had already exited but was not published"
        )
        status["jobs"][f"{item.job.stage}:{item.job.job_id}"] = {
            "state": state,
            "gpu": gpu,
            "pid": item.process.pid,
            "process_identity": item.process_identity,
            "exit_code": exit_code,
            "reason": state_reason,
            "attempt_dir": str(item.attempt_dir),
            "log": str(item.log_path),
        }
        active.pop(gpu, None)
    atomic_json(root / "status.json", status)


def _run_jobs(
    jobs: Sequence[Job],
    *,
    gpus: Sequence[int],
    root: Path,
    repo_root: Path,
    manifest: dict[str, Any],
    status: dict[str, Any],
) -> None:
    gpus = _validated_gpu_ids(gpus)
    pending: list[Job] = []
    for job in jobs:
        state, reason = _job_state(job, manifest)
        if state == "invalid":
            preserved = _preserve_invalid_output(
                root, job.stage, job.job_id, job.output_dir
            )
            status["jobs"][f"{job.stage}:{job.job_id}"] = {
                "state": "pending",
                "reason": f"invalid prior output preserved at {preserved}: {reason}",
            }
            pending.append(job)
        else:
            status["jobs"][f"{job.stage}:{job.job_id}"] = {
                "state": state,
                "reason": reason,
            }
            if state == "pending":
                pending.append(job)
    atomic_json(root / "status.json", status)

    available = sorted(map(int, gpus))
    active: dict[int, ActiveJob] = {}
    stopping = False

    def handle_signal(signum: int, _frame: Any) -> None:
        nonlocal stopping
        stopping = True
        status["interrupted_by_signal"] = int(signum)

    previous_term = signal.signal(signal.SIGTERM, handle_signal)
    previous_int = signal.signal(signal.SIGINT, handle_signal)
    try:
        while pending or active:
            while pending and available and not stopping:
                _verify_campaign_inputs(root, manifest, repo_root)
                gpu = available.pop(0)
                job = pending.pop(0)
                attempt = _unique_attempt_path(root, job.stage, job.job_id)
                log_dir = root / "logs" / job.stage / job.job_id
                log_dir.mkdir(parents=True, exist_ok=True)
                log_path = log_dir / f"{attempt.name}.log"
                handle = log_path.open("ab", buffering=0)
                process = subprocess.Popen(
                    _render_command(job, attempt),
                    cwd=repo_root,
                    env=_child_environment(
                        manifest, stage=job.stage, run_id=job.job_id, gpu=gpu
                    ),
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                process_identity = _process_identity(process.pid)
                active[gpu] = ActiveJob(
                    job, process, handle, gpu, attempt, log_path, process_identity
                )
                status["jobs"][f"{job.stage}:{job.job_id}"] = {
                    "state": "running",
                    "gpu": gpu,
                    "pid": process.pid,
                    "process_identity": process_identity,
                    "log": str(log_path),
                    "attempt_dir": str(attempt),
                    "started_at": time.time(),
                }
                print(f"[launch {job.stage} gpu{gpu}] {job.job_id}", flush=True)
                atomic_json(root / "status.json", status)

            if stopping:
                _terminate_active(
                    active,
                    status=status,
                    root=root,
                    reason="campaign interrupted by signal",
                    manifest=manifest,
                    repo_root=repo_root,
                )
                status["stage"] = "interrupted"
                atomic_json(root / "status.json", status)
                raise KeyboardInterrupt

            finished = [gpu for gpu, item in active.items() if item.process.poll() is not None]
            if not finished:
                if active:
                    time.sleep(1.0)
                continue

            for gpu in finished:
                item = active.pop(gpu)
                exit_code = item.process.poll()
                if not item.handle.closed:
                    item.handle.close()
                key = f"{item.job.stage}:{item.job.job_id}"
                try:
                    _verify_campaign_inputs(root, manifest, repo_root)
                    valid, reason = _verify_job_receipt(
                        item.attempt_dir / "completion_receipt.json",
                        item.job,
                        manifest,
                    )
                    if exit_code != 0 or not valid:
                        raise RuntimeError(
                            f"child exit={exit_code}; receipt: {reason}"
                        )
                    _publish_attempt(root, item.job, item.attempt_dir, manifest)
                except BaseException as exc:
                    status["jobs"][key] = {
                        "state": "failed",
                        "gpu": gpu,
                        "pid": item.process.pid,
                        "process_identity": item.process_identity,
                        "exit_code": exit_code,
                        "reason": str(exc),
                        "attempt_dir": str(item.attempt_dir),
                        "log": str(item.log_path),
                    }
                    atomic_json(root / "status.json", status)
                    raise RuntimeError(
                        f"{item.job.stage} job failed: {item.job.job_id}: {exc}"
                    ) from exc
                status["jobs"][key] = {
                    "state": "complete",
                    "gpu": gpu,
                    "pid": item.process.pid,
                    "process_identity": item.process_identity,
                    "exit_code": exit_code,
                    "reason": "verified and atomically published",
                    "log": str(item.log_path),
                }
                print(
                    f"[complete {item.job.stage} gpu{gpu}] {item.job.job_id}",
                    flush=True,
                )
                available.append(gpu)
                available.sort()
                atomic_json(root / "status.json", status)
    except BaseException:
        if active:
            _terminate_active(
                active,
                status=status,
                root=root,
                reason="terminated because a peer or orchestrator failed",
                manifest=manifest,
                repo_root=repo_root,
            )
        raise
    finally:
        # A defensive cleanup path for exceptions raised while recording status.
        if active:
            _terminate_active(
                active,
                status=status,
                root=root,
                reason="terminated during orchestrator cleanup",
                manifest=manifest,
                repo_root=repo_root,
            )
        signal.signal(signal.SIGTERM, previous_term)
        signal.signal(signal.SIGINT, previous_int)


def _run_phase0(
    *,
    root: Path,
    repo_root: Path,
    protocol_path: Path,
    python: str,
    smoke: bool,
    manifest: dict[str, Any],
    status: dict[str, Any],
) -> None:
    output = root / "phase0"
    expected = manifest["receipt_expectations"]["phase0"]
    valid, reason = verify_completion_receipt(
        output / "completion_receipt.json",
        expected_job_id=expected["job_id"],
        expected_metadata=_receipt_metadata(manifest, "phase0", "phase0"),
    )
    if valid:
        gate = strict_json_load(output / "phase0_gate.json")
        if gate.get("passed") is not True:
            raise RuntimeError("published Phase-0 gate is not passed")
        status["phase0"] = {"state": "complete", "reason": "verified campaign receipt"}
        atomic_json(root / "status.json", status)
        return
    if output.exists():
        preserved = _preserve_invalid_output(root, "phase0", "phase0", output)
        status["phase0"] = {
            "state": "pending",
            "reason": f"invalid prior output preserved at {preserved}: {reason}",
        }

    _verify_campaign_inputs(root, manifest, repo_root)
    attempt = _unique_attempt_path(root, "phase0", "phase0")
    log_dir = root / "logs" / "phase0"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{attempt.name}.log"
    command = [
        python,
        "-m",
        "repro.sagodi_protocol.phase0",
        "--protocol",
        str(protocol_path),
        "--output-root",
        str(attempt),
        "--device",
        "cpu",
    ]
    if smoke:
        command.append("--smoke")
    handle = log_path.open("ab", buffering=0)
    process = subprocess.Popen(
        command,
        cwd=repo_root,
        env=_child_environment(manifest, stage="phase0", run_id="phase0"),
        stdout=handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    process_identity = _process_identity(process.pid)
    status["phase0"] = {
        "state": "running",
        "pid": process.pid,
        "process_identity": process_identity,
        "attempt_dir": str(attempt),
        "log": str(log_path),
    }
    atomic_json(root / "status.json", status)
    stop_signal: int | None = None

    def handle_phase0_signal(signum: int, _frame: Any) -> None:
        nonlocal stop_signal
        stop_signal = int(signum)

    previous_term = signal.signal(signal.SIGTERM, handle_phase0_signal)
    previous_int = signal.signal(signal.SIGINT, handle_phase0_signal)
    try:
        while process.poll() is None and stop_signal is None:
            time.sleep(0.2)
        if stop_signal is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
        exit_code = process.poll()
        _verify_campaign_inputs(root, manifest, repo_root)
        if stop_signal is not None:
            status["phase0"] = {
                "state": "terminated",
                "pid": process.pid,
                "process_identity": process_identity,
                "exit_code": exit_code,
                "reason": f"campaign interrupted by signal {stop_signal}",
                "attempt_dir": str(attempt),
                "log": str(log_path),
            }
            status["stage"] = "interrupted"
            atomic_json(root / "status.json", status)
            raise KeyboardInterrupt
    except BaseException:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
        # This is also the post-child integrity check for exceptional exits.
        _verify_campaign_inputs(root, manifest, repo_root)
        raise
    finally:
        handle.close()
        signal.signal(signal.SIGTERM, previous_term)
        signal.signal(signal.SIGINT, previous_int)
    valid, reason = verify_completion_receipt(
        attempt / "completion_receipt.json",
        expected_job_id=expected["job_id"],
        expected_metadata=_receipt_metadata(manifest, "phase0", "phase0"),
    )
    if exit_code != 0 or not valid:
        status["phase0"] = {
            "state": "failed",
            "pid": process.pid,
            "process_identity": process_identity,
            "exit_code": exit_code,
            "reason": reason,
            "attempt_dir": str(attempt),
            "log": str(log_path),
        }
        atomic_json(root / "status.json", status)
        raise RuntimeError(f"Phase-0 failed: exit={exit_code}, receipt={reason}")
    gate = strict_json_load(attempt / "phase0_gate.json")
    if gate.get("passed") is not True:
        raise RuntimeError("Phase-0 gate failed; refusing to launch Phase 1")
    phase0_job = Job("phase0", "phase0", output, (), expected["job_id"])
    _publish_attempt(root, phase0_job, attempt, manifest)
    status["phase0"] = {
        "state": "complete",
        "pid": process.pid,
        "process_identity": process_identity,
        "exit_code": exit_code,
        "reason": "verified gate and atomically published",
        "log": str(log_path),
    }
    atomic_json(root / "status.json", status)


def _compute_receipt_completion(
    root: Path, manifest: dict[str, Any]
) -> tuple[bool, dict[str, str], dict[str, str]]:
    """Validate Phase-0, training, and analysis outputs independently."""

    reasons: dict[str, str] = {}
    receipt_hashes: dict[str, str] = {}
    for key, expectation in manifest["receipt_expectations"].items():
        receipt = root / expectation["output"] / "completion_receipt.json"
        valid, reason = verify_campaign_output_receipt(root, manifest, key)
        reasons[key] = reason
        if valid:
            receipt_hashes[key] = sha256_file(receipt)
    gate_path = root / "phase0" / "phase0_gate.json"
    try:
        gate_passed = strict_json_load(gate_path).get("passed") is True
    except (OSError, ValueError, json.JSONDecodeError):
        gate_passed = False
    reasons["phase0_gate"] = "passed" if gate_passed else "missing, invalid, or failed"
    complete = gate_passed and len(receipt_hashes) == len(manifest["receipt_expectations"])
    return complete, reasons, receipt_hashes


def _aggregation_paths(root: Path, manifest: dict[str, Any]) -> tuple[Path, Path, Path]:
    specification = manifest.get("pilot_aggregation")
    if not isinstance(specification, dict):
        raise ValueError("campaign manifest lacks a pilot aggregation specification")
    if specification != aggregation_specification():
        raise ValueError("pilot aggregation specification mismatch")
    directory = root / str(specification["directory"])
    return (
        directory,
        directory / str(specification["summary"]),
        directory / str(specification["run_matrix_csv"]),
    )


def verify_pilot_aggregation(
    root: Path, manifest: dict[str, Any]
) -> tuple[bool, str, dict[str, str]]:
    """Verify aggregate bytes against a deterministic rebuild from claim gates."""

    if "pilot_aggregation" not in manifest:
        # Backward-compatible unit/legacy schema path.  New campaigns always
        # sign an aggregation specification into their scientific identity.
        return True, "not required by this legacy manifest", {}
    try:
        _, summary_path, matrix_path = _aggregation_paths(root, manifest)
        expected_summary, expected_matrix = expected_aggregation_bytes(root, manifest)
        observed_summary = summary_path.read_bytes()
        observed_matrix = matrix_path.read_bytes()
    except (OSError, ValueError, TypeError, KeyError) as exc:
        return False, f"pilot aggregation missing, invalid, or not rebuildable: {exc}", {}
    if observed_summary != expected_summary:
        return False, "pilot summary bytes differ from deterministic rebuild", {}
    if observed_matrix != expected_matrix:
        return False, "pilot run-matrix CSV bytes differ from deterministic rebuild", {}
    hashes = {
        str(summary_path.relative_to(root)): sha256_file(summary_path),
        str(matrix_path.relative_to(root)): sha256_file(matrix_path),
    }
    return True, "deterministically rebuilt summary and CSV match", hashes


def _write_pilot_aggregation(
    root: Path, manifest: dict[str, Any]
) -> dict[str, str]:
    """Atomically publish the pilot summary only after every receipt validates."""

    receipts_complete, reasons, _ = _compute_receipt_completion(root, manifest)
    if not receipts_complete:
        raise RuntimeError(
            "refusing to aggregate before every analysis receipt validates: "
            f"{reasons}"
        )
    destination, summary_path, matrix_path = _aggregation_paths(root, manifest)
    expected_summary, expected_matrix = expected_aggregation_bytes(root, manifest)
    if destination.exists():
        valid, _, hashes = verify_pilot_aggregation(root, manifest)
        if valid:
            return hashes
        _preserve_invalid_output(
            root, "campaign_setup", "pilot_aggregation", destination
        )
    attempt = _unique_attempt_path(
        root, "campaign_setup", "pilot_aggregation", label="aggregate"
    )
    atomic_bytes(attempt / summary_path.name, expected_summary)
    atomic_bytes(attempt / matrix_path.name, expected_matrix)
    os.replace(attempt, destination)
    valid, reason, hashes = verify_pilot_aggregation(root, manifest)
    if not valid:
        raise RuntimeError(f"published pilot aggregation failed verification: {reason}")
    return hashes


def compute_campaign_completion(
    root: Path, manifest: dict[str, Any]
) -> tuple[bool, dict[str, str], dict[str, Any]]:
    """Return receipt/gate/aggregation validity without trusting status or COMPLETE."""

    receipts_complete, reasons, receipt_hashes = _compute_receipt_completion(
        root, manifest
    )
    aggregation_valid = False
    aggregation_hashes: dict[str, str] = {}
    if receipts_complete:
        aggregation_valid, aggregation_reason, aggregation_hashes = (
            verify_pilot_aggregation(root, manifest)
        )
    else:
        aggregation_reason = "not checked until all output receipts and Phase-0 gate validate"
    reasons["pilot_aggregation"] = aggregation_reason
    hashes: dict[str, Any] = {
        "receipts": receipt_hashes,
        "pilot_aggregation": aggregation_hashes,
    }
    return receipts_complete and aggregation_valid, reasons, hashes


def _write_complete_marker(root: Path, manifest: dict[str, Any]) -> None:
    complete, reasons, hashes = compute_campaign_completion(root, manifest)
    if not complete:
        raise RuntimeError(f"refusing to write COMPLETE for an incomplete campaign: {reasons}")
    atomic_json(
        root / "COMPLETE",
        {
            "schema_version": 3,
            "campaign_id": manifest["campaign_id"],
            "scientific_identity": manifest["scientific_identity"],
            "protocol_fingerprint": manifest["protocol_canonical_fingerprint"],
            "receipt_sha256": hashes["receipts"],
            "pilot_aggregation_sha256": hashes["pilot_aggregation"],
            "completed_at": time.time(),
            "scope": "phase0_and_nonconfirmatory_phase1_ring_pilot_only",
        },
    )


@_single_orchestrator_locked
def run_campaign(
    *,
    protocol_path: Path,
    artifact_root: Path,
    python: str,
    gpus: Sequence[int],
    smoke: bool = False,
    dry_run: bool = False,
) -> Path:
    protocol_path = Path(protocol_path).resolve()
    artifact_root = Path(artifact_root).resolve()
    repo_root = Path(__file__).resolve().parents[2]
    python = _resolve_python(python)
    gpus = _validated_gpu_ids(gpus)
    protocol = load_protocol(protocol_path)
    git_state = _git_state(repo_root)
    if not smoke and git_state["worktree_dirty"]:
        raise RuntimeError(
            "full campaign requires a clean committed git worktree; commit the frozen code first"
        )
    environment = _environment_fingerprint(python, gpus)
    _prepare_root(artifact_root)
    evaluation_bank = _materialize_evaluation_bank(artifact_root, protocol)
    perturbation_bank = _materialize_perturbation_bank(artifact_root, protocol)
    source_hashes = _source_hashes(repo_root, Path(__file__).resolve().parent)
    runs = list(expand_phase1_runs(protocol))
    fingerprint = protocol_fingerprint(protocol)
    campaign_id = f"{protocol['freeze_id']}-{fingerprint[:12]}"
    expectations = _receipt_expectations(runs)
    scientific_payload = {
        "campaign_id": campaign_id,
        "protocol_file_sha256": sha256_file(protocol_path),
        "protocol_canonical_fingerprint": fingerprint,
        "source_protocol_sha256": protocol["source_protocol"]["sha256"],
        "source_hashes": source_hashes,
        "evaluation_bank": evaluation_bank,
        "perturbation_bank": perturbation_bank,
        "code": git_state,
        "environment": environment,
        "smoke": bool(smoke),
        "run_matrix": runs,
        "receipt_expectations": expectations,
        "pilot_aggregation": aggregation_specification(),
        "pilot_only": True,
    }
    manifest = {
        "schema_version": 2,
        **scientific_payload,
        "protocol_file": str(protocol_path),
        "python": str(Path(python).resolve()),
        "gpus": list(map(int, gpus)),
        "scientific_identity_payload": scientific_payload,
        "scientific_identity": canonical_hash(scientific_payload),
    }
    _write_or_check_manifest(artifact_root, manifest)
    _verify_campaign_inputs(artifact_root, manifest, repo_root)
    bank_path = artifact_root / evaluation_bank["path"]
    perturbation_bank_path = artifact_root / perturbation_bank["path"]
    training = _training_jobs(
        protocol,
        root=artifact_root,
        python=python,
        protocol_path=protocol_path,
        evaluation_bank=bank_path,
        campaign_identity=manifest["scientific_identity"],
        smoke=smoke,
    )
    analyses = _analysis_jobs(
        training,
        root=artifact_root,
        python=python,
        protocol_path=protocol_path,
        evaluation_bank=bank_path,
        perturbation_bank=perturbation_bank_path,
        campaign_identity=manifest["scientific_identity"],
        smoke=smoke,
    )
    if dry_run:
        print(
            json.dumps(
                {
                    "campaign_id": manifest["campaign_id"],
                    "scientific_identity": manifest["scientific_identity"],
                    "evaluation_bank": evaluation_bank,
                    "perturbation_bank": perturbation_bank,
                    "pilot_aggregation": manifest["pilot_aggregation"],
                    "phase0": str(artifact_root / "phase0"),
                    "training_jobs": [job.job_id for job in training],
                    "analysis_jobs": [job.job_id for job in analyses],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return artifact_root

    status: dict[str, Any] = {
        "schema_version": 2,
        "campaign_id": manifest["campaign_id"],
        "scientific_identity": manifest["scientific_identity"],
        "stage": "phase0",
        "phase0": {"state": "pending"},
        "jobs": {},
        "started_at": time.time(),
    }
    atomic_json(artifact_root / "status.json", status)
    _run_phase0(
        root=artifact_root,
        repo_root=repo_root,
        protocol_path=protocol_path,
        python=python,
        smoke=smoke,
        manifest=manifest,
        status=status,
    )

    status["stage"] = "training"
    atomic_json(artifact_root / "status.json", status)
    _run_jobs(
        training,
        gpus=gpus,
        root=artifact_root,
        repo_root=repo_root,
        manifest=manifest,
        status=status,
    )
    status["stage"] = "analysis"
    atomic_json(artifact_root / "status.json", status)
    _run_jobs(
        analyses,
        gpus=gpus,
        root=artifact_root,
        repo_root=repo_root,
        manifest=manifest,
        status=status,
    )
    _verify_campaign_inputs(artifact_root, manifest, repo_root)
    aggregation_hashes = _write_pilot_aggregation(artifact_root, manifest)
    status["pilot_aggregation"] = {
        "state": "complete",
        "sha256": aggregation_hashes,
        "inference": "descriptive_nonconfirmatory_no_p_values",
    }
    atomic_json(artifact_root / "status.json", status)
    _write_complete_marker(artifact_root, manifest)
    status["stage"] = "complete"
    status["completed_at"] = time.time()
    atomic_json(artifact_root / "status.json", status)
    return artifact_root


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gpus", default="0,1,2,3,4,5")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    gpus = tuple(int(item) for item in args.gpus.split(",") if item.strip())
    output = run_campaign(
        protocol_path=args.protocol,
        artifact_root=args.artifact_root,
        python=args.python,
        gpus=gpus,
        smoke=args.smoke,
        dry_run=args.dry_run,
    )
    print(
        json.dumps(
            {
                "status": "dry_run" if args.dry_run else "complete",
                "artifact_root": str(output),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
