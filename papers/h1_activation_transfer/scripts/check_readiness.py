"""Check whether the local workspace is ready for H1 clean reruns."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

_DEFAULT_CLEAN_EVAL = Path("data/tasks/multi_hop_reasoning/clean_eval.jsonl")
_DEFAULT_MANIFEST = Path("papers/h1_activation_transfer/data_audit/clean_eval_manifest.json")
_DEFAULT_CKPT_DIR = Path("results/phase1/checkpoints")
_MODEL_IDS = ("EleutherAI/pythia-160m", "EleutherAI/pythia-410m")
_SEEDS = (42, 123, 456)
_DIRECTIONS = ("fwd", "rev")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def jsonl_count(path: Path) -> int:
    with open(path, encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


def hf_cache_root() -> Path:
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]) / "hub"
    if os.environ.get("HUGGINGFACE_HUB_CACHE"):
        return Path(os.environ["HUGGINGFACE_HUB_CACHE"])
    return Path.home() / ".cache" / "huggingface" / "hub"


def model_cache_path(model_id: str) -> Path:
    return hf_cache_root() / f"models--{model_id.replace('/', '--')}"


def check_clean_eval(clean_eval: Path, manifest_path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {
        "path": str(clean_eval),
        "exists": clean_eval.exists(),
        "blocking": [],
        "warnings": [],
    }
    if not clean_eval.exists():
        out["blocking"].append("clean_eval_missing")
        return out

    out["rows"] = jsonl_count(clean_eval)
    out["sha256"] = sha256_file(clean_eval)

    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as fh:
            manifest = json.load(fh)
        out["manifest_path"] = str(manifest_path)
        out["manifest_hash"] = manifest.get("clean_eval_sha256")
        out["manifest_rows"] = manifest.get("clean_summary", {}).get("n_rows")
        out["manifest_acceptance_passed"] = manifest.get("acceptance", {}).get("passed")
        if out["sha256"] != out["manifest_hash"]:
            out["blocking"].append("clean_eval_hash_mismatch")
        if out["rows"] != out["manifest_rows"]:
            out["blocking"].append("clean_eval_row_count_mismatch")
        if out["manifest_acceptance_passed"] is not True:
            out["blocking"].append("clean_eval_manifest_not_accepted")
    else:
        out["warnings"].append("clean_eval_manifest_missing")

    return out


def check_checkpoints(ckpt_dir: Path) -> dict[str, Any]:
    missing = []
    present = []
    for direction in _DIRECTIONS:
        for seed in _SEEDS:
            path = ckpt_dir / f"m6_translation_{direction}_seed{seed}.pt"
            record = {"direction": direction, "seed": seed, "path": str(path)}
            if path.exists():
                record["bytes"] = path.stat().st_size
                present.append(record)
            else:
                missing.append(record)
    return {
        "directory": str(ckpt_dir),
        "exists": ckpt_dir.exists(),
        "present": present,
        "missing": missing,
        "blocking": ["m6_checkpoint_missing"] if missing else [],
        "warnings": [],
    }


def check_models(require_model_cache: bool) -> dict[str, Any]:
    entries = []
    missing = []
    for model_id in _MODEL_IDS:
        path = model_cache_path(model_id)
        record = {"model_id": model_id, "cache_path": str(path), "cache_exists": path.exists()}
        entries.append(record)
        if not path.exists():
            missing.append(record)
    blocking = ["hf_model_cache_missing"] if require_model_cache and missing else []
    warnings = ["hf_model_cache_missing"] if missing and not require_model_cache else []
    return {
        "cache_root": str(hf_cache_root()),
        "models": entries,
        "blocking": blocking,
        "warnings": warnings,
    }


def check_torch(require_cuda: bool) -> dict[str, Any]:
    out: dict[str, Any] = {"blocking": [], "warnings": []}
    try:
        import torch

        out["torch_version"] = torch.__version__
        out["cuda_available"] = bool(torch.cuda.is_available())
        out["cuda_device_count"] = int(torch.cuda.device_count())
    except Exception as exc:  # pragma: no cover - defensive environment check
        out["blocking"].append("torch_import_failed")
        out["error"] = repr(exc)
        return out

    if require_cuda and not out["cuda_available"]:
        out["blocking"].append("cuda_required_but_unavailable")
    elif not out["cuda_available"]:
        out["warnings"].append("cuda_unavailable_cpu_only")
    return out


def flatten_status(report: dict[str, Any]) -> tuple[list[str], list[str]]:
    blocking = []
    warnings = []
    for section in report.values():
        if isinstance(section, dict):
            blocking.extend(section.get("blocking", []))
            warnings.extend(section.get("warnings", []))
    return blocking, warnings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check H1 clean-rerun readiness.")
    parser.add_argument("--clean-eval", type=Path, default=_DEFAULT_CLEAN_EVAL)
    parser.add_argument("--manifest", type=Path, default=_DEFAULT_MANIFEST)
    parser.add_argument("--ckpt-dir", type=Path, default=_DEFAULT_CKPT_DIR)
    parser.add_argument("--require-model-cache", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON only.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = {
        "clean_eval": check_clean_eval(args.clean_eval, args.manifest),
        "checkpoints": check_checkpoints(args.ckpt_dir),
        "models": check_models(args.require_model_cache),
        "torch": check_torch(args.require_cuda),
    }
    blocking, warnings = flatten_status(report)
    report["summary"] = {
        "ready": not blocking,
        "blocking": blocking,
        "warnings": warnings,
    }

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"ready: {report['summary']['ready']}")
        if blocking:
            print("blocking: " + ", ".join(blocking))
        if warnings:
            print("warnings: " + ", ".join(warnings))
        print(f"clean_eval_rows: {report['clean_eval'].get('rows')}")
        print(f"clean_eval_sha256: {report['clean_eval'].get('sha256')}")
        print(f"checkpoints_present: {len(report['checkpoints']['present'])}/6")
        print(f"cuda_available: {report['torch'].get('cuda_available')}")

    if blocking:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
