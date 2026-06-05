"""Build a clean multi-hop evaluation set for the H1 paper.

The script removes exact train/eval overlap and exact duplicate evaluation rows.
It preserves all original fields and adds provenance metadata so paper reruns can
trace every clean sample back to the source evaluation file.

Default inputs are Project Rosetta's current multi-hop files:

    data/tasks/multi_hop_reasoning/train.jsonl
    data/tasks/multi_hop_reasoning/test_enhanced.jsonl

Default outputs:

    data/tasks/multi_hop_reasoning/clean_eval.jsonl
    papers/h1_activation_transfer/data_audit/leakage_report.json
    papers/h1_activation_transfer/data_audit/clean_eval_manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_TRAIN = Path("data/tasks/multi_hop_reasoning/train.jsonl")
DEFAULT_EVAL = Path("data/tasks/multi_hop_reasoning/test_enhanced.jsonl")
DEFAULT_OUT = Path("data/tasks/multi_hop_reasoning/clean_eval.jsonl")
DEFAULT_REPORT = Path("papers/h1_activation_transfer/data_audit/leakage_report.json")
DEFAULT_MANIFEST = Path("papers/h1_activation_transfer/data_audit/clean_eval_manifest.json")

CLEAN_EVAL_VERSION = "h1-clean-eval-v1"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} line {line_no}: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def key_payload(row: dict[str, Any]) -> dict[str, str]:
    missing = [k for k in ("context", "question", "answer") if k not in row]
    if missing:
        raise KeyError(f"Sample missing required keys {missing}: {row}")
    return {
        "context": str(row["context"]),
        "question": str(row["question"]),
        "answer": str(row["answer"]),
    }


def sample_key(row: dict[str, Any]) -> str:
    return json.dumps(key_payload(row), ensure_ascii=False, sort_keys=True)


def key_sha256(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def infer_source_component(row: dict[str, Any]) -> str:
    batch = row.get("batch")
    if batch == "v1":
        return "test.jsonl"
    if batch == "v2":
        return "test_v2.jsonl"
    return "test_enhanced.jsonl"


def infer_answer_type(answer: Any) -> str:
    text = str(answer).strip()
    if text.isdigit():
        return "number"
    if len(text.split()) > 1:
        return "phrase"
    return "entity"


def infer_domain(row: dict[str, Any]) -> str:
    """Best-effort domain label for stratified analysis.

    The historical v2 data did not store domain explicitly. These labels are
    deterministic heuristics for analysis, not ground-truth annotations.
    """
    if row.get("batch") == "v1":
        return "geography"

    text = f"{row.get('context', '')} {row.get('question', '')}".lower()
    if any(term in text for term in ("framework", "programming", "backend", "platform", "storefront")):
        return "technology_heuristic"
    if any(term in text for term in ("during", "century", "revolution", "ancient", "war")):
        return "history_heuristic"
    if any(term in text for term in ("metal", "element", "combustion", "conductive", "semiconductor", "industry")):
        return "science_heuristic"
    return "unspecified"


def stable_provenance_path(path: Path) -> str:
    """Return the frozen clean-eval provenance path format.

    The original clean-eval hash was frozen from a Windows run where
    `Path.__str__` used backslashes. Keep that byte-level representation stable
    across operating systems so CI can reproduce the published hash.
    """

    return str(path).replace("/", "\\")


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    hop_counts = Counter(str(row.get("hops", "missing")) for row in rows)
    batch_counts = Counter(str(row.get("batch", "missing")) for row in rows)
    domain_counts = Counter(
        str(row.get("domain_inferred", row.get("domain", infer_domain(row)))) for row in rows
    )
    answer_type_counts = Counter(
        str(row.get("answer_type", infer_answer_type(row.get("answer", "")))) for row in rows
    )
    answer_counts = Counter(str(row.get("answer", "missing")) for row in rows)

    return {
        "n_rows": len(rows),
        "hop_distribution": dict(sorted(hop_counts.items())),
        "batch_distribution": dict(sorted(batch_counts.items())),
        "domain_inferred_distribution": dict(sorted(domain_counts.items())),
        "answer_type_distribution": dict(sorted(answer_type_counts.items())),
        "n_unique_answers": len(answer_counts),
        "top_answers": [
            {"answer": answer, "count": count}
            for answer, count in answer_counts.most_common(20)
        ],
    }


def build_clean_eval(
    train_path: Path,
    eval_path: Path,
    out_path: Path,
    report_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    train_rows = read_jsonl(train_path)
    eval_rows = read_jsonl(eval_path)

    train_key_to_indices: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(train_rows):
        train_key_to_indices[sample_key(row)].append(idx)
    train_keys = set(train_key_to_indices)
    seen_eval_keys: set[str] = set()
    eval_key_to_indices: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(eval_rows):
        eval_key_to_indices[sample_key(row)].append(idx)

    clean_rows: list[dict[str, Any]] = []
    removed_overlap: list[dict[str, Any]] = []
    removed_duplicates: list[dict[str, Any]] = []

    for idx, row in enumerate(eval_rows):
        key = sample_key(row)
        if key in train_keys:
            removed_overlap.append(
                {
                    "source_eval_index": idx,
                    "reason": "exact_train_eval_overlap",
                    "dedup_key_sha256": key_sha256(key),
                    "overlap_train_indices": train_key_to_indices[key],
                    "context": row.get("context"),
                    "question": row.get("question"),
                    "answer": row.get("answer"),
                    "batch": row.get("batch"),
                    "hops": row.get("hops"),
                }
            )
            continue

        if key in seen_eval_keys:
            removed_duplicates.append(
                {
                    "source_eval_index": idx,
                    "reason": "exact_eval_duplicate",
                    "first_eval_index": eval_key_to_indices[key][0],
                    "dedup_key_sha256": key_sha256(key),
                    "context": row.get("context"),
                    "question": row.get("question"),
                    "answer": row.get("answer"),
                    "batch": row.get("batch"),
                    "hops": row.get("hops"),
                }
            )
            continue

        seen_eval_keys.add(key)
        clean_row = dict(row)
        clean_row["clean_eval_id"] = f"mh_clean_{len(clean_rows):06d}"
        clean_row["source_eval_file"] = stable_provenance_path(eval_path)
        clean_row["source_component_file"] = infer_source_component(row)
        clean_row["source_eval_index"] = idx
        clean_row["clean_eval_version"] = CLEAN_EVAL_VERSION
        clean_row["dedup_key_sha256"] = key_sha256(key)
        clean_row.setdefault("answer_type", infer_answer_type(row.get("answer", "")))
        clean_row.setdefault("domain_inferred", infer_domain(row))
        clean_rows.append(clean_row)

    write_jsonl(out_path, clean_rows)

    clean_keys = [sample_key(row) for row in clean_rows]
    clean_key_counts = Counter(clean_keys)
    remaining_duplicate_keys = [key for key, count in clean_key_counts.items() if count > 1]
    remaining_overlap = [key for key in clean_keys if key in train_keys]

    report = {
        "clean_eval_version": CLEAN_EVAL_VERSION,
        "train_path": str(train_path),
        "source_eval_path": str(eval_path),
        "clean_eval_path": str(out_path),
        "train_rows": len(train_rows),
        "source_eval_rows": len(eval_rows),
        "clean_eval_rows": len(clean_rows),
        "removed_overlap_rows": len(removed_overlap),
        "removed_duplicate_rows": len(removed_duplicates),
        "remaining_overlap_rows": len(remaining_overlap),
        "remaining_duplicate_keys": len(remaining_duplicate_keys),
        "removed_overlap_examples": removed_overlap[:50],
        "removed_duplicate_examples": removed_duplicates[:50],
    }
    write_json(report_path, report)

    manifest = {
        "clean_eval_version": CLEAN_EVAL_VERSION,
        "train_path": str(train_path),
        "source_eval_path": str(eval_path),
        "clean_eval_path": str(out_path),
        "train_sha256": file_sha256(train_path),
        "source_eval_sha256": file_sha256(eval_path),
        "clean_eval_sha256": file_sha256(out_path),
        "source_summary": summarize_rows(eval_rows),
        "clean_summary": summarize_rows(clean_rows),
        "removed_overlap_rows": len(removed_overlap),
        "removed_duplicate_rows": len(removed_duplicates),
        "acceptance": {
            "remaining_overlap_rows": len(remaining_overlap),
            "remaining_duplicate_keys": len(remaining_duplicate_keys),
            "passed": len(remaining_overlap) == 0 and len(remaining_duplicate_keys) == 0,
        },
    }
    write_json(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build H1 clean multi-hop eval set.")
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--eval", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()

    manifest = build_clean_eval(
        train_path=args.train,
        eval_path=args.eval,
        out_path=args.out,
        report_path=args.report,
        manifest_path=args.manifest,
    )
    acceptance = manifest["acceptance"]
    print(f"clean_eval_rows={manifest['clean_summary']['n_rows']}")
    print(f"removed_overlap_rows={manifest['removed_overlap_rows']}")
    print(f"removed_duplicate_rows={manifest['removed_duplicate_rows']}")
    print(f"passed={acceptance['passed']}")
    if not acceptance["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
