"""Summarize H1 final-run JSON files into paper-facing metrics.

The final runner stores legacy contains/exact scores for continuity with the
historical experiments.  This script recomputes the paper metrics from
per-sample predictions so result tables do not depend on hand-copied numbers.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import string
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable

_PUNCT_TRANS = str.maketrans("", "", string.punctuation)


def normalize_text(text: Any) -> str:
    """Lowercase, remove common punctuation, and collapse whitespace."""
    normalized = str(text).strip().lower().translate(_PUNCT_TRANS)
    return re.sub(r"\s+", " ", normalized).strip()


def exact_match(prediction: str, gold: str) -> bool:
    return prediction.strip().lower() == gold.strip().lower()


def normalized_exact_match(prediction: str, gold: str) -> bool:
    return normalize_text(prediction) == normalize_text(gold)


def word_boundary_contains(prediction: str, gold: str) -> bool:
    norm_pred = normalize_text(prediction)
    norm_gold = normalize_text(gold)
    if not norm_gold:
        return False
    pattern = rf"(?<!\w){re.escape(norm_gold)}(?!\w)"
    return re.search(pattern, norm_pred) is not None


def legacy_contains(prediction: str, gold: str) -> bool:
    return gold.strip().lower() in prediction.lower()


def first_answer_span(prediction: str) -> str:
    """Extract a simple first-answer span from raw LM output."""
    first_line = str(prediction).strip().splitlines()[0] if str(prediction).strip() else ""
    first_sentence = re.split(r"[.!?]", first_line, maxsplit=1)[0]
    span = first_sentence.strip()
    prefixes = (
        "the answer is",
        "answer is",
        "answer:",
        "a:",
        "it is",
        "it's",
    )
    low = span.lower()
    for prefix in prefixes:
        if low.startswith(prefix):
            return span[len(prefix) :].strip(" :-")
    return span


def first_answer_span_match(prediction: str, gold: str) -> bool:
    return normalize_text(first_answer_span(prediction)) == normalize_text(gold)


def sample_metrics(prediction: str, gold: str) -> dict[str, bool]:
    return {
        "exact_match": exact_match(prediction, gold),
        "normalized_exact_match": normalized_exact_match(prediction, gold),
        "word_boundary_contains": word_boundary_contains(prediction, gold),
        "first_answer_span_match": first_answer_span_match(prediction, gold),
        "legacy_contains": legacy_contains(prediction, gold),
    }


def _get_prediction(sample: dict[str, Any]) -> str:
    for key in ("prediction", "predicted", "generated"):
        if key in sample:
            return str(sample[key])
    return ""


def _get_gold(sample: dict[str, Any]) -> str:
    for key in ("gold", "answer"):
        if key in sample:
            return str(sample[key])
    source = sample.get("source")
    if isinstance(source, dict) and "answer" in source:
        return str(source["answer"])
    return ""


def load_result_files(paths: Iterable[Path], *, direction: str | None = "fwd") -> list[dict[str, Any]]:
    results = []
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
        per_sample = payload.get("per_sample")
        if not isinstance(per_sample, list):
            continue
        if direction is not None and payload.get("direction") != direction:
            continue
        payload["_path"] = str(path)
        results.append(payload)
    return results


def summarize_result(result: dict[str, Any]) -> dict[str, Any]:
    per_sample = result["per_sample"]
    n = len(per_sample)
    metric_counts = defaultdict(int)
    for row in per_sample:
        metrics = sample_metrics(_get_prediction(row), _get_gold(row))
        for key, value in metrics.items():
            metric_counts[key] += int(value)
    rates = {f"{key}_acc": (metric_counts[key] / n if n else 0.0) for key in metric_counts}
    return {
        "path": result["_path"],
        "condition": result.get("condition"),
        "direction": result.get("direction"),
        "seed": result.get("seed"),
        "deterministic": bool(result.get("deterministic", False)),
        "n": n,
        "clean_eval_hash": result.get("config", {}).get("clean_eval_hash"),
        "checkpoint": result.get("checkpoint"),
        **rates,
    }


def summarize_groups(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["condition"]), str(row["direction"]))].append(row)

    metric_keys = [
        "exact_match_acc",
        "normalized_exact_match_acc",
        "word_boundary_contains_acc",
        "first_answer_span_match_acc",
        "legacy_contains_acc",
    ]
    out = []
    for (condition, direction), group_rows in sorted(groups.items()):
        record: dict[str, Any] = {
            "condition": condition,
            "direction": direction,
            "runs": len(group_rows),
            "seeds": [r["seed"] for r in group_rows],
            "n": group_rows[0]["n"] if group_rows else 0,
        }
        for key in metric_keys:
            vals = [float(r.get(key, 0.0)) for r in group_rows]
            record[f"{key}_mean"] = mean(vals) if vals else 0.0
            record[f"{key}_std"] = stdev(vals) if len(vals) > 1 else 0.0
        out.append(record)
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with open(path, "w", newline="", encoding="utf-8") as fh:
            fh.write("")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, grouped_rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# H1 Final Result Summary",
        "",
        "| Condition | Direction | Runs | N | Normalized EM | Boundary Contains | Legacy Contains |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in grouped_rows:
        lines.append(
            "| {condition} | {direction} | {runs} | {n} | {nem:.4f} +/- {nem_std:.4f} | "
            "{wbc:.4f} +/- {wbc_std:.4f} | {legacy:.4f} +/- {legacy_std:.4f} |".format(
                condition=row["condition"],
                direction=row["direction"],
                runs=row["runs"],
                n=row["n"],
                nem=row["normalized_exact_match_acc_mean"],
                nem_std=row["normalized_exact_match_acc_std"],
                wbc=row["word_boundary_contains_acc_mean"],
                wbc_std=row["word_boundary_contains_acc_std"],
                legacy=row["legacy_contains_acc_mean"],
                legacy_std=row["legacy_contains_acc_std"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize H1 final result JSON files.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("papers/h1_activation_transfer/results/final"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("papers/h1_activation_transfer/results"),
    )
    parser.add_argument("--glob", default="h1_*.json")
    parser.add_argument(
        "--direction",
        default="fwd",
        help="Result direction to include; use 'all' to disable filtering.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    direction = None if args.direction == "all" else args.direction
    results = load_result_files(sorted(args.results_dir.glob(args.glob)), direction=direction)
    per_run_rows = [summarize_result(result) for result in results]
    grouped_rows = summarize_groups(per_run_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / "final_summary.json", "w", encoding="utf-8") as fh:
        json.dump(
            {"per_run": per_run_rows, "by_condition": grouped_rows},
            fh,
            ensure_ascii=False,
            indent=2,
        )
    write_csv(args.output_dir / "final_summary_per_run.csv", per_run_rows)
    write_csv(args.output_dir / "final_summary_by_condition.csv", grouped_rows)
    write_markdown(args.output_dir / "final_summary.md", grouped_rows)
    print(f"Summarized {len(per_run_rows)} result files into {args.output_dir}")


if __name__ == "__main__":
    main()
