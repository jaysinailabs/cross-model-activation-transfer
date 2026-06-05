"""Build a manual-audit table for metric disagreements in H1 final results."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from summarize_results import sample_metrics

MAIN_CONDITIONS = {
    "no_inject",
    "nl_relay",
    "additive",
    "replace",
    "scale_corrected",
    "best_alpha",
}


def load_results(results_dir: Path) -> list[dict[str, Any]]:
    results = []
    for path in sorted(results_dir.glob("h1_*.json")):
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
        payload["_file"] = path.name
        results.append(payload)
    return results


def excerpt(text: str, max_chars: int = 220) -> str:
    compact = " ".join(str(text).split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3] + "..."


def audit_label(metrics: dict[str, bool]) -> tuple[str, str]:
    if metrics["first_answer_span_match"]:
        return "verbose_correct", "first answer span matches the gold answer"
    if not metrics["word_boundary_contains"]:
        return "substring_noise", "legacy hit is not a word-boundary match"
    return "accidental_mention", "gold appears later as a word-boundary mention"


def collect_disagreements(
    results: list[dict[str, Any]], per_result_file: int
) -> list[dict[str, Any]]:
    rows = []
    for result in results:
        condition = result.get("condition")
        if condition not in MAIN_CONDITIONS:
            continue
        selected = 0
        for sample in result.get("per_sample", []):
            prediction = str(sample.get("prediction", ""))
            gold = str(sample.get("gold", ""))
            metrics = sample_metrics(prediction, gold)
            if not metrics["legacy_contains"] or metrics["normalized_exact_match"]:
                continue
            label, notes = audit_label(metrics)
            rows.append(
                {
                    "condition": condition,
                    "seed": result.get("seed"),
                    "file": result["_file"],
                    "idx": sample.get("idx"),
                    "sample_id": sample.get("sample_id"),
                    "gold": gold,
                    "word_boundary_contains": metrics["word_boundary_contains"],
                    "first_answer_span_match": metrics["first_answer_span_match"],
                    "prediction_excerpt": excerpt(prediction),
                    "audit_label": label,
                    "audit_notes": notes,
                }
            )
            selected += 1
            if selected >= per_result_file:
                break
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    label_counts = Counter(row["audit_label"] for row in rows)
    lines = [
        "# H1 Metric Disagreement Audit",
        "",
        "Rows where `legacy_contains=true` and `normalized_exact_match=false`.",
        "",
        "Labels are rule-based from the stricter helper metrics: `verbose_correct`, `substring_noise`, or `accidental_mention`.",
        "",
        "## Label Counts",
        "",
        "| Label | Count |",
        "| --- | ---: |",
    ]
    for label, count in sorted(label_counts.items()):
        lines.append(f"| {label} | {count} |")
    lines.extend(
        [
            "",
            "## Rows",
            "",
            "| Condition | Seed | Sample | Gold | Boundary | First Span | Prediction Excerpt | Label |",
            "| --- | ---: | --- | --- | ---: | ---: | --- | --- |",
        ]
    )
    for row in rows:
        lines.append(
            "| {condition} | {seed} | {sample_id} | {gold} | {boundary} | {span} | {pred} | {label} |".format(
                condition=row["condition"],
                seed="" if row["seed"] is None else row["seed"],
                sample_id=row["sample_id"],
                gold=str(row["gold"]).replace("|", "\\|"),
                boundary=row["word_boundary_contains"],
                span=row["first_answer_span_match"],
                pred=row["prediction_excerpt"].replace("|", "\\|"),
                label=row["audit_label"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build metric disagreement audit table.")
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
    parser.add_argument("--per-result-file", type=int, default=5)
    parser.add_argument("--per-condition", type=int, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    per_result_file = args.per_condition if args.per_condition is not None else args.per_result_file
    rows = collect_disagreements(load_results(args.results_dir), per_result_file)
    write_csv(args.output_dir / "metric_disagreement_audit.csv", rows)
    write_markdown(args.output_dir / "metric_disagreement_audit.md", rows)
    print(f"Wrote {len(rows)} metric-disagreement audit rows.")


if __name__ == "__main__":
    main()
