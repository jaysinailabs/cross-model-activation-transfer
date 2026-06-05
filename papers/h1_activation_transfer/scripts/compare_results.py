"""Pairwise comparisons for H1 final result files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from summarize_results import sample_metrics

METRICS = (
    "word_boundary_contains",
    "legacy_contains",
    "normalized_exact_match",
)
BASELINES = ("no_inject", "nl_relay")


def stable_seed(*parts: Any) -> int:
    payload = "|".join(str(part) for part in parts)
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8], 16)


def load_results(
    results_dirs: list[Path],
    *,
    direction: str | None = "fwd",
    glob: str = "h1_*.json",
) -> list[dict[str, Any]]:
    results = []
    seen_paths = set()
    for results_dir in results_dirs:
        for path in sorted(results_dir.glob(glob)):
            if path.resolve() in seen_paths:
                continue
            seen_paths.add(path.resolve())
            with open(path, encoding="utf-8") as fh:
                payload = json.load(fh)
            if direction is not None and payload.get("direction") != direction:
                continue
            payload["_path"] = str(path)
            payload["_file"] = path.name
            results.append(payload)
    return results


def sample_id(row: dict[str, Any], fallback: int) -> str:
    return str(row.get("sample_id", f"idx:{fallback}"))


def metric_vector(result: dict[str, Any], metric: str) -> dict[str, bool]:
    out = {}
    for idx, row in enumerate(result["per_sample"]):
        prediction = str(row.get("prediction", row.get("predicted", row.get("generated", ""))))
        gold = str(row.get("gold", row.get("answer", "")))
        out[sample_id(row, idx)] = bool(sample_metrics(prediction, gold)[metric])
    return out


def bootstrap_ci(deltas: list[int], n_boot: int, seed: int) -> tuple[float, float, float]:
    """Percentile bootstrap CI over paired sample-level deltas.

    This is a simple percentile interval, not BCa.  It is intended for the
    paper tables' paired descriptive comparisons.
    """

    observed = mean(deltas) if deltas else 0.0
    if not deltas or n_boot <= 0:
        return observed, observed, observed
    rng = random.Random(seed)
    n = len(deltas)
    means = []
    for _ in range(n_boot):
        means.append(sum(deltas[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo = means[int(0.025 * (n_boot - 1))]
    hi = means[int(0.975 * (n_boot - 1))]
    return observed, lo, hi


def compare_vectors(
    condition_vec: dict[str, bool],
    baseline_vec: dict[str, bool],
    *,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    ids = sorted(set(condition_vec) & set(baseline_vec))
    deltas = [int(condition_vec[sid]) - int(baseline_vec[sid]) for sid in ids]
    delta, ci_lo, ci_hi = bootstrap_ci(deltas, n_boot=n_boot, seed=seed)
    gained = sum(1 for sid in ids if condition_vec[sid] and not baseline_vec[sid])
    lost = sum(1 for sid in ids if not condition_vec[sid] and baseline_vec[sid])
    tied_correct = sum(1 for sid in ids if condition_vec[sid] and baseline_vec[sid])
    tied_wrong = sum(1 for sid in ids if not condition_vec[sid] and not baseline_vec[sid])
    discordant = gained + lost
    if discordant:
        chi2 = (max(abs(gained - lost) - 1, 0) ** 2) / discordant
        p_mcnemar_approx = math.erfc(math.sqrt(chi2 / 2.0))
    else:
        p_mcnemar_approx = 1.0
    return {
        "n": len(ids),
        "delta": delta,
        "ci95_low": ci_lo,
        "ci95_high": ci_hi,
        "gained": gained,
        "lost": lost,
        "tied_correct": tied_correct,
        "tied_wrong": tied_wrong,
        "mcnemar_p_approx": p_mcnemar_approx,
        "_deltas": deltas,
    }


def run_comparisons(results: list[dict[str, Any]], *, n_boot: int) -> list[dict[str, Any]]:
    baseline_by_condition = {
        result["condition"]: result for result in results if result.get("condition") in BASELINES
    }
    rows = []
    for result in results:
        condition = result.get("condition")
        if condition in BASELINES:
            continue
        for baseline_name, baseline in baseline_by_condition.items():
            for metric in METRICS:
                comparison = compare_vectors(
                    metric_vector(result, metric),
                    metric_vector(baseline, metric),
                    n_boot=n_boot,
                    seed=stable_seed(condition, result.get("seed"), baseline_name, metric),
                )
                rows.append(
                    {
                        "condition": condition,
                        "seed": result.get("seed"),
                        "baseline": baseline_name,
                        "metric": metric,
                        "condition_file": result["_file"],
                        "baseline_file": baseline["_file"],
                        **comparison,
                    }
                )
    return rows


def group_rows(rows: list[dict[str, Any]], *, n_boot: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["condition"], row["baseline"], row["metric"])].append(row)
    out = []
    for (condition, baseline, metric), group in sorted(grouped.items()):
        pooled_deltas = [delta for row in group for delta in row.get("_deltas", [])]
        delta, ci_lo, ci_hi = bootstrap_ci(
            pooled_deltas,
            n_boot=n_boot,
            seed=stable_seed("pooled", condition, baseline, metric),
        )
        out.append(
            {
                "condition": condition,
                "baseline": baseline,
                "metric": metric,
                "runs": len(group),
                "seeds": sorted(
                    (row["seed"] for row in group),
                    key=lambda seed: -1 if seed is None else int(seed),
                ),
                "n_per_run": group[0]["n"] if group else 0,
                "n_pooled": len(pooled_deltas),
                "delta_mean": delta,
                "ci95_low_pooled": ci_lo,
                "ci95_high_pooled": ci_hi,
                "ci95_method": "pooled_percentile_bootstrap",
                "gained_mean": mean(row["gained"] for row in group),
                "lost_mean": mean(row["lost"] for row in group),
                "gained_total": sum(row["gained"] for row in group),
                "lost_total": sum(row["lost"] for row in group),
                "mcnemar_p_approx_min": min(row["mcnemar_p_approx"] for row in group),
            }
        )
    return out


def public_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{key: value for key, value in row.items() if not key.startswith("_")} for row in rows]


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
    lines = [
        "# H1 Pairwise Comparisons",
        "",
        "Grouped confidence intervals pool paired sample-level deltas across runs and use a percentile bootstrap interval; they are not BCa intervals.",
        "",
        "| Condition | Baseline | Metric | Runs | Delta | 95% CI | Gained | Lost |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        if row["metric"] not in ("word_boundary_contains", "legacy_contains"):
            continue
        lines.append(
            "| {condition} | {baseline} | {metric} | {runs} | {delta:.4f} | "
            "[{lo:.4f}, {hi:.4f}] | {gained:.1f} | {lost:.1f} |".format(
                condition=row["condition"],
                baseline=row["baseline"],
                metric=row["metric"],
                runs=row["runs"],
                delta=row["delta_mean"],
                lo=row["ci95_low_pooled"],
                hi=row["ci95_high_pooled"],
                gained=row["gained_mean"],
                lost=row["lost_mean"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pairwise H1 result comparisons.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        nargs="+",
        default=[Path("papers/h1_activation_transfer/results/final")],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("papers/h1_activation_transfer/results"),
    )
    parser.add_argument("--bootstrap", type=int, default=5000)
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
    rows = run_comparisons(
        load_results(args.results_dir, direction=direction, glob=args.glob),
        n_boot=args.bootstrap,
    )
    grouped = group_rows(rows, n_boot=args.bootstrap)
    public_per_run = public_rows(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / "final_pairwise_comparisons.json", "w", encoding="utf-8") as fh:
        json.dump(
            {"per_run": public_per_run, "by_condition": grouped},
            fh,
            ensure_ascii=False,
            indent=2,
        )
    write_csv(args.output_dir / "final_pairwise_comparisons_per_run.csv", public_per_run)
    write_csv(args.output_dir / "final_pairwise_comparisons_by_condition.csv", grouped)
    write_markdown(args.output_dir / "final_pairwise_comparisons.md", grouped)
    print(f"Wrote {len(rows)} per-run comparisons into {args.output_dir}")


if __name__ == "__main__":
    main()
