"""TOST equivalence analysis for H1 final results.

Upgrades the descriptive paired bootstrap comparisons into explicit
equivalence evidence: the low-strength additive condition is tested against
the no-injection and NL-relay baselines with two one-sided tests (TOST)
against practical margins pre-specified for this reanalysis (the margins were
chosen at analysis time, after the frozen rerun, and are not pre-registered).
The stronger-transfer variants are tested for one-sided degradation, and the
full-causal-weight conditions are additionally compared pairwise (replacement
vs. random vs. zero controls) to test practical equivalence.  Only
the locked final result files are consumed; no new model runs are required.

Three views are reported per condition/baseline/metric:
- per_run: each seed separately (within-run samples are independent).
- clustered: per-sample deltas averaged across runs (one value per prompt),
  which removes the cross-seed correlation induced by the shared eval set.
- pooled: all run-level deltas concatenated (descriptive only; the shared
  eval set makes its standard error optimistic, so no TOST verdict is drawn
  from this view).
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from compare_results import load_results, metric_vector, stable_seed

METRICS = ("word_boundary_contains", "legacy_contains")
BASELINES = ("no_inject", "nl_relay")
EQUIVALENCE_CONDITION = "additive"
DEGRADATION_CONDITIONS = (
    "best_alpha",
    "scale_corrected",
    "replace",
    "same_norm_random",
    "zero_replacement",
    "b_to_b_self_inject",
)
PAIRWISE_FULL_WEIGHT = (
    ("replace", "same_norm_random"),
    ("replace", "zero_replacement"),
    ("same_norm_random", "zero_replacement"),
)


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def paired_delta_map(
    condition_vec: dict[str, bool], baseline_vec: dict[str, bool]
) -> dict[str, int]:
    ids = sorted(set(condition_vec) & set(baseline_vec))
    return {sid: int(condition_vec[sid]) - int(baseline_vec[sid]) for sid in ids}


def tost_summary(deltas: list[float], margins: tuple[float, ...]) -> dict[str, Any]:
    """Mean, normal-approximation 90% CI, and TOST p-values per margin."""

    n = len(deltas)
    m = mean(deltas) if deltas else 0.0
    sd = stdev(deltas) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n else float("inf")
    out: dict[str, Any] = {
        "n": n,
        "mean_delta": m,
        "se": se,
        "ci90_low": m - 1.645 * se,
        "ci90_high": m + 1.645 * se,
        "one_sided_p_worse": normal_cdf(m / se) if se > 0 else (0.0 if m < 0 else 1.0),
    }
    for margin in margins:
        key = f"{margin:.2f}".replace(".", "p")
        if se == 0.0:
            p_tost = 0.0 if abs(m) < margin else 1.0
        else:
            p_lower = 1.0 - normal_cdf((m + margin) / se)
            p_upper = 1.0 - normal_cdf((margin - m) / se)
            p_tost = max(p_lower, p_upper)
        out[f"tost_p_margin_{key}"] = p_tost
        out[f"equivalent_margin_{key}"] = p_tost < 0.05
    return out


def bootstrap_ci90(deltas: list[float], *, n_boot: int, seed: int) -> tuple[float, float]:
    if not deltas or n_boot <= 0:
        value = mean(deltas) if deltas else 0.0
        return value, value
    rng = random.Random(seed)
    n = len(deltas)
    means = sorted(
        sum(deltas[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot)
    )
    return means[int(0.05 * (n_boot - 1))], means[int(0.95 * (n_boot - 1))]


def analyze_condition(
    runs: list[dict[str, Any]],
    baseline: dict[str, Any],
    metric: str,
    *,
    margins: tuple[float, ...],
    n_boot: int,
) -> dict[str, Any]:
    baseline_vec = metric_vector(baseline, metric)
    per_run = []
    delta_maps = []
    pooled: list[float] = []
    for run in sorted(runs, key=lambda r: (r.get("seed") is None, r.get("seed"))):
        delta_map = paired_delta_map(metric_vector(run, metric), baseline_vec)
        delta_maps.append(delta_map)
        deltas = [float(v) for v in delta_map.values()]
        pooled.extend(deltas)
        per_run.append({"seed": run.get("seed"), **tost_summary(deltas, margins)})

    sample_ids = sorted(set.intersection(*(set(m) for m in delta_maps)))
    clustered = [
        mean(float(delta_map[sid]) for delta_map in delta_maps) for sid in sample_ids
    ]
    boot_lo, boot_hi = bootstrap_ci90(
        clustered,
        n_boot=n_boot,
        seed=stable_seed("equivalence", metric, baseline.get("condition")),
    )
    return {
        "per_run": per_run,
        "clustered": {
            **tost_summary(clustered, margins),
            "ci90_bootstrap_low": boot_lo,
            "ci90_bootstrap_high": boot_hi,
        },
        "pooled": {
            "n": len(pooled),
            "mean_delta": mean(pooled) if pooled else 0.0,
            "note": "descriptive only; runs share the eval set",
        },
    }


def pair_runs(
    runs_a: list[dict[str, Any]], runs_b: list[dict[str, Any]]
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Pair runs by seed; a single deterministic run pairs with every seed."""

    if len(runs_b) == 1:
        return [(run_a, runs_b[0]) for run_a in runs_a]
    if len(runs_a) == 1:
        return [(runs_a[0], run_b) for run_b in runs_b]
    by_seed = {run.get("seed"): run for run in runs_b}
    return [
        (run_a, by_seed[run_a.get("seed")])
        for run_a in runs_a
        if run_a.get("seed") in by_seed
    ]


def analyze_pairwise(
    runs_a: list[dict[str, Any]],
    runs_b: list[dict[str, Any]],
    metric: str,
    *,
    margins: tuple[float, ...],
) -> dict[str, Any]:
    delta_maps = []
    for run_a, run_b in pair_runs(runs_a, runs_b):
        delta_maps.append(
            paired_delta_map(metric_vector(run_a, metric), metric_vector(run_b, metric))
        )
    sample_ids = sorted(set.intersection(*(set(m) for m in delta_maps)))
    clustered = [
        mean(float(delta_map[sid]) for delta_map in delta_maps) for sid in sample_ids
    ]
    return {"clustered": tost_summary(clustered, margins)}


def build_report(
    results: list[dict[str, Any]], *, margins: tuple[float, ...], n_boot: int
) -> dict[str, Any]:
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        by_condition[str(result.get("condition"))].append(result)

    report: dict[str, Any] = {
        "margins": list(margins),
        "alpha": 0.05,
        "metrics": list(METRICS),
        "baseline_rates": {},
        "equivalence": {},
        "degradation": {},
    }
    for baseline_name in BASELINES:
        baseline = by_condition[baseline_name][0]
        for metric in METRICS:
            vec = metric_vector(baseline, metric)
            report["baseline_rates"].setdefault(baseline_name, {})[metric] = (
                sum(vec.values()) / len(vec)
            )

    for baseline_name in BASELINES:
        baseline = by_condition[baseline_name][0]
        report["equivalence"][baseline_name] = {
            metric: analyze_condition(
                by_condition[EQUIVALENCE_CONDITION],
                baseline,
                metric,
                margins=margins,
                n_boot=n_boot,
            )
            for metric in METRICS
        }

    no_inject = by_condition["no_inject"][0]
    for condition in DEGRADATION_CONDITIONS:
        if condition not in by_condition:
            continue
        report["degradation"][condition] = {
            metric: analyze_condition(
                by_condition[condition],
                no_inject,
                metric,
                margins=margins,
                n_boot=n_boot,
            )
            for metric in METRICS
        }

    report["pairwise_full_weight"] = {}
    for cond_a, cond_b in PAIRWISE_FULL_WEIGHT:
        if cond_a not in by_condition or cond_b not in by_condition:
            continue
        report["pairwise_full_weight"][f"{cond_a}_vs_{cond_b}"] = {
            metric: analyze_pairwise(
                by_condition[cond_a],
                by_condition[cond_b],
                metric,
                margins=margins,
            )
            for metric in METRICS
        }
    return report


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    margins = report["margins"]
    lines = [
        "# H1 Equivalence Analysis (TOST)",
        "",
        "Primary question: is the low-strength additive injection statistically",
        "equivalent to the baselines within a practical margin?  Margins were",
        "pre-specified for this reanalysis (chosen at analysis time, after the",
        "frozen rerun; they are not pre-registered).  The clustered view",
        "averages each sample's paired delta across runs, so the shared eval",
        "set does not understate the standard error.  TOST verdicts use",
        "alpha=0.05.",
        "",
    ]
    for baseline_name, metrics in report["equivalence"].items():
        lines.append(f"## additive vs {baseline_name}")
        lines.append("")
        header_margins = " | ".join(
            f"TOST p (margin {margin:.2f})" for margin in margins
        )
        lines.append(f"| Metric | View | N | Delta | 90% CI | {header_margins} |")
        lines.append(
            "| --- | --- | ---: | ---: | ---: | " + " | ".join("---:" for _ in margins) + " |"
        )
        for metric, block in metrics.items():
            rows = [(f"seed {r['seed']}", r) for r in block["per_run"]]
            rows.append(("clustered", block["clustered"]))
            for label, row in rows:
                cells = " | ".join(
                    "{:.2e}".format(row[f"tost_p_margin_{f'{m:.2f}'.replace('.', 'p')}"])
                    for m in margins
                )
                lines.append(
                    "| {metric} | {label} | {n} | {delta:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {cells} |".format(
                        metric=metric,
                        label=label,
                        n=row["n"],
                        delta=row["mean_delta"],
                        lo=row["ci90_low"],
                        hi=row["ci90_high"],
                        cells=cells,
                    )
                )
        lines.append("")

    lines.append("## Stronger-transfer variants vs no_inject (one-sided degradation)")
    lines.append("")
    lines.append("| Condition | Metric | N (clustered) | Delta | 90% CI | p(worse) |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
    for condition, metrics in report["degradation"].items():
        for metric, block in metrics.items():
            row = block["clustered"]
            lines.append(
                "| {condition} | {metric} | {n} | {delta:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {p:.2e} |".format(
                    condition=condition,
                    metric=metric,
                    n=row["n"],
                    delta=row["mean_delta"],
                    lo=row["ci90_low"],
                    hi=row["ci90_high"],
                    p=row["one_sided_p_worse"],
                )
            )
    lines.append("")
    lines.append("## Full-causal-weight controls: pairwise practical equivalence")
    lines.append("")
    lines.append("Tests whether replacement with translated states is practically")
    lines.append("equivalent to replacement with random / zero-vector controls")
    lines.append("(clustered view; runs paired by seed, deterministic runs paired with")
    lines.append("every seed).")
    lines.append("")
    header_margins = " | ".join(f"TOST p (margin {margin:.2f})" for margin in margins)
    lines.append(f"| Pair | Metric | N | Delta | 90% CI | {header_margins} |")
    lines.append(
        "| --- | --- | ---: | ---: | ---: | "
        + " | ".join("---:" for _ in margins)
        + " |"
    )
    for pair_name, metrics in report.get("pairwise_full_weight", {}).items():
        for metric, block in metrics.items():
            row = block["clustered"]
            cells = " | ".join(
                "{:.2e}".format(row[f"tost_p_margin_{f'{m:.2f}'.replace('.', 'p')}"])
                for m in margins
            )
            lines.append(
                "| {pair} | {metric} | {n} | {delta:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {cells} |".format(
                    pair=pair_name,
                    metric=metric,
                    n=row["n"],
                    delta=row["mean_delta"],
                    lo=row["ci90_low"],
                    hi=row["ci90_high"],
                    cells=cells,
                )
            )
    lines.append("")
    while lines and lines[-1] == "":
        lines.pop()
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run H1 TOST equivalence analysis.")
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
    parser.add_argument("--glob", default="h1_*.json")
    parser.add_argument(
        "--direction",
        default="fwd",
        help="Result direction to include; use 'all' to disable filtering.",
    )
    parser.add_argument(
        "--margins",
        type=float,
        nargs="+",
        default=[0.05, 0.03],
        help="Equivalence margins on the accuracy-delta scale.",
    )
    parser.add_argument("--bootstrap", type=int, default=10000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    direction = None if args.direction == "all" else args.direction
    results = load_results(args.results_dir, direction=direction, glob=args.glob)
    report = build_report(
        results, margins=tuple(args.margins), n_boot=args.bootstrap
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / "equivalence_report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    write_markdown(args.output_dir / "equivalence_report.md", report)
    print(f"Wrote equivalence report into {args.output_dir}")


if __name__ == "__main__":
    main()
