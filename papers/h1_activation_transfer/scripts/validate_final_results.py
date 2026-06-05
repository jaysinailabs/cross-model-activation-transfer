"""Validate H1 final result files before paper use or external audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

EXPECTED_HASH = "504e077cf17433e22967c86e98d321532d4e803dbe24d96af14c7e8ecdd0dcbb"
EXPECTED_N = 396
STRICT_MATCHED_N = 383
STRICT_MATCHED_EXCLUDED_SINGLETONS = 13
EXPECTED_SEEDS = (42, 123, 456)
DETERMINISTIC_CONDITIONS = ("no_inject", "nl_relay", "b_to_b_self_inject")
SEEDED_CONDITIONS = (
    "additive",
    "replace",
    "scale_corrected",
    "best_alpha",
    "same_norm_random",
    "shuffled_translation",
)
STRICT_MATCHED_CONDITION = "shuffled_translation_strict_matched"


def load_results(results_dirs: list[Path], *, direction: str = "fwd") -> list[dict[str, Any]]:
    results = []
    seen_paths = set()
    for results_dir in results_dirs:
        for path in sorted(results_dir.glob("h1_*.json")):
            resolved = path.resolve()
            if resolved in seen_paths:
                continue
            seen_paths.add(resolved)
            with open(path, encoding="utf-8") as fh:
                payload = json.load(fh)
            if direction != "all" and payload.get("direction") != direction:
                continue
            payload["_path"] = str(path)
            payload["_file"] = path.name
            results.append(payload)
    return results


def expected_run_keys(*, include_strict_matched: bool = False) -> set[tuple[str, int | None]]:
    keys: set[tuple[str, int | None]] = {
        (condition, None) for condition in DETERMINISTIC_CONDITIONS
    }
    seeded_conditions = list(SEEDED_CONDITIONS)
    if include_strict_matched:
        seeded_conditions.append(STRICT_MATCHED_CONDITION)
    for condition in seeded_conditions:
        for seed in EXPECTED_SEEDS:
            keys.add((condition, seed))
    return keys


def actual_run_key(result: dict[str, Any]) -> tuple[str, int | None]:
    seed = result.get("seed")
    return result.get("condition"), int(seed) if seed is not None else None


def expected_n_for_condition(condition: str | None) -> int:
    if condition == STRICT_MATCHED_CONDITION:
        return STRICT_MATCHED_N
    return EXPECTED_N


def validate(
    results: list[dict[str, Any]], *, include_strict_matched: bool = False, direction: str = "fwd"
) -> dict[str, Any]:
    blocking: list[str] = []
    warnings: list[str] = []

    expected = expected_run_keys(include_strict_matched=include_strict_matched)
    actual = {actual_run_key(result) for result in results}
    missing = sorted(expected - actual, key=lambda x: (x[0], -1 if x[1] is None else x[1]))
    extra = sorted(actual - expected, key=lambda x: (x[0], -1 if x[1] is None else x[1]))
    if missing:
        blocking.append("missing_expected_runs")
    if extra:
        warnings.append("extra_result_files_present")

    per_run = []
    for result in results:
        diagnostics = result.get("diagnostics", {})
        config = result.get("config", {})
        metrics = result.get("metrics", {})
        injection = result.get("injection", {})
        condition = result.get("condition")
        expected_n = expected_n_for_condition(condition)
        shuffle_fallback = diagnostics.get("shuffle_self_fallback_count")
        shuffle_excluded = injection.get("shuffle_excluded_singleton_count")
        record = {
            "file": result["_file"],
            "condition": condition,
            "seed": result.get("seed"),
            "n": metrics.get("n"),
            "expected_n": expected_n,
            "clean_eval_hash": config.get("clean_eval_hash"),
            "contains_acc": metrics.get("contains_acc"),
            "exact_match_acc": metrics.get("exact_match_acc"),
            "seq_len_mismatch_count": diagnostics.get("seq_len_mismatch_count"),
            "token_mismatch_count": diagnostics.get("token_mismatch_count"),
            "shuffle_self_fallback_count": shuffle_fallback,
            "shuffle_excluded_singleton_count": shuffle_excluded,
            "source_clean_eval_n_samples": config.get("source_clean_eval_n_samples"),
            "mean_translated_norm": diagnostics.get("mean_translated_norm"),
            "mean_b_hidden_norm": diagnostics.get("mean_b_hidden_norm"),
            "elapsed_sec": result.get("elapsed_sec"),
        }
        per_run.append(record)

        if metrics.get("n") != expected_n:
            blocking.append(f"unexpected_n:{result['_file']}")
        if config.get("clean_eval_hash") != EXPECTED_HASH:
            blocking.append(f"clean_eval_hash_mismatch:{result['_file']}")
        if diagnostics.get("seq_len_mismatch_count", 0) != 0:
            blocking.append(f"seq_len_mismatch:{result['_file']}")
        if diagnostics.get("token_mismatch_count", 0) != 0:
            blocking.append(f"token_mismatch:{result['_file']}")
        if condition == STRICT_MATCHED_CONDITION:
            if config.get("source_clean_eval_n_samples") != EXPECTED_N:
                blocking.append(f"strict_source_n_mismatch:{result['_file']}")
            if shuffle_excluded != STRICT_MATCHED_EXCLUDED_SINGLETONS:
                blocking.append(f"strict_excluded_count_mismatch:{result['_file']}")
            if shuffle_fallback not in (None, 0):
                blocking.append(f"strict_shuffle_self_fallback:{result['_file']}")
        elif shuffle_fallback not in (None, 0):
            warnings.append(f"shuffle_self_fallback:{result['_file']}")

    clean_hashes = sorted({r.get("config", {}).get("clean_eval_hash") for r in results})
    return {
        "summary": {
            "n_files": len(results),
            "expected_files": len(expected),
            "direction_filter": direction,
            "expected_n": EXPECTED_N,
            "strict_matched_expected_n": STRICT_MATCHED_N if include_strict_matched else None,
            "expected_hash": EXPECTED_HASH,
            "clean_eval_hashes": clean_hashes,
            "blocking": sorted(set(blocking)),
            "warnings": sorted(set(warnings)),
            "passes_blocking_checks": not blocking,
            "ready_for_external_audit": not blocking,
        },
        "missing_runs": [{"condition": c, "seed": s} for c, s in missing],
        "extra_runs": [{"condition": c, "seed": s} for c, s in extra],
        "per_run": per_run,
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    summary = report["summary"]
    lines = [
        "# H1 Final Result Validation",
        "",
        f"- result files: {summary['n_files']} / {summary['expected_files']}",
        f"- direction filter: `{summary['direction_filter']}`",
        f"- expected n: {summary['expected_n']}",
        f"- expected clean eval hash: `{summary['expected_hash']}`",
        f"- blocking checks passed: `{summary['passes_blocking_checks']}`",
        f"- ready for external audit: `{summary['ready_for_external_audit']}`",
        "",
        "## Blocking",
        "",
    ]
    if summary.get("strict_matched_expected_n") is not None:
        lines.insert(5, f"- strict matched expected n: {summary['strict_matched_expected_n']}")
    if summary["blocking"]:
        lines.extend(f"- `{item}`" for item in summary["blocking"])
    else:
        lines.append("- none")
    lines.extend(["", "## Warnings", ""])
    if summary["warnings"]:
        lines.extend(f"- `{item}`" for item in summary["warnings"])
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Per-Run Diagnostics",
            "",
            "| Condition | Seed | N | Expected N | Contains | Seq Mismatch | Token Mismatch | Shuffle Fallback | Excluded Singletons |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in report["per_run"]:
        lines.append(
            "| {condition} | {seed} | {n} | {expected_n} | {contains:.4f} | {seq} | {token} | {shuffle} | {excluded} |".format(
                condition=row["condition"],
                seed="" if row["seed"] is None else row["seed"],
                n=row["n"],
                expected_n=row["expected_n"],
                contains=float(row["contains_acc"] or 0.0),
                seq=row["seq_len_mismatch_count"],
                token=row["token_mismatch_count"],
                shuffle=(
                    ""
                    if row["shuffle_self_fallback_count"] is None
                    else row["shuffle_self_fallback_count"]
                ),
                excluded=(
                    ""
                    if row["shuffle_excluded_singleton_count"] is None
                    else row["shuffle_excluded_singleton_count"]
                ),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate H1 final result files.")
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
    parser.add_argument(
        "--include-strict-matched",
        action="store_true",
        help="Expect and validate shuffled_translation_strict_matched result files.",
    )
    parser.add_argument(
        "--direction",
        default="fwd",
        choices=["fwd", "rev", "all"],
        help="Validate one direction by default; use 'all' for diagnostic sweeps.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = validate(
        load_results(args.results_dir, direction=args.direction),
        include_strict_matched=args.include_strict_matched,
        direction=args.direction,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / "final_validation_report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    write_markdown(args.output_dir / "final_validation_report.md", report)
    print(f"passes_blocking_checks: {report['summary']['passes_blocking_checks']}")
    if report["summary"]["warnings"]:
        print("warnings: " + ", ".join(report["summary"]["warnings"]))
    if report["summary"]["blocking"]:
        print("blocking: " + ", ".join(report["summary"]["blocking"]))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
