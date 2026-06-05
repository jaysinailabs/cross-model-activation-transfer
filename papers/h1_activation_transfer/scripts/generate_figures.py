"""Generate reproducible figures for the H1 activation-transfer paper package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


DEFAULT_ROOT = Path(__file__).resolve().parents[1]

CONDITION_LABELS = {
    "no_inject": "No injection",
    "b_to_b_self_inject": "B-to-B\nself-inject",
    "nl_relay": "Natural-\nlanguage relay",
    "additive": "Additive",
    "best_alpha": "Best alpha",
    "replace": "Replace",
    "scale_corrected": "Scale-\ncorrected",
    "same_norm_random": "Same-norm\nrandom",
    "zero_replacement": "Zero\nreplacement",
    "shuffled_translation": "Shuffled\ntranslation",
    "shuffled_translation_strict_matched": "Shuffled strict\nmatched",
}

SUMMARY_ORDER = [
    "no_inject",
    "b_to_b_self_inject",
    "nl_relay",
    "additive",
    "best_alpha",
    "replace",
    "scale_corrected",
    "same_norm_random",
    "zero_replacement",
    "shuffled_translation",
    "shuffled_translation_strict_matched",
]

DELTA_ORDER = [
    "nl_relay",
    "additive",
    "best_alpha",
    "replace",
    "scale_corrected",
    "same_norm_random",
    "zero_replacement",
    "shuffled_translation",
    "shuffled_translation_strict_matched",
    "b_to_b_self_inject",
]

NORM_ORDER = ["additive", "replace", "scale_corrected", "best_alpha"]

PALETTE = {
    "baseline": "#4c78a8",
    "relay": "#59a14f",
    "transfer": "#f28e2b",
    "control": "#b07aa1",
    "strict": "#9c755f",
    "negative": "#e15759",
    "neutral": "#bab0ac",
}

CONDITION_COLORS = {
    "no_inject": PALETTE["baseline"],
    "b_to_b_self_inject": PALETTE["baseline"],
    "nl_relay": PALETTE["relay"],
    "additive": PALETTE["transfer"],
    "best_alpha": PALETTE["transfer"],
    "replace": PALETTE["negative"],
    "scale_corrected": PALETTE["negative"],
    "same_norm_random": PALETTE["control"],
    "zero_replacement": PALETTE["control"],
    "shuffled_translation": PALETTE["control"],
    "shuffled_translation_strict_matched": PALETTE["strict"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate paper figures from locked H1 result artifacts."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_ROOT / "results",
        help="Directory containing final summaries and audit CSVs.",
    )
    parser.add_argument(
        "--strict-summary-dir",
        type=Path,
        default=DEFAULT_ROOT / "results" / "final_strict_controls_summary",
        help="Directory containing strict matched control summaries.",
    )
    parser.add_argument(
        "--combined-comparisons-dir",
        type=Path,
        default=DEFAULT_ROOT / "results" / "final_with_strict_controls",
        help="Directory containing pairwise comparisons with strict controls.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_ROOT / "paper" / "figures",
        help="Directory for generated PNG/PDF figures.",
    )
    return parser.parse_args()


def ensure_required(paths: Iterable[Path]) -> None:
    missing = [path for path in paths if not path.exists()]
    if missing:
        formatted = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"Missing required input files:\n{formatted}")


def read_condition_summary(results_dir: Path, strict_summary_dir: Path) -> pd.DataFrame:
    main_path = results_dir / "final_summary_by_condition.csv"
    strict_path = strict_summary_dir / "final_summary_by_condition.csv"
    ensure_required([main_path, strict_path])

    main = pd.read_csv(main_path)
    strict = pd.read_csv(strict_path)
    combined = pd.concat([main, strict], ignore_index=True)
    combined = combined[combined["condition"].isin(SUMMARY_ORDER)].copy()
    combined["condition"] = pd.Categorical(
        combined["condition"], categories=SUMMARY_ORDER, ordered=True
    )
    return combined.sort_values("condition")


def read_pairwise(combined_comparisons_dir: Path) -> pd.DataFrame:
    path = combined_comparisons_dir / "final_pairwise_comparisons_by_condition.csv"
    ensure_required([path])
    return pd.read_csv(path)


def read_audit_counts(results_dir: Path) -> pd.DataFrame:
    path = results_dir / "metric_disagreement_audit.csv"
    ensure_required([path])
    audit = pd.read_csv(path)
    counts = (
        audit["audit_label"]
        .fillna("unlabeled")
        .value_counts()
        .rename_axis("audit_label")
        .reset_index(name="count")
    )
    label_order = [
        "verbose_correct",
        "substring_noise",
        "accidental_mention",
        "unlabeled",
    ]
    counts["audit_label"] = pd.Categorical(
        counts["audit_label"], categories=label_order, ordered=True
    )
    return counts.sort_values("audit_label")


def read_norm_diagnostics(results_dir: Path) -> pd.DataFrame:
    final_dir = results_dir / "final"
    ensure_required([final_dir])
    rows = []
    for path in sorted(final_dir.glob("h1_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        condition = payload.get("condition")
        if condition not in NORM_ORDER:
            continue
        diagnostics = payload.get("diagnostics", {})
        rows.append(
            {
                "condition": condition,
                "seed": payload.get("seed"),
                "translated_norm": diagnostics.get("mean_translated_norm"),
                "receiver_hidden_norm": diagnostics.get("mean_b_hidden_norm"),
                "injected_norm": diagnostics.get("mean_injected_norm"),
            }
        )

    if not rows:
        raise ValueError("No norm diagnostics found in final result JSON files.")

    df = pd.DataFrame(rows)
    return df.groupby("condition", as_index=False).mean(numeric_only=True)


def condition_labels(conditions: Iterable[str]) -> list[str]:
    return [CONDITION_LABELS.get(condition, condition) for condition in conditions]


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return {"stem": stem, "png": png_path.name, "pdf": pdf_path.name}


def style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#d8d8d8", linewidth=0.7, alpha=0.7)
    ax.set_axisbelow(True)


def add_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    *,
    facecolor: str,
    edgecolor: str = "#333333",
    fontsize: int = 10,
    fontweight: str = "normal",
) -> None:
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.0,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=fontweight,
        color="#111111",
    )


def add_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = "#333333",
    linestyle: str = "-",
    linewidth: float = 1.4,
    connectionstyle: str = "arc3,rad=0.0",
) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=13,
        linewidth=linewidth,
        linestyle=linestyle,
        color=color,
        connectionstyle=connectionstyle,
    )
    ax.add_patch(arrow)


def plot_method_diagram(output_dir: Path) -> dict[str, str]:
    fig, ax = plt.subplots(figsize=(11.2, 5.3))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    add_box(
        ax,
        0.04,
        0.64,
        0.14,
        0.16,
        "Task prompt\ncontext + question",
        facecolor="#f4f4f4",
        fontsize=9,
    )
    add_box(
        ax,
        0.24,
        0.66,
        0.16,
        0.18,
        "Sender\nPythia-160M\nlayer 8",
        facecolor="#d9e7f7",
        fontsize=9,
        fontweight="bold",
    )
    add_box(
        ax,
        0.47,
        0.67,
        0.13,
        0.16,
        "Linear\ntranslator\n160M -> 410M",
        facecolor="#fde3c2",
        fontsize=9,
        fontweight="bold",
    )
    add_box(
        ax,
        0.69,
        0.66,
        0.18,
        0.18,
        "Receiver\nPythia-410M\nlayer 16 injection",
        facecolor="#dff0d8",
        fontsize=9,
        fontweight="bold",
    )
    add_box(
        ax,
        0.91,
        0.68,
        0.06,
        0.13,
        "Answer",
        facecolor="#f4f4f4",
        fontsize=9,
    )

    add_arrow(ax, (0.18, 0.72), (0.24, 0.75))
    add_arrow(ax, (0.40, 0.75), (0.47, 0.75))
    add_arrow(ax, (0.60, 0.75), (0.69, 0.75), color=PALETTE["transfer"])
    add_arrow(ax, (0.87, 0.75), (0.91, 0.75))

    ax.text(
        0.635,
        0.82,
        "translated hidden states",
        ha="center",
        va="bottom",
        fontsize=9,
        color=PALETTE["transfer"],
    )

    add_box(
        ax,
        0.24,
        0.30,
        0.16,
        0.13,
        "Sender\ntext relay",
        facecolor="#e4f2df",
        fontsize=9,
    )
    add_box(
        ax,
        0.51,
        0.30,
        0.18,
        0.13,
        "Receiver reads\nrelay + question",
        facecolor="#e4f2df",
        fontsize=9,
    )
    add_arrow(ax, (0.18, 0.68), (0.25, 0.41), color=PALETTE["relay"])
    add_arrow(ax, (0.40, 0.365), (0.51, 0.365), color=PALETTE["relay"])
    add_arrow(
        ax,
        (0.69, 0.365),
        (0.91, 0.72),
        color=PALETTE["relay"],
        connectionstyle="arc3,rad=-0.18",
    )
    ax.text(
        0.48,
        0.45,
        "natural-language relay baseline",
        ha="center",
        va="bottom",
        fontsize=9,
        color=PALETTE["relay"],
    )

    add_box(
        ax,
        0.04,
        0.09,
        0.18,
        0.12,
        "No injection\nreceiver baseline",
        facecolor="#eef3fb",
        fontsize=8,
    )
    add_box(
        ax,
        0.28,
        0.09,
        0.18,
        0.12,
        "B-to-B self-inject\nhook sanity check",
        facecolor="#eef3fb",
        fontsize=8,
    )
    add_box(
        ax,
        0.52,
        0.09,
        0.18,
        0.12,
        "Same-norm random\nmagnitude control",
        facecolor="#f1e8f4",
        fontsize=8,
    )
    add_box(
        ax,
        0.76,
        0.09,
        0.18,
        0.12,
        "Shuffled translation\nsignal control",
        facecolor="#f1e8f4",
        fontsize=8,
    )
    ax.text(
        0.49,
        0.24,
        "Controls separate hook correctness, vector magnitude, and transferred-signal specificity.",
        ha="center",
        va="center",
        fontsize=9,
        color="#333333",
    )

    fig.tight_layout()
    return save_figure(fig, output_dir, "fig00_method_diagram")


def plot_boundary_summary(summary: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    plot_df = summary.copy()
    plot_df["score_pct"] = 100.0 * plot_df["word_boundary_contains_acc_mean"]
    plot_df["std_pct"] = 100.0 * plot_df["word_boundary_contains_acc_std"].fillna(0.0)

    fig, ax = plt.subplots(figsize=(11.2, 5.2))
    x = np.arange(len(plot_df))
    colors = [CONDITION_COLORS.get(c, PALETTE["neutral"]) for c in plot_df["condition"]]
    ax.bar(
        x,
        plot_df["score_pct"],
        yerr=plot_df["std_pct"],
        capsize=3,
        color=colors,
        edgecolor="#333333",
        linewidth=0.5,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(condition_labels(plot_df["condition"]), rotation=0, ha="center")
    ax.set_ylabel("Word-boundary answer containment (%)")
    ax.set_title("Final clean-eval performance by condition")
    ax.set_ylim(0, max(6.5, plot_df["score_pct"].max() + 1.0))
    style_axes(ax)
    fig.tight_layout()
    return save_figure(fig, output_dir, "fig01_boundary_contains_by_condition")


def plot_delta(
    pairwise: pd.DataFrame, output_dir: Path, baseline: str, stem: str, title: str
) -> dict[str, str]:
    plot_df = pairwise[
        (pairwise["baseline"] == baseline)
        & (pairwise["metric"] == "word_boundary_contains")
        & (pairwise["condition"].isin(DELTA_ORDER))
        & (pairwise["condition"] != baseline)
    ].copy()
    plot_df["condition"] = pd.Categorical(
        plot_df["condition"], categories=DELTA_ORDER, ordered=True
    )
    plot_df = plot_df.sort_values("condition")

    if plot_df.empty:
        raise ValueError(f"No word-boundary pairwise rows found for baseline={baseline}.")

    y = np.arange(len(plot_df))
    x = 100.0 * plot_df["delta_mean"].to_numpy()
    low = 100.0 * plot_df["ci95_low_pooled"].to_numpy()
    high = 100.0 * plot_df["ci95_high_pooled"].to_numpy()
    xerr = np.vstack([x - low, high - x])
    colors = [CONDITION_COLORS.get(c, PALETTE["neutral"]) for c in plot_df["condition"]]

    fig, ax = plt.subplots(figsize=(8.4, 5.5))
    ax.axvline(0, color="#333333", linewidth=1.0, linestyle="--")
    ax.errorbar(
        x,
        y,
        xerr=xerr,
        fmt="none",
        ecolor="#333333",
        elinewidth=1.2,
        capsize=3,
        zorder=1,
    )
    ax.scatter(x, y, s=58, color=colors, edgecolor="#333333", linewidth=0.6, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels(condition_labels(plot_df["condition"]))
    ax.set_xlabel("Paired delta in word-boundary containment (percentage points)")
    ax.set_title(title)
    ax.grid(axis="x", color="#d8d8d8", linewidth=0.7, alpha=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return save_figure(fig, output_dir, stem)


def plot_norm_mismatch(norms: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    plot_df = norms[norms["condition"].isin(NORM_ORDER)].copy()
    plot_df["condition"] = pd.Categorical(
        plot_df["condition"], categories=NORM_ORDER, ordered=True
    )
    plot_df = plot_df.sort_values("condition")

    diagnostic_labels = {
        "translated_norm": "Translated vector",
        "receiver_hidden_norm": "Receiver hidden state",
        "injected_norm": "Injected vector",
    }

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    width = 0.24
    diagnostics = ["translated_norm", "receiver_hidden_norm", "injected_norm"]
    offsets = [-width, 0.0, width]
    colors = ["#4c78a8", "#f28e2b", "#59a14f"]
    x = np.arange(len(plot_df))

    for diagnostic, offset, color in zip(diagnostics, offsets, colors):
        values = plot_df[diagnostic].to_numpy()
        ax.bar(
            x + offset,
            values,
            width=width,
            label=diagnostic_labels[diagnostic],
            color=color,
            edgecolor="#333333",
            linewidth=0.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(condition_labels(plot_df["condition"]))
    ax.set_yscale("log")
    ax.set_ylabel("Mean L2 norm (log scale)")
    ax.set_title("Activation norm scale mismatch")
    ax.legend(frameon=False, ncols=1)
    style_axes(ax)
    fig.tight_layout()
    return save_figure(fig, output_dir, "fig04_activation_norm_mismatch")


def plot_audit_labels(counts: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    label_map = {
        "verbose_correct": "Verbose correct",
        "substring_noise": "Substring noise",
        "accidental_mention": "Accidental\nmention",
        "unlabeled": "Unlabeled",
    }
    plot_df = counts.copy()
    plot_df["label"] = plot_df["audit_label"].astype(str).map(label_map)

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    colors = ["#59a14f", "#e15759", "#b07aa1", "#bab0ac"]
    ax.bar(
        plot_df["label"],
        plot_df["count"],
        color=colors[: len(plot_df)],
        edgecolor="#333333",
        linewidth=0.5,
    )
    ax.set_ylabel("Disagreement cases")
    ax.set_title("Legacy contains disagreement audit labels")
    style_axes(ax)
    fig.tight_layout()
    return save_figure(fig, output_dir, "fig05_metric_disagreement_audit")


def write_manifest(output_dir: Path, entries: list[dict[str, str]]) -> None:
    manifest_path = output_dir / "figures_manifest.json"
    manifest_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")

    readme_lines = [
        "# H1 Paper Figures",
        "",
        "Generated by `papers/h1_activation_transfer/scripts/generate_figures.py`.",
        "Each figure is emitted as both PNG for quick review and PDF for paper builds.",
        "",
        "| Figure | Purpose | Files |",
        "|---|---|---|",
    ]
    purposes = {
        "fig00_method_diagram": "Method overview showing sender, translator, receiver injection point, relay baseline, and controls.",
        "fig01_boundary_contains_by_condition": "Main condition-level performance overview on the locked clean-eval set.",
        "fig02_paired_delta_vs_no_inject": "Paired word-boundary deltas against the no-injection baseline with pooled bootstrap CIs.",
        "fig03_paired_delta_vs_nl_relay": "Paired word-boundary deltas against the natural-language relay baseline with pooled bootstrap CIs.",
        "fig04_activation_norm_mismatch": "Diagnostic norm-scale comparison for activation-transfer variants.",
        "fig05_metric_disagreement_audit": "Audit labels for legacy contains versus word-boundary metric disagreements.",
    }
    for entry in entries:
        stem = entry["stem"]
        readme_lines.append(
            f"| `{stem}` | {purposes.get(stem, 'Generated figure.')} | "
            f"`{Path(entry['png']).name}`, `{Path(entry['pdf']).name}` |"
        )
    readme_lines.append("")
    readme_lines.append(
        "The figures are derived from locked result summaries and do not rerun model inference."
    )
    readme_path = output_dir / "README.md"
    readme_path.write_text("\n".join(readme_lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    summary = read_condition_summary(args.results_dir, args.strict_summary_dir)
    pairwise = read_pairwise(args.combined_comparisons_dir)
    audit_counts = read_audit_counts(args.results_dir)
    norm_diagnostics = read_norm_diagnostics(args.results_dir)

    entries = [
        plot_method_diagram(args.output_dir),
        plot_boundary_summary(summary, args.output_dir),
        plot_delta(
            pairwise,
            args.output_dir,
            baseline="no_inject",
            stem="fig02_paired_delta_vs_no_inject",
            title="Paired deltas against no injection",
        ),
        plot_delta(
            pairwise,
            args.output_dir,
            baseline="nl_relay",
            stem="fig03_paired_delta_vs_nl_relay",
            title="Paired deltas against natural-language relay",
        ),
        plot_norm_mismatch(norm_diagnostics, args.output_dir),
        plot_audit_labels(audit_counts, args.output_dir),
    ]
    write_manifest(args.output_dir, entries)


if __name__ == "__main__":
    main()
