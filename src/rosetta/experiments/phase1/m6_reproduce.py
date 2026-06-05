"""M6 reproducible evaluation — replacement sequence-level injection.

Canonical, version-controlled CLI for reproducing M6 Phase D results.
Addresses m6_academic_review.md §R1 (reproducibility risk: m6_eval.py in scratch/).

This script wraps ``m35_diagnostic.run_p1_ablation`` with the correct parameters
to reproduce all M6 Phase D evaluation conditions:

    replace_fwd  — seeds 42/123/456  (160M→410M, replace injection)
    replace_rev  — seeds 42/123/456  (410M→160M, replace injection)
    additive_fwd — seed  42          (160M→410M, additive control)
    additive_rev — seed  42          (410M→160M, additive control)
    baseline_fwd — seed  42          (410M, no injection, full input)
    baseline_rev — seed  42          (160M, no injection, full input)

Usage
-----
Full Phase D reproduction (all 11 runs)::

    python -m rosetta.experiments.phase1.m6_reproduce \\
        --ckpt-dir results/phase1/checkpoints \\
        --output-dir results/phase1 \\
        --conditions all \\
        --seeds 42 123 456

Replace only (fwd, 3 seeds)::

    python -m rosetta.experiments.phase1.m6_reproduce \\
        --ckpt-dir results/phase1/checkpoints \\
        --conditions replace_fwd \\
        --seeds 42 123 456

Environment (WSL2 GPU)
----------------------
    export TRANSFORMERS_OFFLINE=1
    export HF_DATASETS_OFFLINE=1
    export HF_HUB_OFFLINE=1
    ~/gpu_venv/bin/python -m rosetta.experiments.phase1.m6_reproduce ...

Hardware: RTX 3060 6 GB (WSL2 CUDA). Each seed/direction takes ~15–20 min.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Sequence

import torch

from rosetta.experiments.phase1.m2_baseline import (
    evaluate_multi_hop,
    load_jsonl,
    load_model_and_tokenizer,
    load_test_split,
)
from rosetta.experiments.phase1.m3_translation import (
    _DEFAULT_RELATIVE_LAYER,
    _TIERS,
    resolve_layer_name,
)
from rosetta.experiments.phase1.m35_diagnostic import (
    _evaluate_samples,
    _get_acc,
    _get_sender_text_and_question,
    run_p1_ablation,
)
from rosetta.translation.translation_layer import (
    inject_and_generate,
    load_translation_layer,
)

logger = logging.getLogger(__name__)

# M6 configuration
_TIER_KEY = "tier2"        # Pythia-160M ↔ Pythia-410M
_TASK = "multi_hop"
_TEST_FILE = Path("data/tasks/multi_hop_reasoning/test_enhanced.jsonl")
_CORPUS_PATH = Path("data/corpus/wikitext103_clean.txt")  # required by run_p1_ablation
_LAYER_REL = _DEFAULT_RELATIVE_LAYER   # 0.67

_REPLACE_SEEDS = (42, 123, 456)
_CONTROL_SEED = 42

_ALL_CONDITIONS = (
    "replace_fwd", "replace_rev",
    "additive_fwd", "additive_rev",
    "baseline_fwd", "baseline_rev",
)


def _get_tier(reverse: bool) -> dict:
    cfg = _TIERS[_TIER_KEY]
    if reverse:
        return {
            "sender_id": cfg["receiver_id"],
            "receiver_id": cfg["sender_id"],
            "sender_layer_prefix": cfg["receiver_layer_prefix"],
            "sender_num_layers": cfg["receiver_num_layers"],
            "receiver_layer_prefix": cfg["sender_layer_prefix"],
            "receiver_num_layers": cfg["sender_num_layers"],
        }
    return cfg


def _ckpt_path(ckpt_dir: Path, direction: str, seed: int) -> Path:
    return ckpt_dir / f"m6_translation_{direction}_seed{seed}.pt"


def _save_result(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def run_replace_condition(
    ckpt_dir: Path,
    output_dir: Path,
    direction: str,
    seeds: Sequence[int],
    device: str,
    test_file: Path,
) -> list[dict]:
    """Reproduce D1: replacement injection (3 seeds, one direction)."""
    reverse = direction == "rev"
    cfg = _get_tier(reverse)
    layer_a = resolve_layer_name(
        cfg["sender_layer_prefix"], cfg["sender_num_layers"], _LAYER_REL
    )
    layer_b = resolve_layer_name(
        cfg["receiver_layer_prefix"], cfg["receiver_num_layers"], _LAYER_REL
    )
    logger.info("replace_%s: %s → %s  [%s → %s]",
                direction, cfg["sender_id"], cfg["receiver_id"], layer_a, layer_b)

    logger.info("Loading sender model: %s", cfg["sender_id"])
    model_a, tok_a = load_model_and_tokenizer(cfg["sender_id"], device)
    logger.info("Loading receiver model: %s", cfg["receiver_id"])
    model_b, tok_b = load_model_and_tokenizer(cfg["receiver_id"], device)

    samples = load_jsonl(test_file) if test_file.exists() else load_test_split(_TASK)
    logger.info("  %d test samples", len(samples))

    results = []
    for seed in seeds:
        ckpt = _ckpt_path(ckpt_dir, direction, seed)
        if not ckpt.exists():
            logger.error("Checkpoint not found: %s — skipping", ckpt)
            continue
        logger.info("  seed=%d  ckpt=%s", seed, ckpt.name)
        t0 = time.time()

        result_dict = run_p1_ablation(
            model_a=model_a, tokenizer_a=tok_a,
            model_b=model_b, tokenizer_b=tok_b,
            samples=samples,
            layer_name_a=layer_a, layer_name_b=layer_b,
            device=device,
            corpus_path=_CORPUS_PATH,           # unused in replace mode
            injection_timing="prefill_only",
            injection_mode="replace",
            full_input=True,
            translation_ckpt_override=ckpt,
        )
        elapsed = round(time.time() - t0, 1)
        acc = _get_acc(result_dict["M6_replace"]["metrics"])
        logger.info("    acc=%.4f  elapsed=%.1fs", acc, elapsed)

        out = {
            "condition": f"replace_{direction}",
            "seed": seed, "direction": direction,
            "injection_mode": "replace", "full_input": True,
            "n_samples": len(samples), "acc": acc, "elapsed_sec": elapsed,
            "checkpoint": str(ckpt), "layer_a": layer_a, "layer_b": layer_b,
            "model_a": cfg["sender_id"], "model_b": cfg["receiver_id"],
            "per_sample": result_dict["M6_replace"]["metrics"],
        }
        results.append({k: v for k, v in out.items() if k != "per_sample"})
        _save_result(output_dir / f"m6_reproduce_replace_{direction}_{seed}.json", out)
    return results


def run_additive_condition(
    ckpt_dir: Path,
    output_dir: Path,
    direction: str,
    seed: int,
    device: str,
    test_file: Path,
) -> dict:
    """Reproduce D2: additive injection control (single seed).

    Uses load_translation_layer + inject_and_generate directly (same as
    m6_eval.py::run_additive_eval), bypassing run_p1_ablation which does not
    accept translation_ckpt_override in additive mode.
    """
    import torch
    reverse = direction == "rev"
    cfg = _get_tier(reverse)
    layer_a = resolve_layer_name(
        cfg["sender_layer_prefix"], cfg["sender_num_layers"], _LAYER_REL
    )
    layer_b = resolve_layer_name(
        cfg["receiver_layer_prefix"], cfg["receiver_num_layers"], _LAYER_REL
    )
    ckpt = _ckpt_path(ckpt_dir, direction, seed)

    model_a, tok_a = load_model_and_tokenizer(cfg["sender_id"], device)
    model_b, tok_b = load_model_and_tokenizer(cfg["receiver_id"], device)
    samples = load_jsonl(test_file) if test_file.exists() else load_test_split(_TASK)

    logger.info("additive_%s: seed=%d  ckpt=%s", direction, seed, ckpt.name)
    tl = load_translation_layer(str(ckpt)).to(device).eval()

    t0 = time.time()
    predictions: list[str] = []
    for i, sample in enumerate(samples):
        ctx, q = _get_sender_text_and_question(sample, _TASK)
        prompt = f"Context: {ctx}\nQ: {q}\nA:"

        # Extract 3D activations from model A
        act_cache: dict = {}

        def _hook(_m, _inp, out, _c=act_cache):  # noqa: ARG001
            _c["h"] = (out[0] if isinstance(out, tuple) else out).detach()

        mod = model_a
        for part in layer_a.split("."):
            mod = getattr(mod, part)
        handle = mod.register_forward_hook(_hook)
        enc = tok_a(prompt, return_tensors="pt", truncation=True, max_length=256).to(device)
        with torch.no_grad():
            model_a(**enc)
        handle.remove()
        raw_act = act_cache["h"]  # (1, seq, dim_a)

        with torch.no_grad():
            translated = tl.translate(raw_act)  # (1, seq, dim_b)

        pred = inject_and_generate(
            model_b, tok_b, translated, layer_b,
            question=q, task=_TASK, device=device,
            injection_scale=0.01, injection_timing="prefill_only",
            injection_mode="additive", context=ctx, full_input=True,
        )
        predictions.append(pred)
        if (i + 1) % 100 == 0:
            logger.info("  [additive_%s seed=%d] %d/%d", direction, seed, i + 1, len(samples))

    elapsed = round(time.time() - t0, 1)

    n_correct = sum(
        evaluate_multi_hop(pred, s["answer"]).get("correct", False)
        for pred, s in zip(predictions, samples)
    )
    acc = n_correct / len(samples)
    logger.info("    acc=%.4f  elapsed=%.1fs", acc, elapsed)

    out = {
        "condition": f"additive_{direction}",
        "seed": seed, "direction": direction,
        "injection_mode": "additive", "injection_scale": 0.01,
        "full_input": True, "n_samples": len(samples),
        "acc": acc, "elapsed_sec": elapsed,
        "checkpoint": str(ckpt), "layer_a": layer_a, "layer_b": layer_b,
    }
    _save_result(output_dir / f"m6_reproduce_additive_{direction}_{seed}.json", out)
    return out


def run_baseline_condition(
    output_dir: Path,
    direction: str,
    seed: int,
    device: str,
    test_file: Path,
) -> dict:
    """Reproduce D3: no-injection baseline (full input)."""
    reverse = direction == "rev"
    cfg = _get_tier(reverse)
    # Baseline uses only model B (receiver)
    model_b, tok_b = load_model_and_tokenizer(cfg["receiver_id"], device)
    samples = load_jsonl(test_file) if test_file.exists() else load_test_split(_TASK)

    logger.info("baseline_%s: model=%s  n=%d", direction, cfg["receiver_id"], len(samples))
    t0 = time.time()

    predictions: list[str] = []
    for sample in samples:
        ctx, q = _get_sender_text_and_question(sample, _TASK)
        prompt = f"Context: {ctx}\nQ: {q}\nA:"
        enc = tok_b(prompt, return_tensors="pt", truncation=True, max_length=256).to(device)
        with torch.no_grad():
            ids = model_b.generate(
                **enc, max_new_tokens=64, do_sample=False,
                repetition_penalty=1.3, no_repeat_ngram_size=3,
                pad_token_id=tok_b.eos_token_id,
            )
        pred = tok_b.decode(ids[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        predictions.append(pred)

    evaled = _evaluate_samples(samples, _TASK, predictions)
    acc = _get_acc(evaled["metrics"])
    elapsed = round(time.time() - t0, 1)
    logger.info("    acc=%.4f  elapsed=%.1fs", acc, elapsed)

    out = {
        "condition": f"baseline_{direction}",
        "seed": seed, "direction": direction,
        "injection_mode": "none", "full_input": True,
        "n_samples": len(samples), "acc": acc, "elapsed_sec": elapsed,
        "model_b": cfg["receiver_id"],
        "per_sample": evaled["metrics"],
    }
    _save_result(output_dir / f"m6_reproduce_baseline_{direction}_{seed}.json", out)
    return {k: v for k, v in out.items() if k != "per_sample"}


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="Reproduce M6 Phase D evaluation (replacement injection).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="See module docstring for full usage examples.",
    )
    p.add_argument("--ckpt-dir", type=Path, default=Path("results/phase1/checkpoints"),
                   help="Directory with m6_translation_*.pt checkpoints.")
    p.add_argument("--output-dir", type=Path, default=Path("results/phase1"),
                   help="Directory for output JSON files.")
    p.add_argument("--conditions", nargs="+",
                   choices=list(_ALL_CONDITIONS) + ["all"], default=["all"],
                   help="Conditions to run (default: all).")
    p.add_argument("--seeds", nargs="+", type=int, default=list(_REPLACE_SEEDS),
                   help="Seeds for replace conditions (default: 42 123 456).")
    p.add_argument("--device", default="auto",
                   help="Device: cuda / cpu / auto (default: auto).")
    p.add_argument("--test-file", type=Path, default=_TEST_FILE,
                   help="Path to test JSONL file.")
    args = p.parse_args(argv)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    logger.info("Device: %s", device)

    conditions = set(_ALL_CONDITIONS if "all" in args.conditions else args.conditions)
    summary = []

    if "replace_fwd" in conditions:
        summary += run_replace_condition(
            args.ckpt_dir, args.output_dir, "fwd", args.seeds, device, args.test_file)
    if "replace_rev" in conditions:
        summary += run_replace_condition(
            args.ckpt_dir, args.output_dir, "rev", args.seeds, device, args.test_file)
    if "additive_fwd" in conditions:
        summary.append(run_additive_condition(
            args.ckpt_dir, args.output_dir, "fwd", _CONTROL_SEED, device, args.test_file))
    if "additive_rev" in conditions:
        summary.append(run_additive_condition(
            args.ckpt_dir, args.output_dir, "rev", _CONTROL_SEED, device, args.test_file))
    if "baseline_fwd" in conditions:
        summary.append(run_baseline_condition(
            args.output_dir, "fwd", _CONTROL_SEED, device, args.test_file))
    if "baseline_rev" in conditions:
        summary.append(run_baseline_condition(
            args.output_dir, "rev", _CONTROL_SEED, device, args.test_file))

    _save_result(args.output_dir / "m6_reproduce_summary.json",
                 {"conditions_run": sorted(conditions), "results": summary})

    logger.info("=" * 60)
    for r in summary:
        logger.info("  %-25s seed=%-5s  acc=%.4f",
                    r["condition"], str(r.get("seed", "—")), r["acc"])
    logger.info("Done. %d conditions completed.", len(summary))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
