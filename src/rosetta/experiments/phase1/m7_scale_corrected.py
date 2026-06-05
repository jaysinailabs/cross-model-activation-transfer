"""M7 scale-corrected replacement injection evaluation.

Tests whether per-position norm rescaling of translated activations resolves the
H1 failure observed in M6 (replace injection, acc ≈ 0.034 vs baseline 0.123).

M6 root cause hypothesis: scale_ratio = 85.1× (translate() output norm ≈ 0.85,
B natural hidden norm ≈ 72.4).  This script injects with B's own magnitude but
the translated direction, and measures whether acc recovers to/above baseline.

Usage
-----
Single run (one direction × seed)::

    python -m rosetta.experiments.phase1.m7_scale_corrected \\
        --direction fwd --seed 42

Full 6-run batch (submit in parallel on GPU machine)::

    for direction in fwd rev; do
      for seed in 42 123 456; do
        nohup ~/gpu_venv/bin/python -m rosetta.experiments.phase1.m7_scale_corrected \\
          --direction $direction --seed $seed \\
          >> ~/m7_g1_${direction}_${seed}.log 2>&1 &
      done
    done

Output: results/phase1/m7_scale_corrected_{direction}_seed{seed}.json

Environment (WSL2 GPU)
----------------------
    export TRANSFORMERS_OFFLINE=1
    export HF_DATASETS_OFFLINE=1
    export HF_HUB_OFFLINE=1
    ~/gpu_venv/bin/python -m rosetta.experiments.phase1.m7_scale_corrected ...

Hardware: RTX 3060 6 GB (WSL2 CUDA).  Each run takes ~10–15 min (same as M6
replace condition).
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
    _get_sender_text_and_question,
)
from rosetta.translation.translation_layer import (
    inject_and_generate,
    load_translation_layer,
)

logger = logging.getLogger(__name__)

# M7 inherits the same task/tier configuration as M6
_TIER_KEY = "tier2"        # Pythia-160M ↔ Pythia-410M
_TASK = "multi_hop"
_TEST_FILE = Path("data/tasks/multi_hop_reasoning/test_enhanced.jsonl")
_LAYER_REL = _DEFAULT_RELATIVE_LAYER   # 0.67


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


def run_scale_corrected(
    direction: str,
    seed: int,
    ckpt_dir: Path,
    output_dir: Path,
    device: str,
    test_file: Path,
    alpha: float = 1.0,
) -> dict:
    """Run one scale-corrected replacement evaluation (single direction × seed).

    Args:
        direction: ``"fwd"`` (160M→410M) or ``"rev"`` (410M→160M).
        seed: Random seed that identifies which M6 checkpoint to load.
        ckpt_dir: Directory containing ``m6_translation_*.pt`` checkpoints.
        output_dir: Destination directory for the output JSON file.
        device: Torch device string.
        test_file: Path to the test JSONL file.

    Returns:
        Result dict (also written to ``output_dir/m7_scale_corrected_{direction}_seed{seed}.json``).
    """
    reverse = direction == "rev"
    cfg = _get_tier(reverse)
    layer_a = resolve_layer_name(
        cfg["sender_layer_prefix"], cfg["sender_num_layers"], _LAYER_REL
    )
    layer_b = resolve_layer_name(
        cfg["receiver_layer_prefix"], cfg["receiver_num_layers"], _LAYER_REL
    )
    ckpt = ckpt_dir / f"m6_translation_{direction}_seed{seed}.pt"

    logger.info(
        "scale_corrected_%s seed=%d: %s → %s  [%s → %s]",
        direction, seed, cfg["sender_id"], cfg["receiver_id"], layer_a, layer_b,
    )
    logger.info("  checkpoint: %s", ckpt)

    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    logger.info("Loading sender model: %s", cfg["sender_id"])
    model_a, tok_a = load_model_and_tokenizer(cfg["sender_id"], device)
    logger.info("Loading receiver model: %s", cfg["receiver_id"])
    model_b, tok_b = load_model_and_tokenizer(cfg["receiver_id"], device)
    tl = load_translation_layer(str(ckpt)).to(device).eval()

    samples = load_jsonl(test_file) if test_file.exists() else load_test_split(_TASK)
    logger.info("  %d test samples", len(samples))

    t0 = time.time()
    predictions: list[str] = []

    for i, sample in enumerate(samples):
        ctx, q = _get_sender_text_and_question(sample, _TASK)
        prompt = f"Context: {ctx}\nQ: {q}\nA:"

        # Extract sequence-level activations from model A
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
            translated = tl.translate(raw_act)  # (1, seq, dim_b), norm ≈ 0.85

        pred = inject_and_generate(
            model_b, tok_b, translated, layer_b,
            question=q, task=_TASK, device=device,
            injection_scale=1.0,
            injection_timing="prefill_only",
            injection_mode="replace_scale_corrected",
            injection_alpha=alpha,
            context=ctx, full_input=True,
        )
        predictions.append(pred)

        if (i + 1) % 100 == 0:
            logger.info("  [sc_%s seed=%d] %d/%d", direction, seed, i + 1, len(samples))

    elapsed = round(time.time() - t0, 1)

    n_correct = sum(
        evaluate_multi_hop(pred, s["answer"]).get("correct", False)
        for pred, s in zip(predictions, samples)
    )
    acc = n_correct / len(samples)
    logger.info("    acc=%.4f  elapsed=%.1fs", acc, elapsed)

    out = {
        "condition": f"scale_corrected_{direction}",
        "seed": seed,
        "direction": direction,
        "injection_mode": "replace_scale_corrected",
        "injection_alpha": alpha,
        "full_input": True,
        "n_samples": len(samples),
        "acc": acc,
        "elapsed_sec": elapsed,
        "checkpoint": str(ckpt),
        "layer_a": layer_a,
        "layer_b": layer_b,
        "model_a": cfg["sender_id"],
        "model_b": cfg["receiver_id"],
    }

    # Filename includes alpha only for Phase B (alpha < 1.0)
    alpha_tag = "" if alpha == 1.0 else f"_alpha{int(alpha * 100):03d}"
    out_path = output_dir / f"m7_scale_corrected_{direction}_seed{seed}{alpha_tag}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info("  Saved → %s", out_path)

    return out


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry-point for a single M7 evaluation run."""
    p = argparse.ArgumentParser(
        description="M7: scale-corrected replacement injection evaluation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="See module docstring for full usage examples.",
    )
    p.add_argument(
        "--direction", choices=["fwd", "rev"], required=True,
        help="Translation direction: fwd (160M→410M) or rev (410M→160M).",
    )
    p.add_argument(
        "--seed", type=int, required=True, choices=[42, 123, 456],
        help="Random seed (determines which M6 checkpoint to load).",
    )
    p.add_argument(
        "--ckpt-dir", type=Path, default=Path("results/phase1/checkpoints"),
        help="Directory with m6_translation_*.pt checkpoints.",
    )
    p.add_argument(
        "--output-dir", type=Path, default=Path("results/phase1"),
        help="Directory for output JSON files.",
    )
    p.add_argument(
        "--device", default="auto",
        help="Device: cuda / cpu / auto (default: auto).",
    )
    p.add_argument(
        "--test-file", type=Path, default=_TEST_FILE,
        help="Path to test JSONL file.",
    )
    p.add_argument(
        "--alpha", type=float, default=1.0,
        help=(
            "Blending coefficient for scale-corrected injection. "
            "1.0 (default) = full replacement. "
            "0 < alpha < 1 = linear mix α×corrected + (1-α)×original (Phase B)."
        ),
    )
    args = p.parse_args(argv)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    logger.info("Device: %s  alpha=%.2f", device, args.alpha)

    run_scale_corrected(
        direction=args.direction,
        seed=args.seed,
        ckpt_dir=args.ckpt_dir,
        output_dir=args.output_dir,
        device=device,
        test_file=args.test_file,
        alpha=args.alpha,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
