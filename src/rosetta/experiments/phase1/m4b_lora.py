"""M4b: LoRA shared-basis alignment experiment.

Experiment sequence:
  T0 — LoRA alignment training (both models, frozen translation layer)
  T1 — No-inject baseline after LoRA (capability retention check: AC2)
  T2 — P2 translation-layer evaluation after LoRA (H2 main test: AC3/AC4)

Usage::

    # T0: train LoRA alignment (seed=42, ~1-2h on DirectML)
    python -m rosetta.experiments.phase1.m4b_lora --phase train --seed 42 --device dml

    # T1: baseline after LoRA
    python -m rosetta.experiments.phase1.m4b_lora --phase baseline --seed 42 --device dml

    # T2: P2 evaluation with LoRA-aligned models (3 seeds)
    python -m rosetta.experiments.phase1.m4b_lora --phase eval --seed 42 --device dml
    python -m rosetta.experiments.phase1.m4b_lora --phase eval --seed 123 --device dml
    python -m rosetta.experiments.phase1.m4b_lora --phase eval --seed 456 --device dml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

# Project imports
from rosetta.experiments.phase1.m2_baseline import load_model_and_tokenizer
from rosetta.alignment.lora_align import train_lora_alignment, _load_alignment_texts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_RESULTS = Path("results/phase1")
_CKPT_DIR = _RESULTS / "checkpoints"
_DATA = Path("data/tasks/multi_hop_reasoning")
_CORPUS = Path("data/corpus")

_TIER = "tier2"
_SENDER = "EleutherAI/pythia-160m"
_RECEIVER = "EleutherAI/pythia-410m"
_LORA_RANK = 8
_LORA_ALPHA = 16
_LAMBDA_ALIGN = 0.01
_EPOCHS = 7
_LR = 2e-5
_N_TEXTS = 1000
_SCALE = 0.01       # best scale from M4 T4
_NULL_RELAY_ACC = 0.1600   # M4-Enhance baseline for AC4
_BASELINE_ACC = 0.0798     # M4-Enhance for AC2 reference


def _lora_ckpt_tag(seed: int) -> str:
    # Encode epochs in the tag so each (rank, epochs, seed) combination
    # maps to a unique file — prevents silent overwriting across runs.
    return f"m4b_lora_tier2_r{_LORA_RANK}_e{_EPOCHS}_s{seed}"


def _translation_ckpt(seed: int) -> Path:
    """Return the M4 translation-layer checkpoint for a given seed."""
    return _CKPT_DIR / f"m4_P1b_tier2_mlp_1hidden_n5000_s{seed}.pt"


# ---------------------------------------------------------------------------
# T0: LoRA alignment training
# ---------------------------------------------------------------------------

def phase_train(args: argparse.Namespace) -> None:
    """Train LoRA adapters to align model_a and model_b representations."""
    import os
    seed = args.seed
    device = args.device

    logger.info("=" * 60)
    logger.info("T0: LoRA Alignment Training  (seed=%d)", seed)
    logger.info("=" * 60)

    # CPU thread tuning only matters when device is CPU.
    if device == "cpu":
        import os
        n_intra = max(1, (os.cpu_count() or 4) - 1)
        torch.set_num_threads(n_intra)
        torch.set_num_interop_threads(2)
        logger.info("CPU threads: intra=%d  inter=%d",
                    torch.get_num_threads(), torch.get_num_interop_threads())
    else:
        logger.info("Device: %s (GPU — CPU thread settings skipped)", device)

    # Check translation layer checkpoint
    tl_ckpt = _translation_ckpt(seed)
    if not tl_ckpt.exists():
        # Fall back to seed=42 checkpoint if the seed-specific one is missing
        tl_ckpt_42 = _translation_ckpt(42)
        if tl_ckpt_42.exists():
            logger.warning("Translation ckpt for seed=%d not found; using seed=42 fallback.", seed)
            tl_ckpt = tl_ckpt_42
        else:
            logger.error("No translation checkpoint found. Run M4 P1 training first.")
            sys.exit(1)

    logger.info("Translation layer checkpoint: %s", tl_ckpt)

    # Load models
    logger.info("Loading models ...")
    model_a, tok_a = load_model_and_tokenizer(_SENDER, device=device)
    model_b, tok_b = load_model_and_tokenizer(_RECEIVER, device=device)

    # Load alignment texts — corpus-only (Wikitext).
    # test_enhanced.jsonl is the evaluation set; never use it for training.
    # Using eval contexts for LoRA training would cause data contamination:
    # LoRA could over-fit to those specific linguistic patterns and produce
    # artificially high accuracy on the held-out test, masking true H2 signal.
    texts = _load_alignment_texts(
        enhanced_jsonl=None,       # intentionally excluded — eval set
        corpus_dir=_CORPUS,
        n_texts=_N_TEXTS,
    )
    logger.info("Alignment texts (corpus-only, no eval contamination): %d", len(texts))

    # Run alignment training
    result = train_lora_alignment(
        model_a=model_a,
        model_b=model_b,
        tokenizer_a=tok_a,
        tokenizer_b=tok_b,
        translation_ckpt=tl_ckpt,
        enhanced_jsonl=None,       # intentionally excluded — eval set
        corpus_dir=_CORPUS,
        n_texts=_N_TEXTS,
        lora_rank=_LORA_RANK,
        lora_alpha=_LORA_ALPHA,
        lambda_align=_LAMBDA_ALIGN,
        epochs=_EPOCHS,
        learning_rate=_LR,
        batch_size=args.batch_size,
        device=device,
        checkpoint_dir=_CKPT_DIR,
        seed=seed,
    )

    # Save training summary
    summary = {
        "phase": "T0_lora_train",
        "seed": seed,
        "lora_rank": _LORA_RANK,
        "lora_alpha": _LORA_ALPHA,
        "lambda_align": _LAMBDA_ALIGN,
        "epochs": _EPOCHS,
        "learning_rate": _LR,
        "n_texts": len(texts),
        "loss_history": result.loss_history,
        "converged": result.converged,
        "checkpoint_a": result.checkpoint_a,
        "checkpoint_b": result.checkpoint_b,
    }
    out_path = _RESULTS / f"m4b_T0_lora_train_{_TIER}_e{_EPOCHS}_s{seed}.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Saved training summary: %s", out_path)

    # AC1 check
    if result.converged:
        logger.info("AC1 PASS: loss converged (%.6f → %.6f)",
                    result.loss_history[-2] if len(result.loss_history) > 1 else float("nan"),
                    result.loss_history[-1])
    else:
        logger.warning("AC1 WARNING: loss did NOT converge. Consider more epochs or lower lr.")

    logger.info("T0 complete. LoRA checkpoints saved.")


# ---------------------------------------------------------------------------
# T1: Baseline after LoRA (capability retention)
# ---------------------------------------------------------------------------

def phase_baseline(args: argparse.Namespace) -> None:
    """Evaluate no-inject baseline with LoRA-aligned models (AC2 check)."""
    seed = args.seed
    device = args.device
    tag = _lora_ckpt_tag(seed)

    logger.info("=" * 60)
    logger.info("T1: LoRA Baseline Evaluation (seed=%d)", seed)
    logger.info("=" * 60)

    ckpt_a = _CKPT_DIR / f"{tag}_a.pt"
    ckpt_b = _CKPT_DIR / f"{tag}_b.pt"
    if not ckpt_a.exists() or not ckpt_b.exists():
        logger.error("LoRA checkpoints not found. Run --phase train first.")
        sys.exit(1)

    # Delegate to m35_diagnostic with --lora-a-ckpt / --lora-b-ckpt flags
    # (these args are added to m35_diagnostic in T2 integration)
    cmd = [
        sys.executable, "-m", "rosetta.experiments.phase1.m35_diagnostic",
        "--phase", "p0",
        "--task", "multi_hop",
        "--tier", _TIER,
        "--seed", str(seed),
        "--prefix", f"m4b_T1_baseline_e{_EPOCHS}",
        "--test-file", str(_DATA / "test_enhanced.jsonl"),
        "--lora-a-ckpt", str(ckpt_a),
        "--lora-b-ckpt", str(ckpt_b),
        "--device", device,
    ]
    import subprocess
    logger.info("Running: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=False)
    if proc.returncode != 0:
        logger.error("Baseline evaluation failed (exit %d).", proc.returncode)
        sys.exit(proc.returncode)

    # Load result and check AC2
    # m35_diagnostic saves the P0 baseline condition as: {prefix}_P0_baseline_{tier}_multi_hop_{seed}.json
    result_path = _RESULTS / f"m4b_T1_baseline_e{_EPOCHS}_P0_baseline_{_TIER}_multi_hop_{seed}.json"
    if result_path.exists():
        data = json.loads(result_path.read_text(encoding="utf-8"))
        metric = data.get("metric", {})
        acc = metric.get("contains_acc", 0.0) if isinstance(metric, dict) else float(metric)
        threshold = _BASELINE_ACC * 0.95
        status = "PASS" if acc >= threshold else "FAIL"
        logger.info("AC2 %s: LoRA baseline acc=%.4f (threshold=%.4f, original=%.4f)",
                    status, acc, threshold, _BASELINE_ACC)
    else:
        logger.warning("Result file not found at %s — check subprocess output.", result_path)


# ---------------------------------------------------------------------------
# T2: P2 evaluation with LoRA (H2 main test)
# ---------------------------------------------------------------------------

def phase_eval(args: argparse.Namespace) -> None:
    """Evaluate P2 translation layer with LoRA-aligned models (AC3/AC4 check)."""
    seed = args.seed
    device = args.device
    tag = _lora_ckpt_tag(seed)

    logger.info("=" * 60)
    logger.info("T2: LoRA P2 Evaluation (seed=%d)", seed)
    logger.info("=" * 60)

    ckpt_a = _CKPT_DIR / f"{tag}_a.pt"
    ckpt_b = _CKPT_DIR / f"{tag}_b.pt"
    if not ckpt_a.exists() or not ckpt_b.exists():
        logger.error("LoRA checkpoints not found. Run --phase train first.")
        sys.exit(1)

    cmd = [
        sys.executable, "-m", "rosetta.experiments.phase1.m35_diagnostic",
        "--phase", "p2",
        "--task", "multi_hop",
        "--tier", _TIER,
        "--seed", str(seed),
        "--prefix", f"m4b_T2_e{_EPOCHS}",
        "--null-baseline", str(_NULL_RELAY_ACC),
        "--test-file", str(_DATA / "test_enhanced.jsonl"),
        "--p2-single-scale", str(_SCALE),
        "--lora-a-ckpt", str(ckpt_a),
        "--lora-b-ckpt", str(ckpt_b),
        "--device", device,
    ]
    import subprocess
    logger.info("Running: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=False)
    if proc.returncode != 0:
        logger.error("P2 evaluation failed (exit %d).", proc.returncode)
        sys.exit(proc.returncode)

    # Load result and report
    # m35_diagnostic names the file as: {prefix}_P2_P1b_scale{scale}_{tier}_multi_hop_{seed}.json
    result_path = _RESULTS / f"m4b_T2_e{_EPOCHS}_P2_P1b_scale{_SCALE}_{_TIER}_multi_hop_{seed}.json"
    if result_path.exists():
        data = json.loads(result_path.read_text(encoding="utf-8"))
        metric = data.get("metric", {})
        acc = metric.get("contains_acc", 0.0) if isinstance(metric, dict) else float(metric)
        logger.info("T2 P2 acc=%.4f  Δ_vs_null=+%.4f  Δ_vs_baseline=+%.4f",
                    acc, acc - _NULL_RELAY_ACC, acc - _BASELINE_ACC)
        if acc >= _NULL_RELAY_ACC:
            logger.info("AC4 PASS candidate: acc >= null relay (%.4f).", _NULL_RELAY_ACC)
        elif acc > _BASELINE_ACC:
            logger.info("AC3 candidate: acc > no-LoRA P2 (%.4f).", _BASELINE_ACC)
        else:
            logger.info("No AC passed: acc=%.4f <= baseline %.4f.", acc, _BASELINE_ACC)
    else:
        logger.warning("Result file not found at %s — check subprocess output.", result_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="M4b LoRA alignment experiment")
    parser.add_argument("--phase", choices=["train", "baseline", "eval"],
                        required=True, help="Experiment phase to run")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", default="dml",
                        help="Device: dml (DirectML), cuda, or cpu")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size for LoRA training (reduce if OOM)")
    args = parser.parse_args()

    # Resolve device string: "dml" → actual DirectML device ("privateuseone:0")
    if args.device == "dml":
        try:
            import torch_directml  # type: ignore[import]
            args.device = str(torch_directml.device())
        except ImportError:
            logger.error("--device dml requires torch-directml. pip install torch-directml")
            sys.exit(1)

    if args.phase == "train":
        phase_train(args)
    elif args.phase == "baseline":
        phase_baseline(args)
    elif args.phase == "eval":
        phase_eval(args)


if __name__ == "__main__":
    main()
