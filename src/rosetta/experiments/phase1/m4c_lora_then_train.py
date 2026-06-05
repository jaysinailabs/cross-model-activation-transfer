"""M4c: LoRA-then-retrain — correct-order validation of H2.

M4b trained translation layer T *before* LoRA alignment, causing a
representation-space mismatch.  M4c fixes the order:

  1. Load M4b LoRA checkpoints (e7) or continue-train to e12.
  2. Retrain T_new on LoRA-aligned activations.
  3. Evaluate: inject T_new(h_a_LoRA) into model_b_LoRA → downstream accuracy.

Phases:
  retrain_t     — Extract post-LoRA activations + train T_new (core logic)
  continue_lora — Warm-start from e7 LoRA, train additional epochs → e12
  baseline      — No-inject baseline with LoRA models (AC2 check)
  eval          — P2 evaluation with LoRA + T_new (H2 main test)

Usage::

    # Round 1: retrain T on M4b e7 LoRA
    python -m rosetta.experiments.phase1.m4c_lora_then_train \\
      --phase retrain_t --seed 42 --lora-epoch 7 --t-epochs 50 --device cuda

    # Round 1: evaluate
    python -m rosetta.experiments.phase1.m4c_lora_then_train \\
      --phase eval --seed 42 --lora-epoch 7 --device cuda

    # Round 2: continue LoRA e7→e12
    python -m rosetta.experiments.phase1.m4c_lora_then_train \\
      --phase continue_lora --seed 42 --lora-epoch 7 --add-epochs 5 --device cuda

    # Round 2: retrain T on e12 LoRA + evaluate
    python -m rosetta.experiments.phase1.m4c_lora_then_train \\
      --phase retrain_t --seed 42 --lora-epoch 12 --t-epochs 50 --device cuda
    python -m rosetta.experiments.phase1.m4c_lora_then_train \\
      --phase eval --seed 42 --lora-epoch 12 --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

from rosetta.experiments.phase1.m2_baseline import load_model_and_tokenizer
from rosetta.experiments.phase1.m3_translation import load_train_texts
from rosetta.translation.translation_layer import (
    TranslationLayer,
    extract_activation_pairs,
    train_translation_layer,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — must match M4b / M4 P1b exactly to ensure single-variable design
# ---------------------------------------------------------------------------

_RESULTS = Path("results/phase1")
_CKPT_DIR = _RESULTS / "checkpoints"
_DATA = Path("data/tasks/multi_hop_reasoning")
_CORPUS_PATH = Path("data/corpus/wikitext103_clean.txt")

_TIER = "tier2"
_SENDER = "EleutherAI/pythia-160m"
_RECEIVER = "EleutherAI/pythia-410m"

# Layer extraction: relative depth 67% → layer 8/12 (sender), 16/24 (receiver)
_LAYER_A = "gpt_neox.layers.8"
_LAYER_B = "gpt_neox.layers.16"

# Translation layer architecture — MUST match M4 P1b (single-variable principle)
_T_ARCH = "mlp_1hidden"
_T_HIDDEN_DIM = 1024       # max(768, 1024) per M4 P1b config
_T_NORMALIZE = False       # P1b config
_T_POOLING = "last_token"  # P1b config; matches eval pipeline
_T_N_TRAIN = 5000          # wikitext corpus size, same as P1b
_T_LR = 1e-4               # default from train_translation_layer

# LoRA config — must match M4b
_LORA_RANK = 8
_LORA_ALPHA = 16

# LoRA alignment training config — for continue_lora phase
_LAMBDA_ALIGN = 0.01
_LORA_LR = 2e-5
_LORA_N_TEXTS = 1000

# Evaluation constants
_SCALE = 0.01              # best scale from M4 T4
_NULL_RELAY_ACC = 0.1600   # E3a n=514
_BASELINE_ACC = 0.0798     # E3a n=514 (no-inject, no-LoRA)

# Sender dim / receiver dim for TranslationLayer
_DIM_A = 768
_DIM_B = 1024


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _lora_ckpt_paths(epoch: int, seed: int) -> tuple[Path, Path]:
    """Return (path_a, path_b) for LoRA checkpoints at given epoch/seed."""
    if epoch == 7:
        # M4b e7 checkpoints
        tag = f"m4b_lora_tier2_r{_LORA_RANK}_e{epoch}_s{seed}"
    else:
        # M4c continued training checkpoints
        tag = f"m4c_lora_tier2_r{_LORA_RANK}_e{epoch}_s{seed}"
    return _CKPT_DIR / f"{tag}_a.pt", _CKPT_DIR / f"{tag}_b.pt"


def _t_new_ckpt_path(lora_epoch: int, seed: int) -> Path:
    """Return checkpoint path for T_new trained on LoRA-e{epoch} activations."""
    return _CKPT_DIR / f"m4c_T_lora{lora_epoch}_tier2_mlp_1hidden_s{seed}.pt"


def _load_lora_weights(
    model: torch.nn.Module,
    ckpt_path: str | Path,
    lora_rank: int = _LORA_RANK,
    lora_alpha: int = _LORA_ALPHA,
) -> torch.nn.Module:
    """Apply PEFT LoRA config and load saved delta weights.

    Identical logic to m35_diagnostic._load_lora_weights — duplicated here
    to avoid importing the full m35 module (which triggers heavy global init).
    """
    from peft import get_peft_model, LoraConfig, TaskType

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=["query_key_value", "dense"],
        lora_dropout=0.0,
        bias="none",
        inference_mode=True,
    )
    model_lora = get_peft_model(model, lora_cfg)
    state = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    model_lora.load_state_dict(state, strict=False)
    model_lora.eval()
    logger.info("Loaded LoRA weights from %s", Path(ckpt_path).name)
    return model_lora


# ---------------------------------------------------------------------------
# Phase: retrain_t — core M4c logic
# ---------------------------------------------------------------------------

def phase_retrain_t(args: argparse.Namespace) -> None:
    """Extract post-LoRA activations and train T_new."""
    seed = args.seed
    device = args.device
    lora_epoch = args.lora_epoch
    t_epochs = args.t_epochs

    logger.info("=" * 60)
    logger.info("M4c retrain_t: LoRA e%d, seed=%d, T epochs=%d",
                lora_epoch, seed, t_epochs)
    logger.info("=" * 60)

    # 1. Verify LoRA checkpoints exist
    ckpt_a, ckpt_b = _lora_ckpt_paths(lora_epoch, seed)
    if not ckpt_a.exists() or not ckpt_b.exists():
        logger.error("LoRA checkpoints not found:\n  %s\n  %s", ckpt_a, ckpt_b)
        logger.error("Run M4b T0 (--phase train) or M4c continue_lora first.")
        sys.exit(1)

    # 2. Load models + apply LoRA
    logger.info("Loading models and applying LoRA e%d ...", lora_epoch)
    model_a, tok_a = load_model_and_tokenizer(_SENDER, device=device)
    model_b, tok_b = load_model_and_tokenizer(_RECEIVER, device=device)
    model_a = _load_lora_weights(model_a, ckpt_a)
    model_b = _load_lora_weights(model_b, ckpt_b)

    # 3. Extract LoRA-space activation pairs (same corpus as T_orig)
    logger.info("Extracting activation pairs (n=%d, pooling=%s) ...",
                _T_N_TRAIN, _T_POOLING)
    texts = load_train_texts(
        n_samples=_T_N_TRAIN, seed=seed, corpus_path=_CORPUS_PATH,
    )
    acts_a, acts_b = extract_activation_pairs(
        model_a, tok_a, model_b, tok_b,
        texts=texts,
        layer_name_a=_LAYER_A,
        layer_name_b=_LAYER_B,
        device=device,
        pooling_mode=_T_POOLING,
    )
    logger.info("Activation shapes: a=%s, b=%s", acts_a.shape, acts_b.shape)

    # Record hidden norms for M4b-lessons item #1 (model_b drift diagnostic)
    norm_b_lora = acts_b.float().norm(dim=-1).mean().item()
    logger.info("LoRA model_b hidden norm (mean): %.4f", norm_b_lora)

    # Free GPU memory — T_new training happens on CPU (model is tiny ~2MB)
    del model_a, model_b, tok_a, tok_b
    torch.cuda.empty_cache() if device != "cpu" else None

    # 4. Train T_new with identical architecture to T_orig
    logger.info("Training T_new (arch=%s, hidden=%d, normalize=%s) ...",
                _T_ARCH, _T_HIDDEN_DIM, _T_NORMALIZE)
    tl_new = TranslationLayer(
        _DIM_A, _DIM_B,
        arch=_T_ARCH,
        hidden_dim=_T_HIDDEN_DIM,
        normalize=_T_NORMALIZE,
    )
    ckpt_path = _t_new_ckpt_path(lora_epoch, seed)
    stats = train_translation_layer(
        tl_new, acts_a, acts_b,
        epochs=t_epochs,
        lr=_T_LR,
        device="cpu",
        checkpoint_path=ckpt_path,
        seed=seed,
    )

    # 5. AC1 check: val_loss convergence (CV of last 3 epochs < 0.02)
    val_losses = stats.get("val_losses", [])
    if len(val_losses) >= 3:
        last3 = val_losses[-3:]
        mean_l3 = np.mean(last3)
        std_l3 = np.std(last3, ddof=1)
        cv = std_l3 / mean_l3 if mean_l3 > 0 else float("inf")
        ac1_pass = cv < 0.02
        logger.info("AC1 check: last 3 val_loss=[%.6f, %.6f, %.6f] CV=%.4f → %s",
                    last3[0], last3[1], last3[2], cv,
                    "PASS" if ac1_pass else "FAIL")
    else:
        cv = float("nan")
        ac1_pass = False
        logger.warning("AC1 check: fewer than 3 val_loss entries, cannot verify")

    # 6. Save training summary
    summary = {
        "phase": "retrain_t",
        "lora_epoch": lora_epoch,
        "seed": seed,
        "t_epochs": t_epochs,
        "t_arch": _T_ARCH,
        "t_hidden_dim": _T_HIDDEN_DIM,
        "t_normalize": _T_NORMALIZE,
        "pooling_mode": _T_POOLING,
        "n_train_texts": len(texts),
        "train_losses": [float(v) for v in stats.get("train_losses", [])],
        "val_losses": [float(v) for v in val_losses],
        "best_epoch": int(stats["best_epoch"]) if stats.get("best_epoch") is not None else None,
        "best_val_loss": float(stats["best_val_loss"]) if stats.get("best_val_loss") is not None else None,
        "elapsed_sec": float(stats["elapsed_sec"]) if stats.get("elapsed_sec") is not None else None,
        "ac1_cv": float(cv) if cv == cv else None,  # nan-safe
        "ac1_pass": bool(ac1_pass),
        "norm_b_lora_mean": float(norm_b_lora),
        "checkpoint": str(ckpt_path),
    }
    out_path = _RESULTS / f"m4c_T_train_lora{lora_epoch}_{_TIER}_s{seed}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Saved T_new training summary: %s", out_path.name)
    logger.info("T_new checkpoint: %s", ckpt_path.name)


# ---------------------------------------------------------------------------
# Phase: continue_lora — warm-start from e7 → e12
# ---------------------------------------------------------------------------

def phase_continue_lora(args: argparse.Namespace) -> None:
    """Continue LoRA alignment training from an existing checkpoint."""
    seed = args.seed
    device = args.device
    base_epoch = args.lora_epoch
    add_epochs = args.add_epochs
    total_epochs = base_epoch + add_epochs

    logger.info("=" * 60)
    logger.info("M4c continue_lora: e%d → e%d, seed=%d",
                base_epoch, total_epochs, seed)
    logger.info("=" * 60)

    # Verify source checkpoints
    src_a, src_b = _lora_ckpt_paths(base_epoch, seed)
    if not src_a.exists() or not src_b.exists():
        logger.error("Source LoRA checkpoints not found:\n  %s\n  %s", src_a, src_b)
        sys.exit(1)

    # Translation layer checkpoint (T_old, frozen — same as M4b alignment loss)
    tl_ckpt = _CKPT_DIR / f"m4_P1b_tier2_mlp_1hidden_n5000_s{seed}.pt"
    if not tl_ckpt.exists():
        tl_ckpt_42 = _CKPT_DIR / "m4_P1b_tier2_mlp_1hidden_n5000_s42.pt"
        if tl_ckpt_42.exists():
            logger.warning("T_old for seed=%d not found, using seed=42 fallback.", seed)
            tl_ckpt = tl_ckpt_42
        else:
            logger.error("No T_old checkpoint found. Need M4 P1b checkpoint.")
            sys.exit(1)

    # Load models
    logger.info("Loading models ...")
    model_a, tok_a = load_model_and_tokenizer(_SENDER, device=device)
    model_b, tok_b = load_model_and_tokenizer(_RECEIVER, device=device)

    # Build aligner and load existing LoRA weights (warm-start)
    from rosetta.alignment.lora_align import LoraAligner, _load_alignment_texts

    aligner = LoraAligner.from_config(
        model_a, model_b, tok_a, tok_b,
        translation_ckpt=tl_ckpt,
        lora_rank=_LORA_RANK,
        lora_alpha=_LORA_ALPHA,
        lambda_align=_LAMBDA_ALIGN,
        device=device,
    )

    # Load M4b e7 weights (warm-start)
    aligner.load(str(src_a), str(src_b))
    logger.info("Warm-started from e%d LoRA weights", base_epoch)

    # Load alignment texts (corpus-only, no eval data)
    texts = _load_alignment_texts(
        enhanced_jsonl=None,
        corpus_dir=Path("data/corpus"),
        n_texts=_LORA_N_TEXTS,
    )
    logger.info("Alignment texts: %d", len(texts))

    # Continue training
    result = aligner.train(
        texts,
        epochs=add_epochs,
        learning_rate=_LORA_LR,
        batch_size=args.batch_size,
    )

    # Save LoRA weights manually (bypass LoraAligner.save() m4b prefix)
    _CKPT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"m4c_lora_tier2_r{_LORA_RANK}_e{total_epochs}_s{seed}"
    lora_state_a = {k: v for k, v in aligner.model_a.state_dict().items()
                    if "lora_" in k}
    lora_state_b = {k: v for k, v in aligner.model_b.state_dict().items()
                    if "lora_" in k}
    path_a = _CKPT_DIR / f"{tag}_a.pt"
    path_b = _CKPT_DIR / f"{tag}_b.pt"
    torch.save(lora_state_a, path_a)
    torch.save(lora_state_b, path_b)
    logger.info("Saved LoRA e%d: %s / %s", total_epochs, path_a.name, path_b.name)

    # Report convergence (AC1-LoRA: informational, not gating)
    loss_hist = result.loss_history
    if len(loss_hist) >= 2:
        ratio = loss_hist[-1] / loss_hist[-2] if loss_hist[-2] > 0 else float("nan")
        logger.info("LoRA loss ratio (last 2 epochs): %.4f (< 0.95 desired)", ratio)
    logger.info("LoRA loss history: %s",
                [f"{v:.4f}" for v in loss_hist])

    # Save summary
    summary = {
        "phase": "continue_lora",
        "seed": seed,
        "base_epoch": base_epoch,
        "add_epochs": add_epochs,
        "total_epochs": total_epochs,
        "loss_history": loss_hist,
        "converged": result.converged,
        "checkpoint_a": str(path_a),
        "checkpoint_b": str(path_b),
    }
    out_path = _RESULTS / f"m4c_continue_lora_e{total_epochs}_{_TIER}_s{seed}.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Saved continue_lora summary: %s", out_path.name)


# ---------------------------------------------------------------------------
# Phase: baseline — no-inject with LoRA models (AC2)
# ---------------------------------------------------------------------------

def phase_baseline(args: argparse.Namespace) -> None:
    """Evaluate no-inject baseline with LoRA models (AC2 check)."""
    seed = args.seed
    device = args.device
    lora_epoch = args.lora_epoch

    logger.info("=" * 60)
    logger.info("M4c baseline: LoRA e%d, seed=%d", lora_epoch, seed)
    logger.info("=" * 60)

    ckpt_a, ckpt_b = _lora_ckpt_paths(lora_epoch, seed)
    if not ckpt_a.exists() or not ckpt_b.exists():
        logger.error("LoRA checkpoints not found:\n  %s\n  %s", ckpt_a, ckpt_b)
        sys.exit(1)

    import subprocess
    cmd = [
        sys.executable, "-m", "rosetta.experiments.phase1.m35_diagnostic",
        "--phase", "p0",
        "--task", "multi_hop",
        "--tier", _TIER,
        "--seed", str(seed),
        "--prefix", f"m4c_T1_baseline_lora{lora_epoch}",
        "--test-file", str(_DATA / "test_enhanced.jsonl"),
        "--lora-a-ckpt", str(ckpt_a),
        "--lora-b-ckpt", str(ckpt_b),
        "--device", device,
    ]
    logger.info("Running: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=False)
    if proc.returncode != 0:
        logger.error("Baseline evaluation failed (exit %d).", proc.returncode)
        sys.exit(proc.returncode)

    # Load result and check AC2
    result_path = (_RESULTS /
                   f"m4c_T1_baseline_lora{lora_epoch}_P0_baseline_{_TIER}_multi_hop_{seed}.json")
    if result_path.exists():
        data = json.loads(result_path.read_text(encoding="utf-8"))
        metric = data.get("metric", {})
        acc = metric.get("contains_acc", 0.0) if isinstance(metric, dict) else float(metric)
        threshold = _BASELINE_ACC * 0.95  # 0.0758
        status = "PASS" if acc >= threshold else "FAIL"
        logger.info("AC2 %s: LoRA e%d baseline acc=%.4f (threshold=%.4f)",
                    status, lora_epoch, acc, threshold)
    else:
        logger.warning("Result file not found: %s", result_path)


# ---------------------------------------------------------------------------
# Phase: eval — P2 with LoRA + T_new (H2 main test)
# ---------------------------------------------------------------------------

def phase_eval(args: argparse.Namespace) -> None:
    """Evaluate P2 translation with LoRA-aligned models + T_new."""
    seed = args.seed
    device = args.device
    lora_epoch = args.lora_epoch

    logger.info("=" * 60)
    logger.info("M4c eval: LoRA e%d + T_new, seed=%d", lora_epoch, seed)
    logger.info("=" * 60)

    # Verify LoRA checkpoints
    ckpt_a, ckpt_b = _lora_ckpt_paths(lora_epoch, seed)
    if not ckpt_a.exists() or not ckpt_b.exists():
        logger.error("LoRA checkpoints not found:\n  %s\n  %s", ckpt_a, ckpt_b)
        sys.exit(1)

    # Verify T_new checkpoint
    t_new_ckpt = _t_new_ckpt_path(lora_epoch, seed)
    if not t_new_ckpt.exists():
        logger.error("T_new checkpoint not found: %s", t_new_ckpt)
        logger.error("Run --phase retrain_t --lora-epoch %d --seed %d first.",
                      lora_epoch, seed)
        sys.exit(1)

    import subprocess
    cmd = [
        sys.executable, "-m", "rosetta.experiments.phase1.m35_diagnostic",
        "--phase", "p2",
        "--task", "multi_hop",
        "--tier", _TIER,
        "--seed", str(seed),
        "--prefix", f"m4c_eval_lora{lora_epoch}",
        "--null-baseline", str(_NULL_RELAY_ACC),
        "--test-file", str(_DATA / "test_enhanced.jsonl"),
        "--p2-single-scale", str(_SCALE),
        "--lora-a-ckpt", str(ckpt_a),
        "--lora-b-ckpt", str(ckpt_b),
        "--translation-ckpt", str(t_new_ckpt),
        "--device", device,
    ]
    logger.info("Running: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=False)
    if proc.returncode != 0:
        logger.error("Eval failed (exit %d).", proc.returncode)
        sys.exit(proc.returncode)

    # Load result and report
    result_path = (_RESULTS /
                   f"m4c_eval_lora{lora_epoch}_P2_P1b_scale{_SCALE}_{_TIER}_multi_hop_{seed}.json")
    if result_path.exists():
        data = json.loads(result_path.read_text(encoding="utf-8"))
        metric = data.get("metric", {})
        acc = metric.get("contains_acc", 0.0) if isinstance(metric, dict) else float(metric)

        # Signal strength diagnostic (M4b lessons #5)
        inject_ratio = data.get("inject_ratio", data.get("mean_inject_ratio", "N/A"))

        logger.info("M4c eval result: acc=%.4f  inject_ratio=%s", acc, inject_ratio)
        logger.info("  Δ vs original baseline (0.0798): %+.4f", acc - _BASELINE_ACC)
        logger.info("  Δ vs LoRA baseline (0.0856):     %+.4f", acc - 0.0856)
        logger.info("  Δ vs null relay (0.1600):        %+.4f", acc - _NULL_RELAY_ACC)

        if acc >= _NULL_RELAY_ACC:
            logger.info("AC4 candidate: acc >= null relay!")
        elif acc > 0.0856:
            logger.info("AC3a candidate: acc > LoRA baseline")
        elif acc > _BASELINE_ACC:
            logger.info("AC3b candidate: acc > original baseline")
        else:
            logger.info("No AC passed: acc <= original baseline")
    else:
        logger.warning("Result file not found: %s", result_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="M4c: LoRA-then-retrain T_new (correct-order H2 validation)",
    )
    parser.add_argument(
        "--phase",
        choices=["retrain_t", "continue_lora", "baseline", "eval"],
        required=True,
        help="Experiment phase",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--lora-epoch", type=int, default=7,
        help="LoRA epoch to use (7=M4b, 12=M4c continued)",
    )
    parser.add_argument(
        "--t-epochs", type=int, default=50,
        help="T_new training epochs (retrain_t phase only)",
    )
    parser.add_argument(
        "--add-epochs", type=int, default=5,
        help="Additional LoRA epochs (continue_lora phase only)",
    )
    parser.add_argument(
        "--device", default="cuda",
        help="Device: cuda, dml, or cpu",
    )
    parser.add_argument(
        "--batch-size", type=int, default=4,
        help="Batch size for LoRA training (continue_lora only)",
    )
    args = parser.parse_args()

    # Resolve device string
    if args.device == "dml":
        try:
            import torch_directml
            args.device = str(torch_directml.device())
        except ImportError:
            logger.error("--device dml requires torch-directml.")
            sys.exit(1)

    if args.phase == "retrain_t":
        phase_retrain_t(args)
    elif args.phase == "continue_lora":
        phase_continue_lora(args)
    elif args.phase == "baseline":
        phase_baseline(args)
    elif args.phase == "eval":
        phase_eval(args)


if __name__ == "__main__":
    main()
