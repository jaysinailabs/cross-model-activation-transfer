"""M3: Translation-layer communication pipeline.

Group 2 of Phase 1 experiments.  Trains a lightweight translation network to
map Model-A intermediate activations into Model-B's representation space, then
evaluates whether activation-level communication outperforms the NL-relay
baseline from M2 (Group 1).

Pipeline per sample (M3 translated):
    1. Extract mean-pooled hidden state from Model-A at layer L (no text output)
    2. Translate via trained TranslationLayer → vector in Model-B space
    3. Inject translated vector additively at Model-B layer L (residual steering)
    4. Model-B generates answer conditioned on injected context
    5. Evaluate with same contains-match metric as M2

Additional baseline (null relay):
    Same as M2 receiver-only, but relay text = "" (empty string).
    Measures receiver prior knowledge — the floor for any relay method.

Usage:
    # AC1 validation — 5-sample subset:
    python -m rosetta.experiments.phase1.m3_translation --subset 5

    # Full extraction + training + evaluation:
    python -m rosetta.experiments.phase1.m3_translation

    # Null-relay baseline only:
    python -m rosetta.experiments.phase1.m3_translation --null-relay

    # Quick layer scan:
    python -m rosetta.experiments.phase1.m3_translation --layer-relative 0.5

Results saved to:
    results/phase1/m3_translation_{tier}_{task}_{arch}_{seed}.json
    results/phase1/m3_nullrelay_{tier}_{task}_{seed}.json
    results/phase1/checkpoints/m3_{tier}_{arch}.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch

from rosetta.experiments.phase1.m2_baseline import (
    evaluate_instruction_following,
    evaluate_knowledge_relay,
    evaluate_multi_hop,
    get_device,
    load_jsonl,
    load_model_and_tokenizer,
    load_test_split,
)
from rosetta.translation.nl_relay import run_receiver
from rosetta.translation.translation_layer import (
    TranslationLayer,
    extract_activation_pairs,
    inject_and_generate,
    load_translation_layer,
    train_translation_layer,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths and model configurations (mirrors m2_baseline.py)
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parents[4]
_DATA_DIR = _ROOT / "data" / "tasks"
_RESULTS_DIR = _ROOT / "results" / "phase1"
_CHECKPOINTS_DIR = _RESULTS_DIR / "checkpoints"

_TIERS: dict[str, dict[str, Any]] = {
    "tier1": {
        "sender_id": "EleutherAI/pythia-160m",
        "receiver_id": "EleutherAI/pythia-160m-deduped",
        "sender_hidden_dim": 768,
        "receiver_hidden_dim": 768,
        "sender_num_layers": 12,
        "receiver_num_layers": 12,
        "sender_layer_prefix": "gpt_neox.layers",
        "receiver_layer_prefix": "gpt_neox.layers",
    },
    "tier2": {
        "sender_id": "EleutherAI/pythia-160m",
        "receiver_id": "EleutherAI/pythia-410m",
        "sender_hidden_dim": 768,
        "receiver_hidden_dim": 1024,
        "sender_num_layers": 12,
        "receiver_num_layers": 24,
        "sender_layer_prefix": "gpt_neox.layers",
        "receiver_layer_prefix": "gpt_neox.layers",
    },
}

_TASK_DIRS: dict[str, str] = {
    "multi_hop": "multi_hop_reasoning",
    "knowledge_relay": "knowledge_relay",
    "instruction_following": "instruction_following",
}

_DEFAULT_RELATIVE_LAYER = 0.67  # literature: 8-10/12 is optimal for semantic tasks


# ---------------------------------------------------------------------------
# Helper: resolve relative layer index → layer name
# ---------------------------------------------------------------------------


def resolve_layer_name(prefix: str, num_layers: int, relative: float) -> str:
    """Convert relative layer position to a dot-separated layer name.

    Args:
        prefix: Layer name prefix (e.g. ``"gpt_neox.layers"``).
        num_layers: Total number of layers in the model.
        relative: Relative depth in [0, 1].

    Returns:
        String like ``"gpt_neox.layers.8"``.
    """
    idx = int(relative * num_layers)
    idx = min(idx, num_layers - 1)
    return f"{prefix}.{idx}"


# ---------------------------------------------------------------------------
# Data loading: training texts for activation extraction
# ---------------------------------------------------------------------------


def load_train_texts(
    n_samples: int = 30000,
    seed: int = 42,
    corpus_path: Path | str | None = None,
) -> list[str]:
    """Collect plain-text training samples for activation pair extraction.

    Two modes:
    - ``corpus_path`` provided: read one-paragraph-per-line from an external
      file (e.g. Wikitext-103 cleaned by ``scripts/download_corpus.py``).
    - ``corpus_path`` is None: fall back to aggregating context/passage/
      instruction fields from all task training splits (legacy v1 mode, ~1k texts).

    Args:
        n_samples: Maximum number of texts to return.
        seed: Random seed for sampling when the source exceeds n_samples.
        corpus_path: Path to a UTF-8 text file with one document per line.

    Returns:
        List of text strings.
    """
    if corpus_path is not None:
        corpus_path = Path(corpus_path)
        if not corpus_path.exists():
            raise FileNotFoundError(
                f"Corpus file not found: {corpus_path}\n"
                "Run: python scripts/download_corpus.py"
            )
        with open(corpus_path, encoding="utf-8") as f:
            texts = [line.rstrip("\n") for line in f if line.strip()]
        logger.info("Loaded %d lines from external corpus: %s", len(texts), corpus_path)
    else:
        texts = []
        for task_key, task_dir in _TASK_DIRS.items():
            train_path = _DATA_DIR / task_dir / "train.jsonl"
            if not train_path.exists():
                logger.warning("Train split not found: %s — skipping", train_path)
                continue
            samples = load_jsonl(train_path)
            for s in samples:
                if task_key == "instruction_following":
                    texts.append(
                        f"Instruction: {s['instruction']}\nInput: {s['input_text']}"
                    )
                elif task_key == "knowledge_relay":
                    texts.append(s["passage"])
                else:
                    texts.append(s["context"])

        if not texts:
            raise RuntimeError(
                "No training texts found.  Run m1_data_generation.py first, "
                "or pass --corpus-path to use an external corpus."
            )
        logger.info("Loaded %d texts from task train splits (v1 mode)", len(texts))

    rng = random.Random(seed)
    if len(texts) > n_samples:
        texts = rng.sample(texts, n_samples)
        logger.info("Sampled %d texts (seed=%d)", len(texts), seed)
    else:
        logger.info(
            "Using all %d texts (requested %d; no sampling needed)", len(texts), n_samples
        )
    return texts


# ---------------------------------------------------------------------------
# Null-relay baseline: receiver only, empty relay text
# ---------------------------------------------------------------------------


def run_sample_null_relay(
    sample: dict,
    model_b,
    tokenizer_b,
    task: str,
) -> dict[str, Any]:
    """Run the null-relay baseline: empty string relay → receiver only.

    This measures receiver prior knowledge with zero information from the
    sender.  If null-relay ≈ NL-relay (M2), it confirms that M2 accuracy
    was entirely from receiver prior and not from the relay.

    Args:
        sample: Task sample dict.
        model_b/tokenizer_b: Receiver (Model B).
        task: Task identifier.

    Returns:
        Result dict with source, predicted, eval_result.
    """
    relay = ""
    if task == "instruction_following":
        question = sample["input_text"]
    else:
        question = sample["question"]

    predicted = run_receiver(model_b, tokenizer_b, relay, question, task=task)

    if task == "multi_hop":
        eval_result = evaluate_multi_hop(predicted, sample["answer"])
        source = {
            "question": sample["question"],
            "answer": sample["answer"],
            "context": sample["context"],
        }
    elif task == "knowledge_relay":
        eval_result = evaluate_knowledge_relay(predicted, sample["answer"])
        source = {
            "question": sample["question"],
            "answer": sample["answer"],
            "passage": sample["passage"],
        }
    else:
        eval_result = evaluate_instruction_following(predicted, sample["constraints"])
        source = {
            "instruction": sample["instruction"],
            "input_text": sample["input_text"],
            "constraints": sample["constraints"],
        }

    return {"source": source, "relay": relay, "predicted": predicted, "eval_result": eval_result}


# ---------------------------------------------------------------------------
# Translated pipeline: single sample
# ---------------------------------------------------------------------------


def run_sample_translated(
    sample: dict,
    model_a,
    tokenizer_a,
    model_b,
    tokenizer_b,
    tl: TranslationLayer,
    layer_name_a: str,
    layer_name_b: str,
    task: str,
    device: str,
) -> dict[str, Any]:
    """Run the M3 translation pipeline on one sample.

    Extracts sender activations for the source text, translates to receiver
    space, injects additively at layer L, and generates the answer.

    Args:
        sample: Task sample dict.
        model_a/tokenizer_a: Sender (Model A).
        model_b/tokenizer_b: Receiver (Model B).
        tl: Trained TranslationLayer.
        layer_name_a: Extraction layer in model_a.
        layer_name_b: Injection layer in model_b.
        task: Task identifier.
        device: Torch device string.

    Returns:
        Result dict with source, sender_layer, receiver_layer,
        predicted, eval_result.
    """
    from rosetta.models.activation_extractor import ActivationExtractor

    # Build the sender input text
    if task == "instruction_following":
        sender_text = (
            f"Instruction: {sample['instruction']}\nInput: {sample['input_text']}"
        )
        question = sample["input_text"]
    elif task == "knowledge_relay":
        sender_text = sample["passage"]
        question = sample["question"]
    else:
        sender_text = sample["context"]
        question = sample["question"]

    # Extract sender activation (mean-pool over tokens)
    enc_a = tokenizer_a(
        sender_text,
        return_tensors="pt",
        truncation=True,
        max_length=256,
    ).to(device)
    with torch.no_grad(), ActivationExtractor(model_a, layer_name_a) as ext:
        model_a(**enc_a)
    act_a = ext.activation  # (1, seq, hidden)
    mask = enc_a["attention_mask"].unsqueeze(-1).float()
    pooled_a = (act_a * mask).sum(1) / mask.sum(1)  # (1, hidden_a)

    # Translate
    tl.eval()
    with torch.no_grad():
        translated = tl.translate(pooled_a.squeeze(0))  # (hidden_b,)

    # Inject and generate
    predicted = inject_and_generate(
        model_b,
        tokenizer_b,
        translated,
        layer_name_b,
        question,
        task,
        device=device,
    )

    # Evaluate
    if task == "multi_hop":
        eval_result = evaluate_multi_hop(predicted, sample["answer"])
        source = {
            "context": sample["context"],
            "question": sample["question"],
            "answer": sample["answer"],
        }
    elif task == "knowledge_relay":
        eval_result = evaluate_knowledge_relay(predicted, sample["answer"])
        source = {
            "passage": sample["passage"],
            "question": sample["question"],
            "answer": sample["answer"],
        }
    else:
        eval_result = evaluate_instruction_following(predicted, sample["constraints"])
        source = {
            "instruction": sample["instruction"],
            "input_text": sample["input_text"],
            "constraints": sample["constraints"],
        }

    return {
        "source": source,
        "sender_layer": layer_name_a,
        "receiver_layer": layer_name_b,
        "predicted": predicted,
        "eval_result": eval_result,
    }


# ---------------------------------------------------------------------------
# Per-task aggregate metric extraction
# ---------------------------------------------------------------------------


def _aggregate_results(raw: list[dict], task: str) -> dict:
    """Compute aggregate metrics from a list of per-sample result dicts."""
    n = len(raw)
    if task in ("multi_hop", "knowledge_relay"):
        contains_acc = sum(r["eval_result"]["correct"] for r in raw) / n if n else 0.0
        exact_acc = sum(r["eval_result"]["exact_match"] for r in raw) / n if n else 0.0
        return {"contains_acc": contains_acc, "exact_match_acc": exact_acc, "n": n}
    else:
        compliance_rate = (
            sum(r["eval_result"]["compliance_rate"] for r in raw) / n if n else 0.0
        )
        return {"compliance_rate": compliance_rate, "n": n}


# ---------------------------------------------------------------------------
# Results display
# ---------------------------------------------------------------------------

# M2 NL-relay reference (from handoff.md; Tier1/Tier2, contains-match / compliance)
_M2_BASELINE: dict[str, dict[str, float]] = {
    "tier1": {
        "multi_hop": 0.225,
        "knowledge_relay": 0.000,
        "instruction_following": 0.125,
    },
    "tier2": {
        "multi_hop": 0.312,
        "knowledge_relay": 0.056,
        "instruction_following": 0.163,
    },
}


def print_results_table(
    translated_results: dict[str, dict[str, dict]],
    null_results: dict[str, dict[str, dict]] | None = None,
) -> None:
    """Print a comparison table: M3 translated vs M2 baseline (vs null relay).

    Args:
        translated_results: ``{tier: {task: metrics_dict}}``.
        null_results: Optional ``{tier: {task: metrics_dict}}`` for null relay.
    """
    tiers = sorted(translated_results.keys())
    tasks = ["multi_hop", "knowledge_relay", "instruction_following"]

    header = f"{'Task':<22} {'Tier':<6} {'M3-Translated':>14} {'M2-NLRelay':>12} {'Δ':>8}"
    if null_results:
        header += f" {'Null-Relay':>12}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    for tier in tiers:
        for task in tasks:
            m3 = translated_results.get(tier, {}).get(task)
            if m3 is None:
                continue
            if task == "instruction_following":
                m3_val = m3.get("compliance_rate", 0.0)
            else:
                m3_val = m3.get("contains_acc", 0.0)

            m2_val = _M2_BASELINE.get(tier, {}).get(task, float("nan"))
            delta = m3_val - m2_val
            # Replace Unicode minus (U+2212) with ASCII hyphen for GBK-console compat
            delta_str = f"{delta:+.3f}".replace("\u2212", "-")

            row = f"{task:<22} {tier:<6} {m3_val:>14.3f} {m2_val:>12.3f} {delta_str:>8}"
            if null_results:
                n = null_results.get(tier, {}).get(task)
                if n is not None:
                    if task == "instruction_following":
                        null_val = n.get("compliance_rate", 0.0)
                    else:
                        null_val = n.get("contains_acc", 0.0)
                    row += f" {null_val:>12.3f}"
            print(row)

    print("=" * len(header))
    print(
        "  Delta = M3-Translated - M2-NLRelay  |  positive = M3 better than NL relay baseline\n"
    )


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


def run_m3(
    tiers: list[str],
    tasks: list[str],
    seed: int,
    subset: int | None,
    arch: str,
    layer_relative: float,
    n_train_samples: int,
    do_null_relay: bool,
    do_translated: bool,
    device_str: str,
    corpus_path: Path | None = None,
) -> None:
    """Full M3 orchestration: extract → train → evaluate.

    Args:
        tiers: List of tier names to run (e.g. ``["tier1", "tier2"]``).
        tasks: List of task names.
        seed: Random seed for data sampling and training.
        subset: If set, evaluate on this many samples per task.
        arch: TranslationLayer architecture (``"linear"``, ``"mlp_1hidden"``, ``"mlp_3hidden"``).
        layer_relative: Relative extraction/injection layer depth (0–1).
        n_train_samples: Number of texts for activation pair extraction.
        do_null_relay: If True, also run null-relay baseline.
        do_translated: If True, run translated-layer evaluation.
        device_str: Torch device string.
        corpus_path: Optional path to external corpus file (one paragraph per line).
            If None, falls back to task train splits (~1k texts, v1 mode).
    """
    device = torch.device(device_str)
    random.seed(seed)

    translated_results: dict[str, dict[str, dict]] = {}
    null_results: dict[str, dict[str, dict]] = {}

    for tier in tiers:
        cfg = _TIERS[tier]
        sender_id = cfg["sender_id"]
        receiver_id = cfg["receiver_id"]
        dim_a = cfg["sender_hidden_dim"]
        dim_b = cfg["receiver_hidden_dim"]
        layer_a = resolve_layer_name(
            cfg["sender_layer_prefix"], cfg["sender_num_layers"], layer_relative
        )
        layer_b = resolve_layer_name(
            cfg["receiver_layer_prefix"], cfg["receiver_num_layers"], layer_relative
        )

        logger.info("=" * 60)
        logger.info("Tier: %s | sender=%s → receiver=%s", tier, sender_id, receiver_id)
        logger.info("Extraction layers: sender=%s  receiver=%s", layer_a, layer_b)

        # ----------------------------------------------------------------
        # Step 1: Load models
        # ----------------------------------------------------------------
        logger.info("Loading sender model...")
        model_a, tokenizer_a = load_model_and_tokenizer(sender_id, device)
        logger.info("Loading receiver model...")
        model_b, tokenizer_b = load_model_and_tokenizer(receiver_id, device)

        # ----------------------------------------------------------------
        # Step 2: Train translation layer (if requested)
        # ----------------------------------------------------------------
        # Include n_train and seed in checkpoint name to avoid cross-run cache collisions
        # (learning curve experiments use different corpus sizes and seeds).
        checkpoint_path = _CHECKPOINTS_DIR / f"m3_{tier}_{arch}_n{n_train_samples}_s{seed}.pt"
        tl: TranslationLayer | None = None

        if do_translated:
            if checkpoint_path.exists():
                logger.info("Checkpoint found — loading: %s", checkpoint_path)
                tl = load_translation_layer(checkpoint_path, device=str(device))
                tl = tl.to(device)  # type: ignore[union-attr]
            else:
                logger.info("Extracting activation pairs (%d samples)...", n_train_samples)
                train_texts = load_train_texts(
                    n_samples=n_train_samples, seed=seed, corpus_path=corpus_path
                )
                acts_a, acts_b = extract_activation_pairs(
                    model_a, tokenizer_a,
                    model_b, tokenizer_b,
                    train_texts,
                    layer_name_a=layer_a,
                    layer_name_b=layer_b,
                    device=str(device),
                    batch_size=8,
                )
                logger.info(
                    "Extracted %d pairs — acts_a: %s  acts_b: %s",
                    len(acts_a), tuple(acts_a.shape), tuple(acts_b.shape),
                )

                logger.info("Training TranslationLayer arch=%s...", arch)
                tl = TranslationLayer(
                    dim_a, dim_b,
                    arch=arch,  # type: ignore[arg-type]
                    normalize=True,
                )
                _CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
                train_log = train_translation_layer(
                    tl, acts_a, acts_b,
                    epochs=30,
                    batch_size=32,
                    lr=1e-4,
                    warmup_steps=100,
                    device=str(device),
                    checkpoint_path=checkpoint_path,
                    seed=seed,
                )
                logger.info(
                    "Training done — best_epoch=%d  best_val_loss=%.5f  elapsed=%.0fs",
                    train_log["best_epoch"],
                    train_log["best_val_loss"],
                    train_log["elapsed_sec"],
                )
                # Save training log alongside checkpoint
                log_path = checkpoint_path.with_suffix(".train_log.json")
                with open(log_path, "w") as f:
                    json.dump(train_log, f, indent=2)

                tl = tl.to(device)  # type: ignore[union-attr]

        # ----------------------------------------------------------------
        # Step 3: Evaluate per task
        # ----------------------------------------------------------------
        translated_results[tier] = {}
        null_results[tier] = {}

        for task in tasks:
            logger.info("-" * 40)
            logger.info("Task: %s", task)

            test_samples = load_test_split(task)
            if subset:
                rng = random.Random(seed)
                test_samples = rng.sample(test_samples, min(subset, len(test_samples)))
            logger.info("  Samples: %d", len(test_samples))

            # --- Null relay ---
            if do_null_relay:
                logger.info("  Running null-relay baseline...")
                null_raw: list[dict] = []
                for i, sample in enumerate(test_samples):
                    result = run_sample_null_relay(sample, model_b, tokenizer_b, task)
                    null_raw.append(result)
                    if (i + 1) % 20 == 0:
                        logger.info("    null relay: %d/%d", i + 1, len(test_samples))

                metrics_null = _aggregate_results(null_raw, task)
                null_results[tier][task] = metrics_null
                logger.info("  Null relay metrics: %s", metrics_null)

                out_path = _RESULTS_DIR / f"m3_nullrelay_{tier}_{task}_{seed}.json"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with open(out_path, "w") as f:
                    json.dump(
                        {
                            "tier": tier,
                            "task": task,
                            "seed": seed,
                            "metric": metrics_null,
                            "n_samples": len(null_raw),
                            "raw": null_raw,
                        },
                        f,
                        indent=2,
                    )
                logger.info("  Saved → %s", out_path)

            # --- Translated ---
            if do_translated and tl is not None:
                logger.info("  Running translated pipeline...")
                trans_raw: list[dict] = []
                t0 = time.time()
                for i, sample in enumerate(test_samples):
                    result = run_sample_translated(
                        sample,
                        model_a, tokenizer_a,
                        model_b, tokenizer_b,
                        tl, layer_a, layer_b,
                        task, str(device),
                    )
                    trans_raw.append(result)
                    if (i + 1) % 20 == 0:
                        logger.info("    translated: %d/%d", i + 1, len(test_samples))

                elapsed = time.time() - t0
                metrics_trans = _aggregate_results(trans_raw, task)
                translated_results[tier][task] = metrics_trans
                logger.info("  Translated metrics: %s  (%.0fs)", metrics_trans, elapsed)

                out_path = _RESULTS_DIR / f"m3_translation_{tier}_{task}_{arch}_{seed}.json"
                gen_params = {
                    "arch": arch,
                    "layer_a": layer_a,
                    "layer_b": layer_b,
                    "layer_relative": layer_relative,
                    "normalize": True,
                    "injection": "additive_residual",
                    "injection_scale": 1.0,
                    "receiver_max_new_tokens": 64,
                    "receiver_do_sample": False,
                    "receiver_repetition_penalty": 1.3,
                    "receiver_no_repeat_ngram_size": 3,
                }
                with open(out_path, "w") as f:
                    json.dump(
                        {
                            "tier": tier,
                            "task": task,
                            "arch": arch,
                            "seed": seed,
                            "metric": metrics_trans,
                            "n_samples": len(trans_raw),
                            "elapsed_sec": elapsed,
                            "gen_params": gen_params,
                            "m2_baseline": _M2_BASELINE.get(tier, {}).get(task),
                            "raw": trans_raw,
                        },
                        f,
                        indent=2,
                    )
                logger.info("  Saved → %s", out_path)

        # Free GPU memory between tiers
        del model_a, model_b
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ----------------------------------------------------------------
    # Step 4: Print summary table
    # ----------------------------------------------------------------
    print_results_table(
        translated_results,
        null_results if do_null_relay else None,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="M3: Translation-layer communication baseline")
    p.add_argument(
        "--tiers",
        nargs="+",
        default=["tier1", "tier2"],
        choices=["tier1", "tier2"],
        help="Tiers to run (default: tier1 tier2)",
    )
    p.add_argument(
        "--tasks",
        nargs="+",
        default=["multi_hop", "knowledge_relay", "instruction_following"],
        choices=["multi_hop", "knowledge_relay", "instruction_following"],
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--subset",
        type=int,
        default=None,
        help="Evaluate on this many samples per task (default: full test split)",
    )
    p.add_argument(
        "--arch",
        default="mlp_1hidden",
        choices=["linear", "mlp_1hidden", "mlp_3hidden"],
        help="TranslationLayer architecture (default: mlp_1hidden)",
    )
    p.add_argument(
        "--layer-relative",
        type=float,
        default=_DEFAULT_RELATIVE_LAYER,
        help=f"Relative extraction/injection layer depth 0-1 (default: {_DEFAULT_RELATIVE_LAYER})",
    )
    p.add_argument(
        "--n-train-samples",
        type=int,
        default=30000,
        help="Number of texts for activation pair extraction (default: 30000)",
    )
    p.add_argument(
        "--corpus-path",
        type=Path,
        default=None,
        help=(
            "Path to external corpus file (one paragraph per line). "
            "If omitted, falls back to task train splits (~1k texts). "
            "Generate with: python scripts/download_corpus.py"
        ),
    )
    p.add_argument(
        "--null-relay",
        action="store_true",
        help="Run null-relay baseline (empty string relay → receiver only)",
    )
    p.add_argument(
        "--no-translated",
        action="store_true",
        help="Skip translated-layer evaluation (useful for null-relay-only runs)",
    )
    p.add_argument(
        "--device",
        default="auto",
        help="Torch device: 'auto', 'cuda', 'dml' (DirectML/AMD), 'cpu' (default: auto)",
    )
    return p.parse_args(argv)


def main(argv=None) -> None:
    """Entry point."""
    args = _parse_args(argv)

    device_str: str
    if args.device == "auto":
        device_str = str(get_device())
    elif args.device == "dml":
        try:
            import torch_directml  # type: ignore[import]
            device_str = str(torch_directml.device())
        except ImportError:
            print("ERROR: --device dml requires torch-directml.  Install with: pip install torch-directml")
            sys.exit(1)
    else:
        device_str = args.device

    do_null = args.null_relay
    do_trans = not args.no_translated

    if not do_null and not do_trans:
        print("Nothing to run: --no-translated without --null-relay.  Add at least one mode.")
        sys.exit(1)

    logger.info("M3 Translation-Layer Experiment")
    logger.info("  tiers=%s  tasks=%s  seed=%d  arch=%s  layer_rel=%.2f  device=%s",
                args.tiers, args.tasks, args.seed, args.arch, args.layer_relative, device_str)
    logger.info(
        "  null_relay=%s  translated=%s  n_train=%d  corpus=%s",
        do_null, do_trans, args.n_train_samples, args.corpus_path,
    )

    run_m3(
        tiers=args.tiers,
        tasks=args.tasks,
        seed=args.seed,
        subset=args.subset,
        arch=args.arch,
        layer_relative=args.layer_relative,
        n_train_samples=args.n_train_samples,
        do_null_relay=do_null,
        do_translated=do_trans,
        device_str=device_str,
        corpus_path=args.corpus_path,
    )


if __name__ == "__main__":
    main()
