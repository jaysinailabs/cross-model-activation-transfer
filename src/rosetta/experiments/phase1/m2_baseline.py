"""M2 baseline: Natural Language (NL) relay pipeline.

Group 1 of Phase 1 experiments.  Establishes the NL-relay accuracy that
the translation-layer approach (M3) must surpass to confirm H1.

Pipeline (per sample):
    1. Sender (Model A) reads source text → generates natural-language relay
    2. Receiver (Model B) reads relay + question → generates predicted answer
    3. Evaluator compares predicted answer to ground truth

Generation parameters (fixed for AC5 reproducibility):
    Sender:   max_new_tokens=128, do_sample=False (greedy), temperature=N/A
    Receiver: max_new_tokens=64,  do_sample=False (greedy), temperature=N/A
    Seed:     controls random.sample() only; does not affect greedy decoding

Usage:
    # AC1 validation — 20-sample subset, no real model (set --dry-run for smoke):
    python -m rosetta.experiments.phase1.m2_baseline --subset 20

    # Full run, Tier 1, seed 42:
    python -m rosetta.experiments.phase1.m2_baseline --tiers tier1 --seed 42

    # Full run, both tiers, 3 seeds:
    python -m rosetta.experiments.phase1.m2_baseline --seed 42 123 456

Results are saved to:
    results/phase1/m2_baseline_{tier}_{task}_{seed}.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rosetta.tasks.instruction_follow import check_constraints
from rosetta.translation.nl_relay import (
    build_sender_prompt,
    generate_relay,
    run_receiver,
)

# ---------------------------------------------------------------------------
# Paths and model configurations
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parents[4]  # project root
_DATA_DIR = _ROOT / "data" / "tasks"
_RESULTS_DIR = _ROOT / "results" / "phase1"

# Canonical model IDs — must match configs/phase1.yaml
_TIERS: dict[str, dict[str, str]] = {
    "tier1": {
        "sender_id": "EleutherAI/pythia-160m",
        "receiver_id": "EleutherAI/pythia-160m-deduped",
    },
    "tier2": {
        "sender_id": "EleutherAI/pythia-160m",
        "receiver_id": "EleutherAI/pythia-410m",
    },
}

_TASK_DIRS: dict[str, str] = {
    "multi_hop": "multi_hop_reasoning",
    "knowledge_relay": "knowledge_relay",
    "instruction_following": "instruction_following",
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_jsonl(path: Path) -> list[dict]:
    """Load all lines from a JSONL file as a list of dicts."""
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_test_split(task_key: str) -> list[dict]:
    """Load the test split for a task.

    Args:
        task_key: One of "multi_hop", "knowledge_relay", "instruction_following".

    Returns:
        List of sample dicts.
    """
    task_dir = _DATA_DIR / _TASK_DIRS[task_key]
    path = task_dir / "test.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"Test split not found: {path}\n" "Run m1_data_generation.py first."
        )
    return load_jsonl(path)


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------


def get_device() -> torch.device:
    """Auto-detect compute device: CUDA/ROCm → DirectML → CPU.

    Priority:
      1. CUDA (NVIDIA or ROCm on Linux)
      2. DirectML (AMD/Intel GPU on Windows via torch-directml)
      3. CPU fallback
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        import torch_directml  # type: ignore[import]
        return torch_directml.device()
    except ImportError:
        pass
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model_and_tokenizer(hf_id: str, device: torch.device):
    """Load a HuggingFace causal-LM model and tokenizer.

    Args:
        hf_id:  HuggingFace model identifier.
        device: Target device.

    Returns:
        (model, tokenizer) tuple. Model is set to eval mode.
    """
    print(f"  Loading tokenizer: {hf_id}")
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    if tokenizer.pad_token is None:
        # Pythia has no pad token; use eos_token as a safe default.
        tokenizer.pad_token = tokenizer.eos_token

    print(f"  Loading model:     {hf_id}")
    model = AutoModelForCausalLM.from_pretrained(hf_id)
    # Set pad_token_id on the config to suppress per-call warnings from generate().
    model.config.pad_token_id = tokenizer.eos_token_id
    model.to(device)
    model.eval()
    print(
        f"  Loaded ({sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params)"
        f"  on {device}"
    )
    return model, tokenizer


# ---------------------------------------------------------------------------
# Evaluation functions
# ---------------------------------------------------------------------------


def _exact_match(predicted: str, answer: str) -> bool:
    """Strict exact match: strip + lower both strings."""
    return predicted.strip().lower() == answer.strip().lower()


def _contains_match(predicted: str, answer: str) -> bool:
    """Contains match: gold answer appears anywhere in predicted output.

    Rationale: raw (non-instruction-tuned) LMs generate completion-style
    output, e.g. "The answer is Canada." rather than bare "Canada".  Strict
    exact match always fails in these cases despite the information being
    present.  Contains-match is the primary metric reported for M2/M3
    comparisons; exact-match is also recorded as a secondary metric.
    """
    return answer.strip().lower() in predicted.lower()


def evaluate_multi_hop(predicted: str, answer: str) -> dict:
    """Evaluate multi_hop_reasoning prediction.

    Args:
        predicted: Model B's output string.
        answer:    Gold-standard answer string.

    Returns:
        Dict with "correct" (contains-match, primary) and
        "exact_match" (secondary) booleans.
    """
    return {
        "correct": _contains_match(predicted, answer),
        "exact_match": _exact_match(predicted, answer),
    }


def evaluate_knowledge_relay(predicted: str, answer: str) -> dict:
    """Evaluate knowledge_relay prediction.

    Args:
        predicted: Model B's output string.
        answer:    Gold-standard answer string.

    Returns:
        Dict with "correct" (contains-match, primary) and
        "exact_match" (secondary) booleans.
    """
    return {
        "correct": _contains_match(predicted, answer),
        "exact_match": _exact_match(predicted, answer),
    }


def evaluate_instruction_following(predicted: str, constraints: list[str]) -> dict[str, Any]:
    """Constraint-compliance evaluation for instruction_following.

    Calls check_constraints() from the task module.  Tone constraints are
    excluded (return None) and are not counted in the compliance rate.

    Empty predictions score 0.0: an empty string trivially satisfies
    language and max_words constraints (vacuous truth), which would
    artificially inflate the compliance rate when the model fails to generate.

    Args:
        predicted:   Model B's output string.
        constraints: List of constraint tags, e.g. ["max_words:20", "tone:formal"].

    Returns:
        dict with keys:
            "per_constraint": {tag: True/False/None}
            "compliance_rate": float in [0, 1]  (None-excluded)
            "empty_prediction": bool (True if model produced no output)
    """
    if not predicted.strip():
        # Empty output: mark all checkable constraints as failed.
        per = {tag: (None if tag.startswith("tone:") else False) for tag in constraints}
        return {"per_constraint": per, "compliance_rate": 0.0, "empty_prediction": True}

    per = check_constraints(predicted, constraints)
    checkable = [v for v in per.values() if v is not None]
    rate = sum(1 for v in checkable if v) / len(checkable) if checkable else 0.0
    return {"per_constraint": per, "compliance_rate": rate, "empty_prediction": False}


# ---------------------------------------------------------------------------
# Single-sample pipeline
# ---------------------------------------------------------------------------


def run_sample(
    sample: dict,
    model_a,
    tokenizer_a,
    model_b,
    tokenizer_b,
    task: str,
) -> dict[str, Any]:
    """Run the full NL-relay pipeline on one sample.

    Args:
        sample:      Task sample dict.
        model_a/tokenizer_a: Sender (Model A).
        model_b/tokenizer_b: Receiver (Model B).
        task:        Task identifier string.

    Returns:
        Result dict with keys: source (original sample fields for diagnosis),
        sender_prompt, relay, receiver_prompt, predicted, eval_result.
    """
    # Build sender prompt and generate relay
    sender_prompt = build_sender_prompt(sample)
    relay = generate_relay(model_a, tokenizer_a, sender_prompt)

    # Determine the "question" field for the receiver
    if task == "instruction_following":
        question = sample["input_text"]
    else:
        question = sample["question"]

    # Build receiver prompt (saved for diagnostics)
    from rosetta.translation.nl_relay import build_receiver_prompt

    receiver_prompt = build_receiver_prompt(relay, question, task=task)

    # Receiver generates the predicted answer
    predicted = run_receiver(model_b, tokenizer_b, relay, question, task=task)

    # Evaluate
    if task == "multi_hop":
        eval_result = evaluate_multi_hop(predicted, sample["answer"])
    elif task == "knowledge_relay":
        eval_result = evaluate_knowledge_relay(predicted, sample["answer"])
    else:  # instruction_following
        eval_result = evaluate_instruction_following(predicted, sample["constraints"])

    # Capture original sample fields needed for post-hoc diagnosis
    if task == "multi_hop":
        source = {
            "context": sample["context"],
            "question": sample["question"],
            "answer": sample["answer"],
            "hops": sample.get("hops"),
        }
    elif task == "knowledge_relay":
        source = {
            "passage": sample["passage"],
            "question": sample["question"],
            "answer": sample["answer"],
            "passage_id": sample.get("passage_id"),
        }
    else:
        source = {
            "instruction": sample["instruction"],
            "input_text": sample["input_text"],
            "constraints": sample["constraints"],
        }

    return {
        "source": source,
        "sender_prompt": sender_prompt,
        "relay": relay,
        "receiver_prompt": receiver_prompt,
        "predicted": predicted,
        "eval_result": eval_result,
    }


# ---------------------------------------------------------------------------
# Task-level runner
# ---------------------------------------------------------------------------


def run_task(
    task: str,
    samples: list[dict],
    model_a,
    tokenizer_a,
    model_b,
    tokenizer_b,
    verbose_first_n: int = 3,
) -> dict[str, Any]:
    """Run the pipeline over all samples for one task.

    Args:
        task:           Task identifier.
        samples:        List of test samples.
        model_a/tokenizer_a: Sender.
        model_b/tokenizer_b: Receiver.
        verbose_first_n: Print relay+predicted for first N samples.

    Returns:
        Summary dict: metric, raw results list, timing.
    """
    raw_results = []
    t0 = time.time()

    for i, sample in enumerate(samples):
        result = run_sample(sample, model_a, tokenizer_a, model_b, tokenizer_b, task)
        raw_results.append(result)

        if i < verbose_first_n:
            src = result["source"]
            ev = result["eval_result"]
            print(f"    [sample {i}]")
            if task in ("multi_hop", "knowledge_relay"):
                print(f"      Q:         {src['question']!r}")
                print(f"      answer:    {src['answer']!r}")
                print(f"      relay:     {result['relay'][:120]!r}")
                print(f"      predicted: {result['predicted'][:80]!r}")
                print(f"      contains={ev['correct']}  exact={ev['exact_match']}")
            else:
                print(f"      constraints: {src['constraints']}")
                print(f"      relay:     {result['relay'][:120]!r}")
                print(f"      predicted: {result['predicted'][:80]!r}")
                print(f"      compliance_rate={ev['compliance_rate']:.2f}")

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"    {i + 1}/{len(samples)} samples  ({elapsed:.1f}s)")

    elapsed = time.time() - t0

    # Compute aggregate metrics
    n = len(raw_results)
    if task in ("multi_hop", "knowledge_relay"):
        n_contains = sum(r["eval_result"]["correct"] for r in raw_results)
        n_exact = sum(r["eval_result"]["exact_match"] for r in raw_results)
        metric = n_contains / n if n else 0.0  # primary
        exact_acc = n_exact / n if n else 0.0  # secondary
        metric_name = "accuracy_contains"
        print(f"    Result: contains={metric:.3f}  exact={exact_acc:.3f}  n={n}")
    else:
        rates = [r["eval_result"]["compliance_rate"] for r in raw_results]
        metric = sum(rates) / len(rates) if rates else 0.0
        exact_acc = None
        metric_name = "compliance_rate"
        print(f"    Result: compliance={metric:.3f}  n={n}")

    return {
        "task": task,
        "n_samples": n,
        "metric_name": metric_name,
        "metric": metric,
        "exact_match_acc": exact_acc,
        "elapsed_sec": elapsed,
        "raw": raw_results,
    }


# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------


def print_results_table(
    results: dict[str, dict[str, dict[str, Any]]],
) -> None:
    """Print a 3-task × 2-tier results table.

    For QA tasks, reports primary (contains-match) and secondary (exact-match).
    For instruction_following, reports constraint compliance rate.

    Args:
        results: {tier: {task: task_summary_dict}}
    """
    tasks = ["multi_hop", "knowledge_relay", "instruction_following"]
    tiers = sorted(results.keys())

    print("\n" + "=" * 80)
    print("M2 Baseline Results (NL relay)")
    print("=" * 80)
    print(f"{'Task':<25} {'Metric':<18}" + "".join(f"{t:>18}" for t in tiers))
    print("-" * 80)
    for task in tasks:
        # Row 1: primary metric
        row_p = f"{task:<25} {'contains-acc':<18}"
        # Row 2: exact-match (QA tasks only)
        row_e = f"{'':25} {'exact-acc':<18}"
        for tier in tiers:
            summary = results.get(tier, {}).get(task)
            if summary is None:
                row_p += f"{'N/A':>18}"
                row_e += f"{'N/A':>18}"
            else:
                n = summary["n_samples"]
                val = summary["metric"]
                em = summary.get("exact_match_acc")
                row_p += f"{val:.3f} (n={n})".rjust(18)
                if em is not None:
                    row_e += f"{em:.3f}".rjust(18)
                else:
                    row_e += f"{'(compliance)':>18}"
        print(row_p)
        print(row_e)
    print("=" * 80)

    # AC4 check (uses contains-match as primary)
    print("\nAC4 difficulty calibration (target: 5%–85%, primary metric):")
    for tier in tiers:
        for task in tasks:
            summary = results.get(tier, {}).get(task)
            if summary is None:
                continue
            val = summary["metric"]
            status = "OK" if 0.05 <= val <= 0.85 else ("TOO HIGH" if val > 0.85 else "TOO LOW")
            print(f"  {tier}/{task}: {val:.3f}  →  {status}")


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------


def save_results(
    tier: str,
    task: str,
    seed: int,
    summary: dict[str, Any],
    prefix: str = "m2_baseline",
) -> Path:
    """Save raw results to results/phase1/{prefix}_{tier}_{task}_{seed}.json.

    Args:
        prefix: File name prefix (default: m2_baseline). Override for enhance
            runs to avoid overwriting original M2 results.

    Returns:
        Path to the saved file.
    """
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _RESULTS_DIR / f"{prefix}_{tier}_{task}_{seed}.json"

    # Strip raw per-sample data from the top-level summary to keep file manageable,
    # but preserve it nested for later analysis.
    payload = {
        "tier": tier,
        "task": task,
        "seed": seed,
        "metric_name": summary["metric_name"],
        "metric": summary["metric"],
        "exact_match_acc": summary.get("exact_match_acc"),
        "n_samples": summary["n_samples"],
        "elapsed_sec": summary["elapsed_sec"],
        "gen_params": {
            "sender_max_new_tokens": 128,
            "receiver_max_new_tokens": 64,
            "do_sample": False,
            "temperature": "N/A (greedy)",
            "repetition_penalty": 1.3,
            "no_repeat_ngram_size": 3,
        },
        "raw": summary["raw"],
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    print(f"  Saved: {out_path.relative_to(_ROOT)}")
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="M2 baseline: NL relay pipeline for Phase 1.")
    parser.add_argument(
        "--tiers",
        nargs="+",
        default=["tier1", "tier2"],
        choices=["tier1", "tier2"],
        help="Which tiers to run (default: both).",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["multi_hop", "knowledge_relay", "instruction_following"],
        choices=["multi_hop", "knowledge_relay", "instruction_following"],
        help="Which tasks to evaluate (default: all three).",
    )
    parser.add_argument(
        "--seed",
        nargs="+",
        type=int,
        default=[42],
        help="Random seed(s). Controls sample ordering only. Default: 42.",
    )
    parser.add_argument(
        "--subset",
        type=int,
        default=None,
        help="If set, take a random subset of this many samples per task (for quick tests).",
    )
    parser.add_argument(
        "--test-file",
        default=None,
        help="Override test data path (e.g. for enhanced datasets). "
        "If not set, uses the default test.jsonl for the selected task.",
    )
    parser.add_argument(
        "--prefix",
        default="m2_baseline",
        help="Output file name prefix (default: m2_baseline). "
        "Use e.g. m4e_E3a_null_relay to avoid overwriting original M2 results.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Entry point for M2 baseline experiments."""
    args = parse_args(argv)
    device = get_device()
    print(f"Device: {device}")

    all_results: dict[str, dict[str, dict[str, Any]]] = {}

    for tier in args.tiers:
        print(f"\n{'='*60}")
        print(f"Tier: {tier}")
        print(f"{'='*60}")
        tier_cfg = _TIERS[tier]
        all_results[tier] = {}

        print("\nLoading sender (Model A):")
        model_a, tokenizer_a = load_model_and_tokenizer(tier_cfg["sender_id"], device)

        print("\nLoading receiver (Model B):")
        model_b, tokenizer_b = load_model_and_tokenizer(tier_cfg["receiver_id"], device)

        for seed in args.seed:
            print(f"\n  Seed: {seed}")
            rng = random.Random(seed)

            for task in args.tasks:
                print(f"\n  Task: {task}")
                if args.test_file:
                    samples = load_jsonl(Path(args.test_file))
                    print(f"  Loaded {len(samples)} samples from {args.test_file}")
                else:
                    samples = load_test_split(task)

                if args.subset is not None:
                    samples = rng.sample(samples, min(args.subset, len(samples)))
                    print(f"  Subset: {len(samples)} samples (seed={seed})")
                else:
                    # For full test, use deterministic ordering (shuffle with seed for
                    # consistent reporting across seeds, though accuracy is unchanged).
                    shuffled = list(samples)
                    rng.shuffle(shuffled)
                    samples = shuffled
                    print(f"  Full test split: {len(samples)} samples")

                summary = run_task(task, samples, model_a, tokenizer_a, model_b, tokenizer_b)
                all_results[tier][task] = summary

                save_results(tier, task, seed, summary, prefix=args.prefix)

        # Free GPU memory before loading next tier's models
        del model_a, model_b
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print_results_table(all_results)
    print("\nDone.")


if __name__ == "__main__":
    main(sys.argv[1:])
