"""
M1: Task Data Generation.

Generates three benchmark task datasets and writes them to
data/tasks/{task_name}/{train,val,test}.jsonl + metadata.json.

Tasks:
    multi_hop_reasoning   — 800 samples (2/3/4-hop entity-relation chains)
    instruction_following — 600 samples (2-4 multi-constraint instructions)
    knowledge_relay       — 96 unique samples (16 passages × 6 QA pairs),
                            split at the passage level to prevent leakage

Split strategy:
    multi_hop_reasoning, instruction_following:
        Random 70/10/20 split (sample-level).
    knowledge_relay:
        Passage-level 70/10/20 split (11/2/3 passages → 66/12/18 samples).
        All QA pairs from one passage stay in the same split; this prevents
        the translation layer (trained on train) from seeing passage texts
        that appear in test.  See split_knowledge_relay() below.

Run:
    python -m rosetta.experiments.phase1.m1_data_generation

Output:
    data/tasks/multi_hop_reasoning/train.jsonl   (560 samples)
    data/tasks/multi_hop_reasoning/val.jsonl     (80 samples)
    data/tasks/multi_hop_reasoning/test.jsonl    (160 samples)
    data/tasks/multi_hop_reasoning/metadata.json
    data/tasks/instruction_following/train.jsonl (420 samples)
    data/tasks/instruction_following/val.jsonl   (60 samples)
    data/tasks/instruction_following/test.jsonl  (120 samples)
    data/tasks/instruction_following/metadata.json
    data/tasks/knowledge_relay/train.jsonl       (66 samples, 11 passages)
    data/tasks/knowledge_relay/val.jsonl         (12 samples, 2 passages)
    data/tasks/knowledge_relay/test.jsonl        (18 samples, 3 passages)
    data/tasks/knowledge_relay/metadata.json
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import rosetta.tasks.multi_hop as multi_hop
import rosetta.tasks.instruction_follow as instruction_follow
import rosetta.tasks.knowledge_relay as knowledge_relay

# ---------------------------------------------------------------------------
# Configuration (mirrors configs/phase1.yaml)
# ---------------------------------------------------------------------------

# Tasks with standard sample-level random splits.
_RANDOM_SPLIT_TASKS: list[dict[str, Any]] = [
    {
        "name":        "multi_hop_reasoning",
        "generator":   multi_hop.generate,
        "num_samples": 800,
        "seed":        42,
    },
    {
        "name":        "instruction_following",
        "generator":   instruction_follow.generate,
        "num_samples": 600,
        "seed":        42,
    },
]

# knowledge_relay uses passage-level splitting (see split_knowledge_relay).
_KR_SEED = 42

TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.10
TEST_SPLIT = 0.20

# Passage-level split for knowledge_relay: 11 train / 2 val / 3 test
# out of 16 total passages (68.75% / 12.5% / 18.75%).
_KR_N_TRAIN_PASSAGES = 11
_KR_N_VAL_PASSAGES = 2
_KR_N_TEST_PASSAGES = 3

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = _PROJECT_ROOT / "data" / "tasks"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def split_samples(
    samples: list[dict[str, Any]],
    train_frac: float,
    val_frac: float,
    seed: int,
) -> tuple[list, list, list]:
    """Shuffle and split samples into train/val/test sets.

    Args:
        samples: Full list of samples.
        train_frac: Fraction for training.
        val_frac: Fraction for validation.
        seed: Shuffle seed.

    Returns:
        Tuple of (train, val, test) sample lists.
    """
    data = list(samples)
    random.Random(seed).shuffle(data)
    n = len(data)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    return data[:n_train], data[n_train:n_train + n_val], data[n_train + n_val:]


def split_knowledge_relay(
    samples: list[dict[str, Any]],
    n_train_passages: int,
    n_val_passages: int,
    seed: int,
) -> tuple[list, list, list]:
    """Split knowledge relay samples at the passage level.

    All QA pairs from a single passage are assigned to the same split,
    preventing data leakage between train and test caused by repeated
    passage texts.

    Args:
        samples:           All unique samples (each with a 'passage_id' key).
        n_train_passages:  Number of passages to assign to train.
        n_val_passages:    Number of passages to assign to val.
        seed:              Shuffle seed for passage assignment.

    Returns:
        Tuple of (train, val, test) sample lists.
    """
    passage_ids = sorted({s["passage_id"] for s in samples})
    rng = random.Random(seed)
    rng.shuffle(passage_ids)

    train_ids = set(passage_ids[:n_train_passages])
    val_ids = set(passage_ids[n_train_passages:n_train_passages + n_val_passages])
    test_ids = set(passage_ids[n_train_passages + n_val_passages:])

    train = [s for s in samples if s["passage_id"] in train_ids]
    val = [s for s in samples if s["passage_id"] in val_ids]
    test = [s for s in samples if s["passage_id"] in test_ids]
    return train, val, test


def write_jsonl(path: Path, samples: list[dict[str, Any]]) -> None:
    """Write samples to a JSONL file (one JSON object per line)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")


def write_metadata(
    path: Path,
    task_name: str,
    n_train: int,
    n_val: int,
    n_test: int,
    seed: int,
    extra: dict[str, Any],
) -> None:
    """Write task metadata as JSON."""
    meta = {
        "task":         task_name,
        "total":        n_train + n_val + n_test,
        "train":        n_train,
        "val":          n_val,
        "test":         n_test,
        "splits":       {"train": TRAIN_SPLIT, "val": VAL_SPLIT, "test": TEST_SPLIT},
        "seed":         seed,
        **extra,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Task-specific quality stats
# ---------------------------------------------------------------------------


def _stats_multi_hop(samples: list[dict]) -> dict:
    hops = Counter(s["hops"] for s in samples)
    return {"hop_distribution": dict(sorted(hops.items()))}


def _stats_instruction_follow(samples: list[dict]) -> dict:
    n_constraints = Counter(len(s["constraints"]) for s in samples)
    constraint_types = Counter(
        tag.split(":")[0] for s in samples for tag in s["constraints"]
    )
    return {
        "constraint_count_distribution": dict(sorted(n_constraints.items())),
        "constraint_type_counts": dict(sorted(constraint_types.items())),
    }


def _stats_knowledge_relay(samples: list[dict]) -> dict:
    lengths = [len(s["passage"].split()) for s in samples]
    unique_passages = len({s["passage_id"] for s in samples})
    return {
        "unique_passages": unique_passages,
        "passage_word_count": {
            "min": min(lengths),
            "max": max(lengths),
            "mean": round(sum(lengths) / len(lengths), 1),
        },
        "split_note": (
            "passage-level split: all QA pairs from each passage "
            "are in the same split (no leakage)"
        ),
    }


_STATS_FN = {
    "multi_hop_reasoning":   _stats_multi_hop,
    "instruction_following": _stats_instruction_follow,
    "knowledge_relay":       _stats_knowledge_relay,
}

# ---------------------------------------------------------------------------
# Quality report (printed to stdout)
# ---------------------------------------------------------------------------


def print_quality_report(
    task_name: str,
    train: list[dict],
    val: list[dict],
    test: list[dict],
    stats: dict,
    seed: int,
) -> None:
    """Print a concise quality report for one task."""
    all_samples = train + val + test
    print(f"\n  Task: {task_name}")
    print(f"  Total: {len(all_samples)}  (train={len(train)}, val={len(val)}, test={len(test)})")

    # Check for empty required fields (missing key or empty string; 0/False are valid)
    required = list(all_samples[0].keys())
    empty_counts: dict[str, int] = {}
    for s in all_samples:
        for k in required:
            val = s.get(k)
            if val is None or val == "":
                empty_counts[k] = empty_counts.get(k, 0) + 1
    if empty_counts:
        print(f"  WARNING: Empty fields detected: {empty_counts}")
    else:
        print("  Field check: OK (no empty required fields)")

    # Task-specific distribution
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # Sample 3 random examples
    rng = random.Random(seed + 99)
    examples = rng.sample(all_samples, k=min(3, len(all_samples)))
    print("  --- Sample examples ---")
    for i, ex in enumerate(examples, 1):
        if "hops" in ex:
            print(f"  [{i}] ({ex['hops']}-hop) Q: {ex['question']!r}  A: {ex['answer']!r}")
        elif "constraints" in ex:
            print(f"  [{i}] ({len(ex['constraints'])} constraints) {ex['constraints']}")
            print(f"       Instruction (truncated): {ex['instruction'][:80]}...")
        elif "passage" in ex:
            words = len(ex["passage"].split())
            print(f"  [{i}] ({words} words) Q: {ex['question']!r}  A: {ex['answer']!r}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 65)
    print("  M1: Task Data Generation")
    print(f"  Output directory: {DATA_DIR}")
    print("=" * 65)

    # --- Tasks with standard sample-level random splits ---
    for cfg in _RANDOM_SPLIT_TASKS:
        task_name: str = cfg["name"]
        generator = cfg["generator"]
        num_samples: int = cfg["num_samples"]
        seed: int = cfg["seed"]

        print(f"\nGenerating: {task_name}  (n={num_samples}, seed={seed})")
        samples = generator(num_samples=num_samples, seed=seed)

        train, val, test = split_samples(samples, TRAIN_SPLIT, VAL_SPLIT, seed)

        task_dir = DATA_DIR / task_name
        write_jsonl(task_dir / "train.jsonl", train)
        write_jsonl(task_dir / "val.jsonl",   val)
        write_jsonl(task_dir / "test.jsonl",  test)

        stats = _STATS_FN[task_name](samples)
        write_metadata(
            task_dir / "metadata.json",
            task_name, len(train), len(val), len(test), seed, stats,
        )

        print_quality_report(task_name, train, val, test, stats, seed)

    # --- knowledge_relay: passage-level split (no leakage) ---
    task_name = "knowledge_relay"
    print(f"\nGenerating: {task_name}  (passage-level split, seed={_KR_SEED})")
    kr_samples = knowledge_relay.generate(seed=_KR_SEED)

    train, val, test = split_knowledge_relay(
        kr_samples, _KR_N_TRAIN_PASSAGES, _KR_N_VAL_PASSAGES, _KR_SEED
    )

    task_dir = DATA_DIR / task_name
    write_jsonl(task_dir / "train.jsonl", train)
    write_jsonl(task_dir / "val.jsonl",   val)
    write_jsonl(task_dir / "test.jsonl",  test)

    stats = _STATS_FN[task_name](kr_samples)
    write_metadata(
        task_dir / "metadata.json",
        task_name, len(train), len(val), len(test), _KR_SEED, stats,
    )
    print_quality_report(task_name, train, val, test, stats, _KR_SEED)

    print(f"\n{'='*65}")
    print("  M1 complete. Data written to data/tasks/")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
