"""Unit tests for LoRA alignment module (M4b).

Smoke tests only — verify construction and forward pass without GPU or PEFT.
Real training tested via integration (m4b_lora.py --phase train).
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Text loading
# ---------------------------------------------------------------------------

def test_load_alignment_texts_from_jsonl(tmp_path: Path) -> None:
    """_load_alignment_texts returns context strings from jsonl."""
    from rosetta.alignment.lora_align import _load_alignment_texts

    jsonl = tmp_path / "test.jsonl"
    records = [
        {"context": "Silicon is a semiconductor.", "question": "Q?", "answer": "A"},
        {"context": "Newton invented calculus.", "question": "Q?", "answer": "A"},
        {"context": "", "question": "Q?", "answer": "A"},   # empty context — skip
    ]
    with open(jsonl, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    texts = _load_alignment_texts(enhanced_jsonl=jsonl, corpus_dir=None, n_texts=100)
    assert len(texts) == 2, "Empty context should be skipped"
    assert "Silicon" in texts[0]
    assert "Newton" in texts[1]


def test_load_alignment_texts_deduplication(tmp_path: Path) -> None:
    """Duplicate context strings are deduplicated."""
    from rosetta.alignment.lora_align import _load_alignment_texts

    jsonl = tmp_path / "test.jsonl"
    with open(jsonl, "w", encoding="utf-8") as f:
        for _ in range(5):
            f.write(json.dumps({"context": "Same text."}) + "\n")

    texts = _load_alignment_texts(enhanced_jsonl=jsonl, corpus_dir=None, n_texts=100)
    assert texts == ["Same text."], "Duplicates should be removed"


def test_load_alignment_texts_truncates_to_n(tmp_path: Path) -> None:
    """Output is capped at n_texts."""
    from rosetta.alignment.lora_align import _load_alignment_texts

    jsonl = tmp_path / "test.jsonl"
    with open(jsonl, "w", encoding="utf-8") as f:
        for i in range(20):
            f.write(json.dumps({"context": f"Text number {i}."}) + "\n")

    texts = _load_alignment_texts(enhanced_jsonl=jsonl, corpus_dir=None, n_texts=5)
    assert len(texts) == 5


def test_load_alignment_texts_missing_file() -> None:
    """Missing file falls back to placeholder sentences."""
    from rosetta.alignment.lora_align import _load_alignment_texts

    texts = _load_alignment_texts(
        enhanced_jsonl="nonexistent_file.jsonl",
        corpus_dir=None,
        n_texts=10,
    )
    assert len(texts) > 0, "Should return placeholder sentences on missing file"


# ---------------------------------------------------------------------------
# AlignmentResult dataclass
# ---------------------------------------------------------------------------

def test_alignment_result_defaults() -> None:
    """AlignmentResult initialises with empty lists and sensible defaults."""
    from rosetta.alignment.lora_align import AlignmentResult

    r = AlignmentResult()
    assert r.loss_history == []
    assert r.converged is False
    assert r.epochs_run == 0
    assert r.checkpoint_a == ""
    assert r.checkpoint_b == ""


# ---------------------------------------------------------------------------
# _ActivationHook
# ---------------------------------------------------------------------------

def test_activation_hook_fires() -> None:
    """_ActivationHook captures output tensor and auto-detaches."""
    import torch
    import torch.nn as nn
    from rosetta.alignment.lora_align import _ActivationHook

    linear = nn.Linear(8, 4)
    hook = _ActivationHook()
    hook.attach(linear)

    x = torch.randn(2, 8)
    _ = linear(x)

    assert hook.activation is not None
    assert hook.activation.shape == (2, 4)
    assert hook._handle is None  # auto-detached


def test_activation_hook_fires_once() -> None:
    """After auto-detach, a second forward pass does not update activation."""
    import torch
    import torch.nn as nn
    from rosetta.alignment.lora_align import _ActivationHook

    linear = nn.Linear(4, 4)
    hook = _ActivationHook()
    hook.attach(linear)

    x1 = torch.ones(1, 4)
    x2 = torch.zeros(1, 4)
    _ = linear(x1)
    first_activation = hook.activation.clone()
    _ = linear(x2)  # hook is detached — activation should NOT update
    assert torch.allclose(hook.activation, first_activation)


# ---------------------------------------------------------------------------
# _TokenisedDataset
# ---------------------------------------------------------------------------

def test_tokenised_dataset_shape() -> None:
    """Dataset chunks tokens correctly."""
    import torch
    from rosetta.alignment.lora_align import _TokenisedDataset

    tokens = torch.arange(256, dtype=torch.long)
    ds = _TokenisedDataset(tokens, chunk_size=32)
    assert len(ds) == 8
    assert ds[0].shape == (32,)
    assert ds[0][0] == 0
    assert ds[1][0] == 32


def test_tokenised_dataset_truncates() -> None:
    """Incomplete trailing chunk is dropped."""
    import torch
    from rosetta.alignment.lora_align import _TokenisedDataset

    tokens = torch.arange(100, dtype=torch.long)  # 100 / 32 = 3 full + 4 leftover
    ds = _TokenisedDataset(tokens, chunk_size=32)
    assert len(ds) == 3
