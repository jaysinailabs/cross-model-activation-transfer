"""Lightweight tests for H1 final paper tooling.

These tests avoid real model loading.  They cover deterministic helper logic
that the clean-rerun runner and paper summarizer rely on.

The summarizer helper lives under ``papers/h1_activation_transfer/scripts/``,
which is committed in a later (C-round) cut.  Tests that depend on it skip
gracefully when the file is absent so a clean checkout / CI run does not
fail on the A0-narrow-scope tree.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from rosetta.experiments.phase1 import h1_final_runner as runner


def _load_summarizer_module():
    path = Path("papers/h1_activation_transfer/scripts/summarize_results.py")
    if not path.is_file():
        pytest.skip(
            f"H1 paper summarizer not yet committed at {path}; "
            "runnable once papers/h1_activation_transfer/ ships in C-round."
        )
    spec = importlib.util.spec_from_file_location("h1_summarize_results", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_direction_cfg_reverses_tier2_models_and_dims():
    fwd = runner._direction_cfg("tier2", "fwd")
    rev = runner._direction_cfg("tier2", "rev")

    assert fwd["sender_id"] == "EleutherAI/pythia-160m"
    assert fwd["receiver_id"] == "EleutherAI/pythia-410m"
    assert rev["sender_id"] == "EleutherAI/pythia-410m"
    assert rev["receiver_id"] == "EleutherAI/pythia-160m"
    assert rev["sender_hidden_dim"] == fwd["receiver_hidden_dim"]
    assert rev["receiver_hidden_dim"] == fwd["sender_hidden_dim"]


def test_full_input_prompt_multi_hop_matches_final_protocol():
    sample = {
        "context": "Alice lives in Paris. Paris is in France.",
        "question": "Where does Alice live?",
        "answer": "Paris",
    }

    prompt, sender_text, question = runner._full_input_prompt(sample, "multi_hop")

    assert sender_text == sample["context"]
    assert question == sample["question"]
    assert (
        prompt
        == "Context: Alice lives in Paris. Paris is in France.\nQ: Where does Alice live?\nA:"
    )


def test_word_boundary_contains_avoids_substring_false_positive():
    summarizer = _load_summarizer_module()

    assert summarizer.word_boundary_contains("The answer is Asia.", "Asia")
    assert not summarizer.word_boundary_contains("The answer is Eurasia.", "Asia")


def test_summarize_result_recomputes_paper_metrics():
    summarizer = _load_summarizer_module()
    result = {
        "_path": "dummy.json",
        "condition": "no_inject",
        "direction": "fwd",
        "seed": None,
        "deterministic": True,
        "config": {"clean_eval_hash": "abc"},
        "checkpoint": None,
        "per_sample": [
            {"prediction": "The answer is Canada.", "gold": "Canada"},
            {"prediction": "Eurasia is mentioned.", "gold": "Asia"},
        ],
    }

    row = summarizer.summarize_result(result)

    assert row["n"] == 2
    assert row["legacy_contains_acc"] == 1.0
    assert row["word_boundary_contains_acc"] == 0.5
    assert row["normalized_exact_match_acc"] == 0.0


def test_controls_alias_expands_to_all_four_lower_bound_conditions():
    # Regression guard: the M7 academic review listed B->B self-injection,
    # same-norm random, shuffled-translation and zero-replacement as required
    # lower-bound controls for the H1 final cut. ``zero_replacement`` was
    # implemented in dispatch but originally omitted from _CONTROL_CONDITIONS,
    # so ``--conditions controls`` would silently skip it. Lock the alias
    # surface here so any future omission fails fast at unit-test time.
    expanded = runner._expand_conditions(["controls"])
    for required in (
        "b_to_b_self_inject",
        "same_norm_random",
        "shuffled_translation",
        "zero_replacement",
    ):
        assert required in expanded, (
            f"--conditions controls must include {required!r} "
            f"(implemented in dispatch but missing from _CONTROL_CONDITIONS)"
        )
