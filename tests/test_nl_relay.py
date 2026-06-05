"""Unit tests for rosetta.translation.nl_relay.

These tests verify prompt format correctness and the generate/receiver
inference wrappers using lightweight mocks — no real model loading required.

Run with:
    pytest tests/test_nl_relay.py -v
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from rosetta.translation.nl_relay import (
    RECEIVER_MAX_NEW_TOKENS,
    SENDER_MAX_NEW_TOKENS,
    build_receiver_prompt,
    build_sender_prompt,
    detect_task,
    generate_relay,
    run_receiver,
)

# ---------------------------------------------------------------------------
# Fixtures: minimal sample dicts for each task
# ---------------------------------------------------------------------------

MULTI_HOP_SAMPLE = {
    "context": "Alice lives in Paris. Paris is in France. France is in Europe.",
    "question": "In which continent does Alice live?",
    "answer": "Europe",
    "hops": 3,
}

KNOWLEDGE_RELAY_SAMPLE = {
    "passage": "The Eiffel Tower is located in Paris, France.",
    "question": "In which country is the Eiffel Tower?",
    "answer": "France",
    "passage_id": 0,
}

INSTRUCTION_FOLLOWING_SAMPLE = {
    "instruction": "Translate to French and use at most 5 words.",
    "input_text": "Hello, how are you?",
    "constraints": ["language:french", "max_words:5"],
}

# ---------------------------------------------------------------------------
# detect_task
# ---------------------------------------------------------------------------


def test_detect_task_multi_hop():
    assert detect_task(MULTI_HOP_SAMPLE) == "multi_hop"


def test_detect_task_knowledge_relay():
    assert detect_task(KNOWLEDGE_RELAY_SAMPLE) == "knowledge_relay"


def test_detect_task_instruction_following():
    assert detect_task(INSTRUCTION_FOLLOWING_SAMPLE) == "instruction_following"


def test_detect_task_unknown_raises():
    with pytest.raises(ValueError, match="Cannot determine task type"):
        detect_task({"unknown_key": "value"})


# ---------------------------------------------------------------------------
# build_sender_prompt
# ---------------------------------------------------------------------------


def test_sender_prompt_multi_hop_contains_context():
    prompt = build_sender_prompt(MULTI_HOP_SAMPLE)
    assert MULTI_HOP_SAMPLE["context"] in prompt


def test_sender_prompt_multi_hop_ends_with_anchor():
    prompt = build_sender_prompt(MULTI_HOP_SAMPLE)
    assert prompt.endswith("Key facts:")


def test_sender_prompt_knowledge_relay_contains_passage():
    prompt = build_sender_prompt(KNOWLEDGE_RELAY_SAMPLE)
    assert KNOWLEDGE_RELAY_SAMPLE["passage"] in prompt


def test_sender_prompt_knowledge_relay_ends_with_anchor():
    prompt = build_sender_prompt(KNOWLEDGE_RELAY_SAMPLE)
    assert prompt.endswith("Key facts:")


def test_sender_prompt_instruction_following_contains_instruction():
    prompt = build_sender_prompt(INSTRUCTION_FOLLOWING_SAMPLE)
    assert INSTRUCTION_FOLLOWING_SAMPLE["instruction"] in prompt


def test_sender_prompt_instruction_following_contains_input():
    prompt = build_sender_prompt(INSTRUCTION_FOLLOWING_SAMPLE)
    assert INSTRUCTION_FOLLOWING_SAMPLE["input_text"] in prompt


def test_sender_prompt_no_question_leakage_multi_hop():
    """Question must not appear in sender prompt for multi_hop/knowledge_relay.
    Model A should relay context only; Model B gets the question separately.
    """
    prompt = build_sender_prompt(MULTI_HOP_SAMPLE)
    assert MULTI_HOP_SAMPLE["question"] not in prompt


# ---------------------------------------------------------------------------
# build_receiver_prompt
# ---------------------------------------------------------------------------


def test_receiver_prompt_multi_hop_format():
    relay = "Alice is from France in Europe."
    question = "In which continent does Alice live?"
    prompt = build_receiver_prompt(relay, question, task="multi_hop")
    assert relay in prompt
    assert question in prompt
    assert "A:" in prompt
    assert "Q:" in prompt


def test_receiver_prompt_knowledge_relay_format():
    relay = "Eiffel Tower is in France."
    question = "In which country is the Eiffel Tower?"
    prompt = build_receiver_prompt(relay, question, task="knowledge_relay")
    assert "A:" in prompt


def test_receiver_prompt_instruction_following_format():
    relay = "Translate to French, max 5 words."
    input_text = "Hello, how are you?"
    prompt = build_receiver_prompt(relay, input_text, task="instruction_following")
    assert relay in prompt
    assert "Output:" in prompt
    # instruction_following passes input_text into relay context, not separately
    assert "A:" not in prompt


def test_receiver_prompt_default_task_is_qa():
    """Default (no task arg) should use QA format."""
    prompt = build_receiver_prompt("relay text", "a question?")
    assert "A:" in prompt


# ---------------------------------------------------------------------------
# generate_relay (mock inference)
# ---------------------------------------------------------------------------


def _make_mock_model_tokenizer(
    prompt_token_ids: list[int],
    generated_token_ids: list[int],
    decoded_text: str,
):
    """Helper: create lightweight mocks for model and tokenizer."""
    # Tokenizer mock: tokenizer(prompt, return_tensors="pt") -> BatchEncoding-like
    all_ids = prompt_token_ids + generated_token_ids
    mock_inputs = MagicMock()
    mock_inputs.__getitem__ = lambda _, k: (
        torch.tensor([prompt_token_ids]) if k == "input_ids" else MagicMock()
    )
    mock_inputs.to.return_value = mock_inputs

    tokenizer = MagicMock()
    tokenizer.return_value = mock_inputs
    tokenizer.decode.return_value = f"  {decoded_text}  "  # simulate untrimmed output

    # Model mock: model.generate(**inputs) -> tensor of shape [1, total_tokens]
    model = MagicMock()
    model.device = torch.device("cpu")
    model.generate.return_value = torch.tensor([all_ids])

    return model, tokenizer, mock_inputs


def test_generate_relay_returns_stripped_string():
    model, tokenizer, _ = _make_mock_model_tokenizer(
        prompt_token_ids=[1, 2, 3],
        generated_token_ids=[4, 5],
        decoded_text="key facts",
    )
    result = generate_relay(model, tokenizer, "some prompt")
    assert result == "key facts"  # stripped


def test_generate_relay_decodes_new_tokens_only():
    """Only the tokens after the prompt should be decoded."""
    model, tokenizer, _ = _make_mock_model_tokenizer(
        prompt_token_ids=[1, 2, 3],
        generated_token_ids=[4, 5],
        decoded_text="new content",
    )
    generate_relay(model, tokenizer, "prompt")
    decoded_arg = tokenizer.decode.call_args[0][0]
    assert decoded_arg.tolist() == [4, 5]


def test_generate_relay_uses_greedy_defaults():
    """do_sample=False and correct max_new_tokens must be passed to generate."""
    model, tokenizer, _ = _make_mock_model_tokenizer(
        prompt_token_ids=[1],
        generated_token_ids=[2],
        decoded_text="x",
    )
    generate_relay(model, tokenizer, "prompt")
    call_kwargs = model.generate.call_args[1]
    assert call_kwargs.get("do_sample") is False
    assert call_kwargs.get("max_new_tokens") == SENDER_MAX_NEW_TOKENS


def test_run_receiver_uses_receiver_defaults():
    """Receiver must use RECEIVER_MAX_NEW_TOKENS and greedy decoding."""
    model, tokenizer, _ = _make_mock_model_tokenizer(
        prompt_token_ids=[1, 2],
        generated_token_ids=[3],
        decoded_text="Europe",
    )
    result = run_receiver(model, tokenizer, "relay text", "question?", task="multi_hop")
    call_kwargs = model.generate.call_args[1]
    assert call_kwargs.get("do_sample") is False
    assert call_kwargs.get("max_new_tokens") == RECEIVER_MAX_NEW_TOKENS
    assert result == "Europe"


def test_run_receiver_instruction_following_uses_output_anchor():
    """For instruction_following the receiver prompt must contain 'Output:'."""
    model, tokenizer, _ = _make_mock_model_tokenizer(
        prompt_token_ids=[1],
        generated_token_ids=[2],
        decoded_text="Bonjour",
    )
    run_receiver(model, tokenizer, "relay", "Hello", task="instruction_following")
    tokenizer_call_prompt = tokenizer.call_args[0][0]
    assert "Output:" in tokenizer_call_prompt
    assert "A:" not in tokenizer_call_prompt
