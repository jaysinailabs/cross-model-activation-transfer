"""NL relay sender for Rosetta Phase 1 M2 (Group 1 baseline).

This module provides the natural-language relay pipeline:
  1. build_sender_prompt(sample)  -- task-aware prompt construction
  2. generate_relay(model, tokenizer, prompt, **gen_kwargs)  -- sender inference
  3. build_receiver_prompt(relay, question, task)  -- receiver prompt construction
  4. run_receiver(model, tokenizer, relay, question, task, **gen_kwargs)  -- receiver inference

Generation parameters (fixed for reproducibility):
    Sender:   max_new_tokens=128, do_sample=False (greedy), repetition_penalty=1.3,
              no_repeat_ngram_size=3
    Receiver: max_new_tokens=64,  do_sample=False (greedy), repetition_penalty=1.3,
              no_repeat_ngram_size=3

    repetition_penalty and no_repeat_ngram_size are required because Pythia (raw
    autoregressive LM, not instruction-tuned) tends to loop on structured prompt
    markers without these guards.  Values were chosen conservatively (1.3, 3-gram)
    to suppress degenerate loops while not distorting the generation distribution.

Task-to-prompt mapping:
    multi_hop_reasoning   -- sample keys: context, question, answer
    knowledge_relay       -- sample keys: passage, question, answer, passage_id
    instruction_following -- sample keys: instruction, input_text, constraints

Receiver "question" field convention:
    multi_hop / knowledge_relay  : sample["question"]   (the actual QA question)
    instruction_following        : sample["input_text"]  (the text model B must transform)
"""

from __future__ import annotations

import torch

# ---------------------------------------------------------------------------
# Generation defaults (fixed for AC5 reproducibility)
# ---------------------------------------------------------------------------

SENDER_MAX_NEW_TOKENS: int = 128
RECEIVER_MAX_NEW_TOKENS: int = 64

_SENDER_GEN_DEFAULTS: dict = {
    "max_new_tokens":       SENDER_MAX_NEW_TOKENS,
    "do_sample":            False,   # greedy decoding; temperature is not applicable
    "repetition_penalty":   1.3,     # suppress degenerate loops in raw LMs
    "no_repeat_ngram_size": 3,       # block 3-gram repetition
}

_RECEIVER_GEN_DEFAULTS: dict = {
    "max_new_tokens":       RECEIVER_MAX_NEW_TOKENS,
    "do_sample":            False,
    "repetition_penalty":   1.3,
    "no_repeat_ngram_size": 3,
}

# ---------------------------------------------------------------------------
# Task detection
# ---------------------------------------------------------------------------


def detect_task(sample: dict) -> str:
    """Infer task type from the sample's keys.

    Args:
        sample: A single data sample dict from a task JSONL file.

    Returns:
        One of "multi_hop", "knowledge_relay", "instruction_following".

    Raises:
        ValueError: If the sample keys do not match any known task.
    """
    if "context" in sample:
        return "multi_hop"
    if "passage" in sample:
        return "knowledge_relay"
    if "instruction" in sample:
        return "instruction_following"
    raise ValueError(
        f"Cannot determine task type from sample keys: {sorted(sample.keys())}"
    )


# ---------------------------------------------------------------------------
# Sender prompt
# ---------------------------------------------------------------------------

# Sender prompt format for each task — recorded here as the canonical spec.
# (v2: simplified to avoid structured "Text:/Relay:" markers that trigger
#  completion-loop in raw autoregressive LMs like Pythia.)
#
# multi_hop / knowledge_relay:
#   "<context or passage>\n\nKey facts:"
#
# instruction_following:
#   "Instruction: <instruction>\nInput: <input_text>\nPlan:"
#
# Rationale: Small non-instruction-tuned LMs do not follow meta-instructions
# like "Read and relay key facts."  They continue the document.  A minimal
# completion anchor ("Key facts:" / "Plan:") is less likely to produce the
# "Text: ... Key facts: ... Text: ..." looping pattern than the previous
# "Text: <text>\n\nRelay:" format.


def build_sender_prompt(sample: dict) -> str:
    """Build the sender (Model A) prompt from a task sample.

    The prompt ends with a short completion anchor so that the model continues
    with relay content rather than repeating the prompt structure.

    Args:
        sample: A task sample dict (multi_hop, knowledge_relay, or
                instruction_following format).

    Returns:
        Formatted prompt string ready for tokenization.
    """
    task = detect_task(sample)
    if task == "multi_hop":
        return f"{sample['context']}\n\nKey facts:"
    if task == "knowledge_relay":
        return f"{sample['passage']}\n\nKey facts:"
    # instruction_following
    return (
        f"Instruction: {sample['instruction']}\n"
        f"Input: {sample['input_text']}\n"
        "Plan:"
    )


# ---------------------------------------------------------------------------
# Receiver prompt
# ---------------------------------------------------------------------------

# Receiver prompt format — recorded here as the canonical spec.
# (v2: tightened to single newlines to reduce the chance of the model treating
#  blank lines as section separators and looping through them.)
#
# multi_hop / knowledge_relay:
#   "<relay>\nQ: <question>\nA:"
#
# instruction_following:
#   "<relay>\nOutput:"


def build_receiver_prompt(relay: str, question: str, task: str = "qa") -> str:
    """Build the receiver (Model B) prompt.

    Args:
        relay:    The NL relay text produced by the sender.
        question: For multi_hop/knowledge_relay: the question text.
                  For instruction_following: the input_text to transform.
        task:     Task identifier ("multi_hop", "knowledge_relay", or
                  "instruction_following").  Any other value defaults to QA format.

    Returns:
        Formatted prompt string for the receiver model.
    """
    if task == "instruction_following":
        return f"{relay}\nOutput:"
    return f"{relay}\nQ: {question}\nA:"


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------


def _generate(
    model: "torch.nn.Module",
    tokenizer,
    prompt: str,
    gen_defaults: dict,
    **gen_kwargs,
) -> str:
    """Shared inference logic: tokenize → generate → decode new tokens only.

    Greedy decoding is the default; callers may override via gen_kwargs, but
    this should only be done in tests (not in production experiments, where
    parameters must stay fixed for reproducibility).
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len: int = inputs["input_ids"].shape[1]

    kwargs = {**gen_defaults, **gen_kwargs}
    with torch.no_grad():
        output_ids = model.generate(**inputs, **kwargs)

    # Decode only the newly generated tokens (exclude the echoed prompt).
    new_token_ids = output_ids[0, prompt_len:]
    return tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()


def generate_relay(model, tokenizer, prompt: str, **gen_kwargs) -> str:
    """Run the sender model to produce a natural-language relay.

    Args:
        model:     HuggingFace causal-LM model (sender, Model A).
        tokenizer: Corresponding tokenizer.
        prompt:    Full sender prompt (use build_sender_prompt to construct).
        **gen_kwargs: Optional overrides for generation parameters.
                      In experiments, keep empty to use reproducible defaults.

    Returns:
        Relay string (new tokens only, stripped).
    """
    return _generate(model, tokenizer, prompt, _SENDER_GEN_DEFAULTS, **gen_kwargs)


def run_receiver(
    model,
    tokenizer,
    relay: str,
    question: str,
    task: str = "qa",
    **gen_kwargs,
) -> str:
    """Run the receiver model to produce a final answer from the relay.

    Args:
        model:     HuggingFace causal-LM model (receiver, Model B).
        tokenizer: Corresponding tokenizer.
        relay:     NL relay text from the sender.
        question:  Question string (or input_text for instruction_following).
        task:      Task identifier — controls prompt format.
        **gen_kwargs: Optional overrides (leave empty for reproducibility).

    Returns:
        Predicted answer string (new tokens only, stripped).
    """
    prompt = build_receiver_prompt(relay, question, task=task)
    return _generate(model, tokenizer, prompt, _RECEIVER_GEN_DEFAULTS, **gen_kwargs)
