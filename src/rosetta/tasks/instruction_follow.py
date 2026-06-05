"""
Instruction following task data generator.

Generates multi-constraint instruction samples.  Each sample has an
instruction combining 2-4 constraints across four orthogonal dimensions
(target language, tone, length, output format).  Generation is
deterministic given a seed; no network access required.

Task design (experiment guide §4.4, Task 2):
    Model A processes the full instruction + input_text.
    Model B receives the relay and must execute the instruction.
    The task exposes whether the relay preserves multi-constraint intent.

Data format per sample:
    {
        "instruction":   "<natural language instruction string>",
        "input_text":    "<simple English sentence to transform>",
        "constraints":   ["dim:value", ...],   # machine-readable list
        "relay_point":   "Model A processes instruction+input; "
                         "Model B receives relay and executes instruction."
    }
"""

from __future__ import annotations

import random
from typing import Any

# ---------------------------------------------------------------------------
# Constraint dimensions and values
# ---------------------------------------------------------------------------

LANGUAGES = [
    ("translate to French",  "language:french"),
    ("translate to German",  "language:german"),
    ("translate to Spanish", "language:spanish"),
    ("rewrite in English",   "language:english"),
    ("translate to Italian", "language:italian"),
    ("translate to Portuguese", "language:portuguese"),
]

TONES = [
    ("use formal, polite language",   "tone:formal"),
    ("use informal, casual language", "tone:informal"),
    ("use academic, scholarly language", "tone:academic"),
    ("use simple, everyday language", "tone:simple"),
    ("use persuasive, confident language", "tone:persuasive"),
]

LENGTHS = [
    ("limit your response to at most 10 words", "max_words:10"),
    ("limit your response to at most 15 words", "max_words:15"),
    ("limit your response to at most 20 words", "max_words:20"),
    ("limit your response to at most 25 words", "max_words:25"),
    ("limit your response to at most 30 words", "max_words:30"),
]

FORMATS = [
    ("end your response with a question mark",  "format:question"),
    ("write your response as a single declarative sentence", "format:statement"),
    ("format your response as a bullet list with 2 items", "format:list"),
    ("write your response as a one-sentence summary", "format:summary"),
    ("present your response as a direct command", "format:command"),
]

# ---------------------------------------------------------------------------
# Input sentences
# ---------------------------------------------------------------------------

INPUT_SENTENCES = [
    "The weather is nice today.",
    "I enjoy reading books in the evening.",
    "Technology has changed the way we communicate.",
    "The train arrives at noon.",
    "She decided to start a small business.",
    "Learning a new language takes time and dedication.",
    "The scientists discovered a new planet.",
    "Fresh vegetables are important for a healthy diet.",
    "The conference will take place next month.",
    "He spent the weekend hiking in the mountains.",
    "The library offers free access to thousands of books.",
    "Electric vehicles are becoming more affordable.",
    "Children learn best through play and exploration.",
    "The artist painted landscapes of the countryside.",
    "Water is essential for all forms of life.",
    "The new policy will take effect next year.",
    "She volunteers at the local hospital every week.",
    "The competition attracted participants from many countries.",
    "Recycling helps reduce the amount of waste in landfills.",
    "The museum opened a new exhibit on ancient civilizations.",
    "Remote work has become common in many industries.",
    "The chef prepared a traditional three-course meal.",
    "Smartphones have transformed how people access information.",
    "The team celebrated their victory with a parade.",
    "Regular exercise improves both physical and mental health.",
    "The documentary explored the effects of climate change.",
    "She received a scholarship to study abroad.",
    "The bridge was built over a century ago.",
    "Music has the power to evoke strong emotions.",
    "The software update fixed several security vulnerabilities.",
]

RELAY_POINT = (
    "Model A processes instruction+input; Model B receives relay and executes instruction."
)

# ---------------------------------------------------------------------------
# Constraint evaluation
# ---------------------------------------------------------------------------

# Constraint dimensions and their checkability.
# "auto"   — checked programmatically by check_constraints()
# "approx" — heuristic check (language via character set); may have false
#             negatives for very short outputs with no diacritic characters
# "skip"   — cannot be reliably checked without an NLP classifier; excluded
#             from automated constraint compliance rate
CONSTRAINT_CHECKABILITY: dict[str, str] = {
    "max_words": "auto",
    "format":    "auto",
    "language":  "approx",
    "tone":      "skip",
}

# Language → distinctive Unicode characters that strongly imply that language.
# Absence of these characters does NOT prove the output is wrong (short outputs
# may be free of diacritics even when correct), hence "approx" checkability.
_LANG_CHARS: dict[str, str] = {
    "french":     "àâæçéèêëîïôœùûüÿÀÂÆÇÉÈÊËÎÏÔŒÙÛÜŸ",
    "german":     "äöüßÄÖÜ",
    "spanish":    "áéíóúüñÁÉÍÓÚÜÑ¡¿",
    "italian":    "àèéìíîòóùúÀÈÉÌÍÎÒÓÙÚ",
    "portuguese": "ãõáéíóúâêîôûçÃÕÁÉÍÓÚÂÊÎÔÛÇ",
    "english":    "",  # proxy: no non-ASCII chars (rough)
}


def check_constraints(
    output: str,
    constraints: list[str],
) -> dict[str, bool | None]:
    """Check whether a model output satisfies each constraint.

    Args:
        output:      The string produced by Model B.
        constraints: List of constraint tags from the sample, e.g.
                     ["max_words:20", "format:question", "language:french"].

    Returns:
        Dict mapping each constraint tag to:
            True  — satisfied
            False — violated
            None  — not checkable programmatically (tone constraints);
                    excluded from the automated compliance rate.

    Compliance rate:
        score = sum(1 for v in result.values() if v is True)
               / sum(1 for v in result.values() if v is not None)
        Only checkable constraints (non-None) count toward the rate.
    """
    result: dict[str, bool | None] = {}
    text = output.strip()

    for tag in constraints:
        dim, value = tag.split(":", 1)

        if dim == "max_words":
            limit = int(value)
            result[tag] = len(text.split()) <= limit

        elif dim == "format":
            if value == "question":
                result[tag] = text.endswith("?")
            elif value == "statement":
                # Single declarative sentence: ends with ".", not "?"
                result[tag] = text.endswith(".") and "?" not in text
            elif value == "list":
                # Contains at least one list marker (bullet or numbered)
                has_bullet = any(m in text for m in ("-", "•", "*"))
                has_number = any(f"{i}." in text for i in range(1, 6))
                result[tag] = has_bullet or has_number
            elif value == "summary":
                # Single sentence (at most one sentence-ending punctuation)
                ends = sum(1 for ch in text if ch in ".!?")
                result[tag] = ends <= 1
            elif value == "command":
                # Starts with an imperative verb; use a small allowlist
                _IMPERATIVES = {
                    "write", "make", "use", "put", "give", "take", "show",
                    "add", "remove", "set", "get", "create", "open", "close",
                    "start", "stop", "read", "send", "turn", "keep", "let",
                    "go", "do", "try", "check", "find", "list", "tell",
                    "translate", "rewrite", "explain", "describe", "state",
                }
                first_word = text.split()[0].lower().rstrip(".,!?") if text else ""
                result[tag] = first_word in _IMPERATIVES
            else:
                result[tag] = None  # unknown format subtype

        elif dim == "language":
            lang_chars = _LANG_CHARS.get(value)
            if lang_chars is None:
                result[tag] = None  # unknown language
            elif value == "english":
                # Proxy: all characters are ASCII
                result[tag] = all(ord(ch) < 128 for ch in text)
            else:
                # Proxy: output contains at least one language-specific char
                result[tag] = any(ch in lang_chars for ch in text)

        elif dim == "tone":
            result[tag] = None  # not programmatically checkable

        else:
            result[tag] = None  # unknown dimension

    return result

# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def _build_instruction(
    constraint_dims: list[tuple[str, str]], rng: random.Random
) -> tuple[str, list[str]]:
    """Build a natural language instruction and constraint list.

    Args:
        constraint_dims: List of (description, machine_tag) tuples.
        rng: Random number generator.

    Returns:
        Tuple of (instruction_text, constraint_list).
    """
    # Shuffle to vary the order of constraints in the instruction text
    shuffled = list(constraint_dims)
    rng.shuffle(shuffled)

    descriptions = [d for d, _ in shuffled]
    tags = [t for _, t in shuffled]

    # Build instruction: "Rewrite the input sentence. <constraints, joined>"
    constraint_text = "; ".join(descriptions)
    instruction = f"Rewrite the input sentence. {constraint_text[0].upper()}{constraint_text[1:]}."

    return instruction, tags


def generate(num_samples: int = 600, seed: int = 42) -> list[dict[str, Any]]:
    """Generate instruction following samples.

    Each sample combines 2, 3, or 4 constraints (distributed equally).
    Constraints are drawn without replacement from the four dimensions
    (language, tone, length, format) so no two constraints conflict.

    Args:
        num_samples: Total number of samples to generate.
        seed: Random seed for reproducibility.

    Returns:
        List of sample dicts with keys: instruction, input_text, constraints,
        relay_point.
    """
    rng = random.Random(seed)
    samples: list[dict[str, Any]] = []

    # Distribute evenly across 2/3/4 constraint counts
    per_count = num_samples // 3
    counts = {2: num_samples - 2 * per_count, 3: per_count, 4: per_count}

    dim_pools = [LANGUAGES, TONES, LENGTHS, FORMATS]

    for n_constraints, n in counts.items():
        for _ in range(n):
            # Pick n_constraints dimensions (always 4 dims available, so safe)
            chosen_dims = rng.sample(dim_pools, k=n_constraints)
            # Pick one value from each chosen dimension
            constraint_dims = [rng.choice(dim) for dim in chosen_dims]

            instruction, tags = _build_instruction(constraint_dims, rng)
            input_text = rng.choice(INPUT_SENTENCES)

            samples.append({
                "instruction":  instruction,
                "input_text":   input_text,
                "constraints":  tags,
                "relay_point":  RELAY_POINT,
            })

    rng.shuffle(samples)
    return samples
