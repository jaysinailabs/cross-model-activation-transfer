# Metrics for H1 Activation Transfer Paper

> Status: draft definitions with an initial implementation in
> `papers/h1_activation_transfer/scripts/summarize_results.py`. The script
> should be treated as the metric source for clean-rerun tables.

## Why Metrics Need Revision

Historical H1 experiments primarily used contains-match:

```text
correct = gold_answer.lower() in generated_text.lower()
```

This is useful for raw causal LMs that generate long completion-style answers,
but it is too permissive as the only paper metric. It can count accidental
mentions of common answers such as "Asia" or "Europe".

The final paper should report strict and lenient metrics together.

## Required Metrics

### 1. Exact Match

Definition:

```text
prediction.strip().lower() == answer.strip().lower()
```

Use:

- Strictest reference.
- Expected to be low for raw non-instruction-tuned LMs.

### 2. Normalized Exact Match

Definition:

1. Lowercase.
2. Strip leading/trailing whitespace.
3. Remove common punctuation.
4. Collapse multiple whitespace characters.
5. Compare exact strings.

Implementation note: the current script removes all characters in Python's
`string.punctuation`. This is acceptable for the frozen clean eval's answer
set, but would be lossy for future gold answers containing apostrophes,
hyphens, or punctuation-bearing names.

Use:

- Strict but less brittle than raw exact match.

### 3. Word-Boundary Contains

Definition:

The normalized answer must appear as a token/phrase boundary match inside the
normalized prediction, not merely as a substring inside another word.

Use:

- Medium-strict main metric candidate.
- Reduces false positives from substring matching.

### 4. First-Answer-Span Match

Definition:

Extract the first plausible answer span from the model output, then compare it
to the normalized answer.

Initial simple heuristic:

1. Take first line or first sentence.
2. Remove common prefixes such as `The answer is`.
3. Normalize and compare against answer.

Use:

- Helps distinguish direct answers from long rambling outputs.
- Should be reported with caveats if heuristic-based.

### 5. Legacy Contains Match

Definition:

Historical contains-match.

Use:

- Enables comparison to M3-M7 historical results.
- Should be labeled lenient.

## Reporting Rules

Main result table should include at least:

- normalized exact match
- word-boundary contains
- legacy contains-match

Appendix should include all metrics.

## False-Positive Audit

For each main condition, sample cases where:

```text
legacy_contains = true
normalized_exact = false
```

Audit at least:

- 20 examples overall, or
- all examples if fewer than 20

The audit table should classify:

- clearly correct but verbose
- accidental mention
- ambiguous
- evaluator artifact

## Statistical Reporting

For final tables:

- report mean across seeds when multiple seeds exist
- report standard deviation
- report confidence interval where appropriate
- for paired sample-level comparisons, prefer paired bootstrap or McNemar-style tests
- when grouping multiple seeded runs, pool paired sample-level deltas before
  bootstrapping instead of averaging seed-wise confidence interval endpoints
- if using the current comparison script, specify that confidence intervals are
  simple percentile bootstrap intervals rather than BCa intervals
- always specify whether the baseline is deterministic

## Implementation Requirement

Metric implementation should live in:

```text
papers/h1_activation_transfer/scripts/summarize_results.py
```

Historical evaluation code may be reused, but final paper metrics should be
computed by the paper workspace script so tables are reproducible.
