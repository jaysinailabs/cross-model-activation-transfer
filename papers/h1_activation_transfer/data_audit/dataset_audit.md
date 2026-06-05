# Dataset Audit for H1 Clean Evaluation

> Status: draft. This document records the audit policy for constructing the
> paper-grade clean evaluation set.

## Source Files

Historical source files live under:

```text
data/tasks/multi_hop_reasoning/
```

Relevant files:

- `train.jsonl`: training split used for multi-hop task data.
- `val.jsonl`: validation split.
- `test.jsonl`: original v1 test set.
- `test_v2.jsonl`: later multi-domain v2 test set.
- `test_enhanced.jsonl`: merged v1 + v2 evaluation set used by M4-M7.
- `metadata.json`: original split metadata for v1 generation.

## Known Issue

`test_enhanced.jsonl` was useful for exploratory M4-M7 experiments, but it is
not clean enough for paper main results.

Known problems:

1. Some v1 examples overlap exactly with `train.jsonl`.
2. Some evaluation rows are exact duplicates.
3. Domain labels are incomplete or absent.

The clean paper evaluation set must remove exact overlap and duplicates before
rerunning experiments.

## Exact-Match Key

The default exact identity key is:

```text
context + "\n" + question + "\n" + answer
```

Rows with the same exact key are treated as duplicates.

Rows whose exact key appears in `train.jsonl` are treated as train/eval overlap
and removed.

This is intentionally conservative and deterministic. It does not attempt to
remove semantic paraphrases.

## Clean Eval Construction Policy

The clean eval builder should:

1. Read `train.jsonl`.
2. Read `test_enhanced.jsonl`.
3. Build exact keys for train rows.
4. Iterate eval rows in original order.
5. Skip rows whose key appears in train.
6. Skip later duplicate eval rows.
7. Preserve all original fields.
8. Add provenance fields:
   - `clean_eval_id`
   - `source_eval_file`
   - `source_component_file`
   - `source_eval_index`
   - `clean_eval_version`
   - `dedup_key_sha256`
   - `answer_type`
   - `domain_inferred`
9. Write `clean_eval.jsonl`.
10. Write `leakage_report.json`.
11. Write `clean_eval_manifest.json`.

`domain_inferred` is a deterministic heuristic label for stratified analysis.
It is not a ground-truth annotation. The historical v2 files do not contain a
stable domain field.

## Required Manifest Fields

`clean_eval_manifest.json` should include:

- source train path
- source eval path
- clean eval path
- source eval hash
- clean eval hash
- train row count
- source eval row count
- clean eval row count
- removed overlap count
- removed duplicate count
- hop distribution
- batch distribution
- answer distribution summary

## Acceptance Criteria

The clean eval set is acceptable for paper reruns only if:

- train/eval exact overlap is 0 after cleaning
- eval duplicate count is 0 after cleaning
- all rows contain `context`, `question`, `answer`
- all rows preserve `hops` when present
- manifest and leakage report are generated

## Limitations

This audit only removes exact duplicates and exact train/eval overlap.

It does not remove:

- paraphrase overlap
- same-answer overlap
- same-question paraphrases
- template-level leakage

These limitations should be acknowledged if the paper discusses dataset quality.
