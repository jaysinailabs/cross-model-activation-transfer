# H1 Clean Rerun Execution Runbook

> Status: draft execution guide. Use this after `final_protocol.md` and
> `experiment_matrix.md` are accepted for a clean rerun.

## 1. Readiness Check

Run:

```bash
python papers/h1_activation_transfer/scripts/check_readiness.py \
  --require-model-cache
```

For a machine that will run the full clean rerun, prefer:

```bash
python papers/h1_activation_transfer/scripts/check_readiness.py \
  --require-model-cache \
  --require-cuda
```

The readiness check verifies:

- `data/tasks/multi_hop_reasoning/clean_eval.jsonl` exists
- the clean-eval hash matches `data_audit/clean_eval_manifest.json`
- all six M6 checkpoints exist for `fwd/rev x 42/123/456`
- Pythia-160M and Pythia-410M are present in the local Hugging Face cache
- CUDA availability, if required

CPU-only machines can run smoke tests, but should not be used for the full
matrix unless runtime is not a concern.

## 2. Offline Environment

For final runs, set offline flags after confirming the model cache is complete:

```bash
set TRANSFORMERS_OFFLINE=1
set HF_DATASETS_OFFLINE=1
set HF_HUB_OFFLINE=1
```

PowerShell equivalent:

```powershell
$env:TRANSFORMERS_OFFLINE = "1"
$env:HF_DATASETS_OFFLINE = "1"
$env:HF_HUB_OFFLINE = "1"
```

This prevents hidden network fetches from changing the execution environment.

## 3. Smoke Tests

Use `results/smoke`, not `results/final`.

Minimal receiver-only smoke:

```bash
python -m rosetta.experiments.phase1.h1_final_runner \
  --conditions no_inject \
  --directions fwd \
  --max-samples 1 \
  --output-dir papers/h1_activation_transfer/results/smoke
```

Injection-path smoke:

```bash
python -m rosetta.experiments.phase1.h1_final_runner \
  --conditions b_to_b_self_inject same_norm_random replace scale_corrected \
  --directions fwd \
  --seeds 42 \
  --max-samples 3 \
  --output-dir papers/h1_activation_transfer/results/smoke
```

Summarize smoke outputs:

```bash
python papers/h1_activation_transfer/scripts/summarize_results.py \
  --results-dir papers/h1_activation_transfer/results/smoke \
  --output-dir papers/h1_activation_transfer/results/smoke_summary
```

Smoke outputs are not paper evidence. They only verify execution plumbing,
schema shape, prompt alignment, and metric summarization.

## 4. Primary Clean Rerun

Run the primary forward matrix:

```bash
python -m rosetta.experiments.phase1.h1_final_runner \
  --conditions main controls \
  --directions fwd \
  --seeds 42 123 456 \
  --test-file data/tasks/multi_hop_reasoning/clean_eval.jsonl \
  --output-dir papers/h1_activation_transfer/results/final
```

Then summarize:

```bash
python papers/h1_activation_transfer/scripts/summarize_results.py
```

Validate:

```bash
python papers/h1_activation_transfer/scripts/validate_final_results.py
```

Before treating these results as paper evidence, inspect:

- `seq_len_mismatch_count`
- `token_mismatch_count`
- `shuffle_self_fallback_count`
- per-condition `clean_eval_hash`
- whether all expected seeds are present

Any nonzero token or sequence mismatch invalidates replacement-style conditions.
Any nonzero shuffle self-fallback should be disclosed or rerun with
`--strict-shuffle`.

For a no-self same-length shuffled control, use the strict matched variant:

```bash
python -m rosetta.experiments.phase1.h1_final_runner \
  --conditions shuffled_translation_strict_matched \
  --directions fwd \
  --seeds 42 123 456 \
  --test-file data/tasks/multi_hop_reasoning/clean_eval.jsonl \
  --output-dir papers/h1_activation_transfer/results/final_strict_controls
```

This variant excludes prompt-length singleton buckets. In the current clean eval,
that means `n=383` from a source set of 396 samples and
`shuffle_self_fallback_count=0`.

Validate the main final directory plus strict matched controls:

```bash
python papers/h1_activation_transfer/scripts/validate_final_results.py \
  --results-dir papers/h1_activation_transfer/results/final \
                papers/h1_activation_transfer/results/final_strict_controls \
  --include-strict-matched \
  --output-dir papers/h1_activation_transfer/results/final_with_strict_controls
```

Generate paired comparisons including strict matched controls:

```bash
python papers/h1_activation_transfer/scripts/compare_results.py \
  --results-dir papers/h1_activation_transfer/results/final \
                papers/h1_activation_transfer/results/final_strict_controls \
  --output-dir papers/h1_activation_transfer/results/final_with_strict_controls
```

## 5. Optional Reverse Direction

Run reverse conditions only after the forward path is stable:

```bash
python -m rosetta.experiments.phase1.h1_final_runner \
  --conditions no_inject replace scale_corrected \
  --directions rev \
  --seeds 42 123 456 \
  --test-file data/tasks/multi_hop_reasoning/clean_eval.jsonl \
  --output-dir papers/h1_activation_transfer/results/final
```

Reverse-direction results should usually be appendix evidence unless the paper
argument is expanded to include direction asymmetry.

## 6. External Audit Point

Request external audit after:

1. Readiness check passes on the execution machine.
2. Smoke tests pass and produce valid summaries.
3. The primary clean rerun has complete per-sample JSON outputs.
4. `summarize_results.py` has produced final tables.
5. No replacement-style condition has token or sequence mismatches.

The audit should review protocol adherence, data hash, checkpoint provenance,
per-sample result schema, metric implementation, and the main result table.
