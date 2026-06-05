# H1 Activation Transfer Artifact Workspace

This directory contains the public code-and-data artifact for the H1
activation-transfer negative-result study within Project Rosetta.

It is intentionally separate from the historical experiment logs in `plans/`
and from the manuscript source tree. The arXiv paper is published separately;
this workspace is for clean protocol files, clean evaluation data, locked
result files, validation scripts, and reproducibility material.

## Scope

H1 asks whether model-to-model communication through translated hidden
activations can outperform natural-language relay.

The current evidence suggests a negative result in the tested setting:

- model pair: Pythia-160M -> Pythia-410M
- task: multi-hop reasoning
- injection depth: about 67 percent
- method: sequence-level activation translation and inference-time injection
- finding: translation directions are learnable, but injected activations do not
  outperform natural-language relay or no-inject baselines

## Included Files

- `protocol/final_protocol.md`: Frozen protocol for the clean reruns.
- `protocol/experiment_matrix.md`: Final experiment conditions and required
  outputs.
- `protocol/metrics.md`: Metric definitions and reporting rules.
- `protocol/execution_runbook.md`: Execution checklist, smoke commands, and
  audit points.
- `appendix/translation_quality.md`: Translation-layer quality and norm
  diagnostics.
- `appendix/metric_disagreement_spot_check.md`: Non-scripted spot-check of
  rule-derived metric-disagreement labels.
- `data_audit/dataset_audit.md`: Dataset audit procedure and clean-eval policy.
- `scripts/build_clean_eval.py`: Deterministic clean-eval builder.
- `scripts/check_readiness.py`: Local readiness check for clean eval,
  checkpoints, model cache, and CUDA.
- `scripts/summarize_results.py`: Recomputes paper metrics from final
  per-sample JSON files.
- `scripts/validate_final_results.py`: Checks final result completeness, clean
  hash, sample count, and injection diagnostics.
- `scripts/compare_results.py`: Computes paired deltas, bootstrap intervals, and
  McNemar-style counts.
- `scripts/build_metric_disagreement_audit.py`: Builds a manual audit table for
  metric disagreements.
- `scripts/generate_figures.py`: Regenerates paper figures from locked result
  summaries when manuscript assets are available.
- `src/rosetta/experiments/phase1/h1_final_runner.py`: Unified clean-rerun
  entry point.

## Current Execution Flow

Build the clean evaluation file:

```bash
python papers/h1_activation_transfer/scripts/build_clean_eval.py
```

Check local readiness:

```bash
python papers/h1_activation_transfer/scripts/check_readiness.py --require-model-cache
```

Run final conditions, for example the primary forward main matrix:

```bash
python -m rosetta.experiments.phase1.h1_final_runner \
  --conditions main \
  --directions fwd \
  --test-file data/tasks/multi_hop_reasoning/clean_eval.jsonl \
  --output-dir papers/h1_activation_transfer/results/final
```

Generate paper-facing summaries after result JSON files exist:

```bash
python papers/h1_activation_transfer/scripts/summarize_results.py
```

Validate final result files:

```bash
python papers/h1_activation_transfer/scripts/validate_final_results.py
```

Generate paired comparisons and metric-disagreement audit material:

```bash
python papers/h1_activation_transfer/scripts/compare_results.py
python papers/h1_activation_transfer/scripts/build_metric_disagreement_audit.py --per-result-file 5
```

Strict no-self shuffled controls, if used, live in:

```text
papers/h1_activation_transfer/results/final_strict_controls/
papers/h1_activation_transfer/results/final_strict_controls_summary/
papers/h1_activation_transfer/results/final_with_strict_controls/
```

To validate the main final result directory together with strict matched
shuffle controls:

```bash
python papers/h1_activation_transfer/scripts/validate_final_results.py \
  --results-dir papers/h1_activation_transfer/results/final \
                papers/h1_activation_transfer/results/final_strict_controls \
  --include-strict-matched \
  --output-dir papers/h1_activation_transfer/results/final_with_strict_controls
```

## Non-Goals

This workspace does not replace Project Rosetta's broader plans, phase logs, or
future H2/H3/H4 work. It only covers the H1 paper artifact.

Historical M3-M7 results should not be copied directly into the paper's main
tables. Main claims should come from final clean reruns defined in `protocol/`.
