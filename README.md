# A Negative Result on Cross-Model Activation Transfer

This repository provides the code, clean evaluation data, and validated result
files for a paper on cross-model activation transfer. The manuscript itself
(arXiv report, technical report, and workshop variants) is not included here;
the paper is available on arXiv.

The experiment asks whether a learned linear map can send hidden states from a
Pythia-160M sender into a Pythia-410M receiver more usefully than a
natural-language relay in a multi-hop reasoning setting. The answer in this
tested setting is negative: the translation layer learns a strong offline
normalized-space map, but downstream activation injection does not improve the
receiver's answer containment over the no-injection or natural-language relay
baselines.

## Repository Contents

- `src/rosetta/`: Python package used by the experiment runners.
- `data/tasks/multi_hop_reasoning/`: source and clean evaluation JSONL files.
- `papers/h1_activation_transfer/protocol/`: frozen protocol, metric
  definitions, experiment matrix, and runbook.
- `papers/h1_activation_transfer/results/`: validated final result JSON files,
  summaries, pairwise comparisons, and metric-disagreement audit material.
- `papers/h1_activation_transfer/appendix/`, `.../data_audit/`: supporting
  analysis notes and dataset audit material.
- `tests/`: reproducibility and metric/tooling tests for the result artifacts.

## Not Included

The repository deliberately does not include the manuscript sources (paper
drafts, figures, and the arXiv/technical-report/workshop variants), model
weights, trained translation checkpoints, activation tensors, downloaded model
caches, raw WikiText corpus files, virtual environments, or local
browser-rendered HTML previews. The paper is published separately on arXiv; the
rest are large or environment-specific.

## Quick Validation

Install dependencies, then run:

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest tests/test_h1_clean_eval_reproducible.py tests/test_h1_final_tools.py tests/test_m7_scale_corrected.py
python papers/h1_activation_transfer/scripts/validate_final_results.py
python papers/h1_activation_transfer/scripts/validate_final_results.py \
  --results-dir papers/h1_activation_transfer/results/final \
                papers/h1_activation_transfer/results/final_strict_controls \
  --include-strict-matched \
  --output-dir papers/h1_activation_transfer/results/final_with_strict_controls
python papers/h1_activation_transfer/scripts/equivalence_analysis.py
```

The full clean rerun requires local Pythia model downloads and translation
checkpoint files under `results/phase1/checkpoints/`; those checkpoint files are
not stored in git.

## Privacy Note

This staged repository replaces local absolute paths in copied text artifacts
with repository-relative paths where possible, and excludes the manuscript
sources. It is intended as a public code-and-data artifact accompanying the
arXiv paper.
