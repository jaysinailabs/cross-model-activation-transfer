# H1 Internal Audit Remediation

Follow-up to `internal_audit.md`.

## Completed P2 Items

| Item | Remediation |
| --- | --- |
| Translation quality summary | Added `appendix/translation_quality.md` with normalized-space R2/cosine, raw-space diagnostics, and final clean-rerun norm evidence. |
| Frozen defaults | Added `Frozen Runner Defaults` to `protocol/final_protocol.md`, covering `max_length=256`, `injection_scale=0.01`, `best_alpha=0.30`, `shuffle_injection_mode=scale_corrected`, decoding budgets, and layer depth. |
| clean_eval reproducibility test | Added `tests/test_h1_clean_eval_reproducible.py`; it rebuilds clean_eval to a temp path and checks `n=396`, acceptance, and SHA-256 `504e077c...dcbb`. |
| Pooled CI | Updated `scripts/compare_results.py` so grouped CIs pool paired sample-level deltas across runs and use a deterministic percentile bootstrap. |
| Protocol split 9a/9b | Split shuffled controls in `protocol/final_protocol.md`; updated `experiment_matrix.md`, `execution_runbook.md`, and `validate_final_results.py`. |

## Completed P3 Items

| Item | Remediation |
| --- | --- |
| deterministic reason | Future runner outputs now include `deterministic_reason` when `deterministic=true`. Existing generated final JSON files were not rewritten. |
| punctuation note | Added punctuation-loss note to `protocol/metrics.md`. |
| percentile method note | Added percentile-bootstrap notes to `protocol/metrics.md` and comparison markdown output. |
| strict relation note | Updated `paper/main.md` to explain full-n shuffle versus strict matched shuffle. |
| audit flag name | Renamed public CLI to `--per-result-file`; kept hidden `--per-condition` compatibility. |
| audit labels | Regenerated `metric_disagreement_audit.csv/md` with rule-based labels: 5 `verbose_correct`, 31 `substring_noise`, 29 `accidental_mention`. |

## Regenerated Outputs

- `papers/h1_activation_transfer/results/final_pairwise_comparisons.*`
- `papers/h1_activation_transfer/results/final_with_strict_controls/final_pairwise_comparisons.*`
- `papers/h1_activation_transfer/results/final_validation_report.*`
- `papers/h1_activation_transfer/results/final_with_strict_controls/final_validation_report.*`
- `papers/h1_activation_transfer/results/metric_disagreement_audit.*`

## Current Validation Status

- Main final directory: 21 / 21 expected files, blocking checks pass.
- Final plus strict matched controls: 24 / 24 expected files, blocking checks pass.
- Remaining warning: original full-n `shuffled_translation` has
  `shuffle_self_fallback_count=13` for each seed. This is now documented as a
  diagnostic/appendix control; strict matched is the cleaner no-self shuffled
  control.

## Post External-Audit Transparency Pass

Follow-up to `internal_audit.md` § 9.7:

| Residual item | Action |
| --- | --- |
| Existing final JSON files lack `deterministic_reason` | No backfill. The generated-result history is preserved; future runner outputs include the field, and this limitation remains disclosed. |
| Full-n shuffled warning may look concerning | `paper/main.md` now states that these warnings are expected for the diagnostic full-n shuffled control and not attached to the paper-primary strict matched control. |
| Rule-derived audit labels need spot-checking | Added `appendix/metric_disagreement_spot_check.md` with 10 non-scripted checks across all three label classes; all agree with the rule labels. |
| `best_alpha=0.30` came from an old pilot | `paper/main.md` now states that alpha 0.30 was frozen from the historical M7 pilot before the clean-eval final rerun, avoiding clean_eval cherry-picking. |
