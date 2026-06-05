# Metric Disagreement Spot-Check

This spot-check reviews 10 rows from
`papers/h1_activation_transfer/results/metric_disagreement_audit.csv` without
using the audit-label rule implementation. It is a transparency check on the
rule-derived labels, not a complete independent adjudication of all 65 rows.

## Label Rule Under Check

- `verbose_correct`: the model emits the gold answer first, then continues.
- `substring_noise`: legacy contains fires only because the gold string appears
  inside a larger word or phrase.
- `accidental_mention`: the gold answer appears as a standalone mention, but
  the surrounding answer is not a correct direct answer.

## Checked Rows

| Condition | Seed | Sample | Gold | Rule Label | Spot-Check Judgment | Rationale |
| --- | ---: | --- | --- | --- | --- | --- |
| nl_relay |  | `mh_clean_000007` | Indonesia | `verbose_correct` | agree | Output begins with `Indonesia.` before continuing. |
| nl_relay |  | `mh_clean_000011` | Austria | `verbose_correct` | agree | Output begins with `Austria.` before adding explanation. |
| nl_relay |  | `mh_clean_000022` | Japan | `verbose_correct` | agree | Output begins with `Japan.` and then continues. |
| additive | 123 | `mh_clean_000015` | Africa | `substring_noise` | agree | Legacy hit is from `African`, not standalone `Africa`. |
| additive | 42 | `mh_clean_000026` | Japan | `substring_noise` | agree | Legacy hit is from `Japanese`, not standalone `Japan`. |
| best_alpha | 123 | `mh_clean_000020` | India | `substring_noise` | agree | Legacy hit is from `Indian`, not standalone `India`. |
| replace | 456 | `mh_clean_000232` | retail | `substring_noise` | agree | Legacy hit is embedded in `retailers`; no direct answer span. |
| no_inject |  | `mh_clean_000000` | Asia | `accidental_mention` | agree | `Asia` is mentioned in a broad generated list, not as a direct answer. |
| scale_corrected | 123 | `mh_clean_000338` | energy | `accidental_mention` | agree | `energy` appears as part of a rambling completion, not a direct answer. |
| replace | 123 | `mh_clean_000015` | Africa | `accidental_mention` | agree | Standalone `Africa` appears in a definitional fragment, not as a clean answer. |

## Outcome

All 10 checked rows agree with the rule-derived audit label. This supports using
the rule labels as a reproducible diagnostic summary, while the paper should
still describe them as rule-derived rather than as full human adjudication.
