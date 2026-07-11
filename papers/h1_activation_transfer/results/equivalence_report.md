# H1 Equivalence Analysis (TOST)

Primary question: is the low-strength additive injection statistically
equivalent to the baselines within a practical margin?  Margins were
pre-specified for this reanalysis (chosen at analysis time, after the
frozen rerun; they are not pre-registered).  The clustered view
averages each sample's paired delta across runs, so the shared eval
set does not understate the standard error.  TOST verdicts use
alpha=0.05.

## additive vs no_inject

| Metric | View | N | Delta | 90% CI | TOST p (margin 0.05) | TOST p (margin 0.03) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| word_boundary_contains | seed 42 | 396 | +0.0000 | [-0.0132, +0.0132] | 2.01e-10 | 8.77e-05 |
| word_boundary_contains | seed 123 | 396 | +0.0000 | [-0.0118, +0.0118] | 1.36e-12 | 1.36e-05 |
| word_boundary_contains | seed 456 | 396 | +0.0126 | [+0.0002, +0.0251] | 3.82e-07 | 1.08e-02 |
| word_boundary_contains | clustered | 396 | +0.0042 | [-0.0053, +0.0137] | 1.11e-15 | 3.97e-06 |
| legacy_contains | seed 42 | 396 | +0.0000 | [-0.0132, +0.0132] | 2.01e-10 | 8.77e-05 |
| legacy_contains | seed 123 | 396 | +0.0025 | [-0.0113, +0.0163] | 7.49e-09 | 5.25e-04 |
| legacy_contains | seed 456 | 396 | +0.0152 | [+0.0021, +0.0282] | 5.98e-06 | 3.11e-02 |
| legacy_contains | clustered | 396 | +0.0059 | [-0.0049, +0.0167] | 9.90e-12 | 1.23e-04 |

## additive vs nl_relay

| Metric | View | N | Delta | 90% CI | TOST p (margin 0.05) | TOST p (margin 0.03) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| word_boundary_contains | seed 42 | 396 | +0.0051 | [-0.0266, +0.0367] | 9.78e-03 | 9.75e-02 |
| word_boundary_contains | seed 123 | 396 | +0.0051 | [-0.0266, +0.0367] | 9.78e-03 | 9.75e-02 |
| word_boundary_contains | seed 456 | 396 | +0.0177 | [-0.0148, +0.0501] | 5.07e-02 | 2.66e-01 |
| word_boundary_contains | clustered | 396 | +0.0093 | [-0.0216, +0.0402] | 1.50e-02 | 1.35e-01 |
| legacy_contains | seed 42 | 396 | +0.0126 | [-0.0204, +0.0456] | 3.12e-02 | 1.93e-01 |
| legacy_contains | seed 123 | 396 | +0.0152 | [-0.0181, +0.0484] | 4.23e-02 | 2.31e-01 |
| legacy_contains | seed 456 | 396 | +0.0278 | [-0.0062, +0.0617] | 1.41e-01 | 4.57e-01 |
| legacy_contains | clustered | 396 | +0.0185 | [-0.0140, +0.0510] | 5.54e-02 | 2.80e-01 |

## Stronger-transfer variants vs no_inject (one-sided degradation)

| Condition | Metric | N (clustered) | Delta | 90% CI | p(worse) |
| --- | --- | ---: | ---: | ---: | ---: |
| best_alpha | word_boundary_contains | 396 | -0.0396 | [-0.0596, -0.0195] | 5.90e-04 |
| best_alpha | legacy_contains | 396 | -0.0320 | [-0.0530, -0.0109] | 6.20e-03 |
| scale_corrected | word_boundary_contains | 396 | -0.0808 | [-0.1037, -0.0579] | 3.20e-09 |
| scale_corrected | legacy_contains | 396 | -0.0892 | [-0.1142, -0.0642] | 2.22e-09 |
| replace | word_boundary_contains | 396 | -0.0859 | [-0.1095, -0.0622] | 1.23e-09 |
| replace | legacy_contains | 396 | -0.0934 | [-0.1188, -0.0681] | 6.52e-10 |
| same_norm_random | word_boundary_contains | 396 | -0.0884 | [-0.1119, -0.0649] | 3.04e-10 |
| same_norm_random | legacy_contains | 396 | -0.1027 | [-0.1277, -0.0776] | 7.70e-12 |
| zero_replacement | word_boundary_contains | 396 | -0.0859 | [-0.1098, -0.0619] | 1.77e-09 |
| zero_replacement | legacy_contains | 396 | -0.1010 | [-0.1266, -0.0754] | 4.46e-11 |
| b_to_b_self_inject | word_boundary_contains | 396 | +0.0000 | [+0.0000, +0.0000] | 1.00e+00 |
| b_to_b_self_inject | legacy_contains | 396 | +0.0000 | [+0.0000, +0.0000] | 1.00e+00 |

## Full-causal-weight controls: pairwise practical equivalence

Tests whether replacement with translated states is practically
equivalent to replacement with random / zero-vector controls
(clustered view; runs paired by seed, deterministic runs paired with
every seed).

| Pair | Metric | N | Delta | 90% CI | TOST p (margin 0.05) | TOST p (margin 0.03) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| replace_vs_same_norm_random | word_boundary_contains | 396 | +0.0025 | [+0.0001, +0.0049] | 0.00e+00 | 0.00e+00 |
| replace_vs_same_norm_random | legacy_contains | 396 | +0.0093 | [+0.0043, +0.0142] | 0.00e+00 | 2.48e-12 |
| replace_vs_zero_replacement | word_boundary_contains | 396 | -0.0000 | [-0.0048, +0.0048] | 0.00e+00 | 0.00e+00 |
| replace_vs_zero_replacement | legacy_contains | 396 | +0.0076 | [+0.0010, +0.0142] | 0.00e+00 | 1.25e-08 |
| same_norm_random_vs_zero_replacement | word_boundary_contains | 396 | -0.0025 | [-0.0067, +0.0016] | 0.00e+00 | 0.00e+00 |
| same_norm_random_vs_zero_replacement | legacy_contains | 396 | -0.0017 | [-0.0061, +0.0027] | 0.00e+00 | 0.00e+00 |
