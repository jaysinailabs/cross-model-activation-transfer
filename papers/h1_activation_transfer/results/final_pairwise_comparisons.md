# H1 Pairwise Comparisons

Grouped confidence intervals pool paired sample-level deltas across runs and use a percentile bootstrap interval; they are not BCa intervals.

| Condition | Baseline | Metric | Runs | Delta | 95% CI | Gained | Lost |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| additive | nl_relay | legacy_contains | 3 | 0.0185 | [-0.0034, 0.0429] | 36.0 | 28.7 |
| additive | nl_relay | word_boundary_contains | 3 | 0.0093 | [-0.0126, 0.0311] | 31.3 | 27.7 |
| additive | no_inject | legacy_contains | 3 | 0.0059 | [-0.0034, 0.0152] | 6.3 | 4.0 |
| additive | no_inject | word_boundary_contains | 3 | 0.0042 | [-0.0042, 0.0126] | 5.3 | 3.7 |
| b_to_b_self_inject | nl_relay | legacy_contains | 1 | 0.0126 | [-0.0278, 0.0530] | 35.0 | 30.0 |
| b_to_b_self_inject | nl_relay | word_boundary_contains | 1 | 0.0051 | [-0.0328, 0.0429] | 30.0 | 28.0 |
| b_to_b_self_inject | no_inject | legacy_contains | 1 | 0.0000 | [0.0000, 0.0000] | 0.0 | 0.0 |
| b_to_b_self_inject | no_inject | word_boundary_contains | 1 | 0.0000 | [0.0000, 0.0000] | 0.0 | 0.0 |
| best_alpha | nl_relay | legacy_contains | 3 | -0.0194 | [-0.0396, 0.0000] | 21.0 | 28.7 |
| best_alpha | nl_relay | word_boundary_contains | 3 | -0.0345 | [-0.0539, -0.0152] | 15.7 | 29.3 |
| best_alpha | no_inject | legacy_contains | 3 | -0.0320 | [-0.0480, -0.0160] | 9.3 | 22.0 |
| best_alpha | no_inject | word_boundary_contains | 3 | -0.0396 | [-0.0547, -0.0244] | 6.3 | 22.0 |
| replace | nl_relay | legacy_contains | 3 | -0.0808 | [-0.0976, -0.0640] | 2.7 | 34.7 |
| replace | nl_relay | word_boundary_contains | 3 | -0.0808 | [-0.0968, -0.0648] | 0.7 | 32.7 |
| replace | no_inject | legacy_contains | 3 | -0.0934 | [-0.1120, -0.0758] | 3.0 | 40.0 |
| replace | no_inject | word_boundary_contains | 3 | -0.0859 | [-0.1019, -0.0699] | 1.0 | 35.0 |
| same_norm_random | nl_relay | legacy_contains | 3 | -0.0901 | [-0.1061, -0.0741] | 0.3 | 36.0 |
| same_norm_random | nl_relay | word_boundary_contains | 3 | -0.0833 | [-0.0993, -0.0682] | 0.0 | 33.0 |
| same_norm_random | no_inject | legacy_contains | 3 | -0.1027 | [-0.1204, -0.0850] | 0.0 | 40.7 |
| same_norm_random | no_inject | word_boundary_contains | 3 | -0.0884 | [-0.1052, -0.0724] | 0.0 | 35.0 |
| scale_corrected | nl_relay | legacy_contains | 3 | -0.0766 | [-0.0943, -0.0598] | 3.3 | 33.7 |
| scale_corrected | nl_relay | word_boundary_contains | 3 | -0.0758 | [-0.0918, -0.0589] | 1.7 | 31.7 |
| scale_corrected | no_inject | legacy_contains | 3 | -0.0892 | [-0.1069, -0.0715] | 3.0 | 38.3 |
| scale_corrected | no_inject | word_boundary_contains | 3 | -0.0808 | [-0.0968, -0.0657] | 1.0 | 33.0 |
| shuffled_translation | nl_relay | legacy_contains | 3 | -0.0816 | [-0.0976, -0.0657] | 2.0 | 34.3 |
| shuffled_translation | nl_relay | word_boundary_contains | 3 | -0.0774 | [-0.0926, -0.0623] | 1.3 | 32.0 |
| shuffled_translation | no_inject | legacy_contains | 3 | -0.0943 | [-0.1120, -0.0766] | 2.0 | 39.3 |
| shuffled_translation | no_inject | word_boundary_contains | 3 | -0.0825 | [-0.0993, -0.0665] | 1.3 | 34.0 |
| zero_replacement | nl_relay | legacy_contains | 1 | -0.0884 | [-0.1187, -0.0606] | 1.0 | 36.0 |
| zero_replacement | nl_relay | word_boundary_contains | 1 | -0.0808 | [-0.1086, -0.0530] | 1.0 | 33.0 |
| zero_replacement | no_inject | legacy_contains | 1 | -0.1010 | [-0.1338, -0.0732] | 1.0 | 41.0 |
| zero_replacement | no_inject | word_boundary_contains | 1 | -0.0859 | [-0.1162, -0.0606] | 1.0 | 35.0 |
