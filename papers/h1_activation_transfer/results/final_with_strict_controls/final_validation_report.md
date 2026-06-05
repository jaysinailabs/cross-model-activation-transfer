# H1 Final Result Validation

- result files: 25 / 24
- direction filter: `fwd`
- expected n: 396
- strict matched expected n: 383
- expected clean eval hash: `504e077cf17433e22967c86e98d321532d4e803dbe24d96af14c7e8ecdd0dcbb`
- blocking checks passed: `True`
- ready for external audit: `True`

## Blocking

- none

## Warnings

- `extra_result_files_present`
- `shuffle_self_fallback:h1_shuffled_translation_fwd_seed123_alpha1p0.json`
- `shuffle_self_fallback:h1_shuffled_translation_fwd_seed42_alpha1p0.json`
- `shuffle_self_fallback:h1_shuffled_translation_fwd_seed456_alpha1p0.json`

## Per-Run Diagnostics

| Condition | Seed | N | Expected N | Contains | Seq Mismatch | Token Mismatch | Shuffle Fallback | Excluded Singletons |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| additive | 123 | 396 | 396 | 0.1061 | 0 | 0 | 0 |  |
| additive | 42 | 396 | 396 | 0.1035 | 0 | 0 | 0 |  |
| additive | 456 | 396 | 396 | 0.1187 | 0 | 0 | 0 |  |
| b_to_b_self_inject |  | 396 | 396 | 0.1035 | 0 | 0 | 0 |  |
| best_alpha | 123 | 396 | 396 | 0.0833 | 0 | 0 | 0 |  |
| best_alpha | 42 | 396 | 396 | 0.0581 | 0 | 0 | 0 |  |
| best_alpha | 456 | 396 | 396 | 0.0732 | 0 | 0 | 0 |  |
| nl_relay |  | 396 | 396 | 0.0909 | 0 | 0 | 0 |  |
| no_inject |  | 396 | 396 | 0.1035 | 0 | 0 | 0 |  |
| replace | 123 | 396 | 396 | 0.0076 | 0 | 0 | 0 |  |
| replace | 42 | 396 | 396 | 0.0076 | 0 | 0 | 0 |  |
| replace | 456 | 396 | 396 | 0.0152 | 0 | 0 | 0 |  |
| same_norm_random | 123 | 396 | 396 | 0.0000 | 0 | 0 | 0 |  |
| same_norm_random | 42 | 396 | 396 | 0.0000 | 0 | 0 | 0 |  |
| same_norm_random | 456 | 396 | 396 | 0.0025 | 0 | 0 | 0 |  |
| scale_corrected | 123 | 396 | 396 | 0.0126 | 0 | 0 | 0 |  |
| scale_corrected | 42 | 396 | 396 | 0.0101 | 0 | 0 | 0 |  |
| scale_corrected | 456 | 396 | 396 | 0.0202 | 0 | 0 | 0 |  |
| shuffled_translation | 123 | 396 | 396 | 0.0076 | 0 | 0 | 13 | 0 |
| shuffled_translation | 42 | 396 | 396 | 0.0051 | 0 | 0 | 13 | 0 |
| shuffled_translation | 456 | 396 | 396 | 0.0152 | 0 | 0 | 13 | 0 |
| zero_replacement |  | 396 | 396 | 0.0025 | 0 | 0 | 0 |  |
| shuffled_translation_strict_matched | 123 | 383 | 383 | 0.0052 | 0 | 0 | 0 | 13 |
| shuffled_translation_strict_matched | 42 | 383 | 383 | 0.0052 | 0 | 0 | 0 | 13 |
| shuffled_translation_strict_matched | 456 | 383 | 383 | 0.0157 | 0 | 0 | 0 | 13 |
