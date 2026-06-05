# Strict Matched Shuffle Control Report

## Reason

The original `shuffled_translation` control preserves replacement-injection
shape safety by shuffling translated activations only within identical
`prompt_len` buckets. This is necessary because replacement-style injection
requires the donor activation tensor and target receiver prompt to have exactly
the same sequence length.

In the clean evaluation set, 13 samples have unique prompt lengths. A same-length
bucket containing one sample cannot be deranged without mapping that sample to
itself. The full-n shuffled control therefore recorded
`shuffle_self_fallback_count=13` for each seed.

## Strict Matched Variant

The strict variant uses condition:

```text
shuffled_translation_strict_matched
```

It drops prompt-length singleton buckets and shuffles only buckets with at least
two samples.

Result:

- source clean eval size: 396
- evaluated size: 383
- excluded singleton prompt-length samples: 13
- `shuffle_self_fallback_count`: 0
- `seq_len_mismatch_count`: 0
- `token_mismatch_count`: 0

## Summary

| Condition | Runs | N | Boundary Contains | Legacy Contains |
| --- | ---: | ---: | ---: | ---: |
| shuffled_translation_strict_matched | 3 | 383 | 0.0052 +/- 0.0045 | 0.0087 +/- 0.0060 |

Pairwise comparison against `no_inject`, using word-boundary contains:

- delta: `-0.0862`
- pooled percentile bootstrap 95% CI: `[-0.1036, -0.0696]`
- mean gained samples: `1.0`
- mean lost samples: `34.0`

Conclusion: the stricter no-self shuffled control matches the qualitative result
of the full-n shuffled control while removing the self-fallback caveat.
