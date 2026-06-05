# Translation Quality and Norm Diagnostics

This appendix records the translation-layer quality evidence used to interpret
the H1 clean rerun. These diagnostics are supporting evidence only; downstream
task metrics are computed from the final clean-eval result JSON files.

## Source Files

- `results/phase1/m6_training_summary.json`
- `results/phase1/m6_norm_analysis.json`
- `papers/h1_activation_transfer/results/final/*.json`

## M6 Translation Training Summary

The primary paper direction is Pythia-160M -> Pythia-410M. The corrected M6
linear translation layers were trained with L2-normalized receiver activations
as targets.

| Direction | Seed | R2(norm) | Cosine(norm) | R2(raw) | Cosine(raw) | Val loss reduction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fwd | 42 | 0.881843 | 0.973414 | 0.031291 | 0.850938 | 85.37% |
| fwd | 123 | 0.882449 | 0.973525 | 0.030759 | 0.835464 | 84.78% |
| fwd | 456 | 0.884028 | 0.973639 | 0.032357 | 0.840831 | 84.57% |
| fwd mean | - | 0.882773 | 0.973526 | 0.031469 | 0.842411 | 84.91% |

Reverse-direction M6 runs also pass the normalized-space diagnostic
(`R2(norm)=0.875187`, `Cosine(norm)=0.971926` mean), but reverse-direction
clean reruns are not part of the current main result table.

## Norm Scale Diagnostics

The translation objective produces vectors in normalized activation space. This
creates a large raw norm mismatch if translated activations are used as direct
replacement residual-stream states.

Historical M6 norm analysis reported:

| Quantity | Mean |
| --- | ---: |
| receiver hidden norm | 72.3704 |
| translated output norm | 0.8504 |
| receiver/translated scale ratio | 85.1 |

That historical norm file was produced during M6 analysis and records
`gpt_neox.layers.15` for the receiver layer. The final protocol resolves the
primary receiver layer to `gpt_neox.layers.16`, so the final clean-rerun result
diagnostics are the authoritative layer-specific evidence.

Final clean-rerun diagnostics show the same scale issue:

| Condition | Runs | Mean translated norm | Mean receiver hidden norm | Mean injected norm |
| --- | ---: | ---: | ---: | ---: |
| additive | 3 | 0.87 | 68.70 | 0.01 |
| replace | 3 | 0.85 | 68.70 | 0.85 |
| scale_corrected | 3 | 0.85 | 68.70 | 68.70 |
| best_alpha | 3 | 0.85 | 68.70 | 66.72 |
| shuffled_translation | 3 | 0.85 | 68.70 | 68.70 |

## Interpretation

The translation layers learn a stable normalized-space map: normalized R2 and
cosine similarity are high and consistent across seeds. Raw-space R2 remains
near zero in the primary direction, and direct replacement inserts vectors with
norm about two orders of magnitude smaller than native receiver hidden states.

The scale-corrected conditions address this norm mismatch by normalizing the
translated activations and rescaling them to the receiver hidden-state norm.
Because scale-corrected replacement still performs far below no-injection and
natural-language relay, the clean rerun does not support the explanation that
the failure is caused only by activation norm mismatch.
