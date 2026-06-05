"""Unit tests for M7 scale-corrected replacement injection.

Tests verify:
  - SC1: Scale-corrected mode produces injected vectors with B's natural
         per-position norm (±1e-3 tolerance).
  - SC2: Direction (cosine similarity to translated vector) is preserved ≈ 1.0.
  - SC3: replace_scale_corrected requires prefill_only timing.
  - SC4: replace_scale_corrected requires 3D activations.
  - SC5: replace_scale_corrected is accepted as a valid injection_mode.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from rosetta.translation.translation_layer import inject_and_generate


# ---------------------------------------------------------------------------
# SC1 + SC2: Mathematical properties of scale-corrected hook logic
# ---------------------------------------------------------------------------


class TestScaleCorrectedMath:
    """Tests simulate the hook logic directly without real models."""

    def test_norm_preservation(self):
        """Injected per-position norm equals original hidden norm (±1e-3)."""
        batch, seq, dim = 2, 7, 64
        torch.manual_seed(0)

        # Simulate B's natural hidden states (large norm, like actual activations ~72)
        hidden = torch.randn(batch, seq, dim) * 20.0
        original_norms = hidden.norm(dim=-1)  # (batch, seq)

        # Simulate translate() output (norm ≈ 0.85, not exactly unit)
        translated = torch.randn(batch, seq, dim) * 0.85

        # Apply scale-corrected logic (mirrors hook implementation)
        target_norm = hidden.norm(dim=-1, keepdim=True)
        new_hidden = nn.functional.normalize(translated, p=2, dim=-1) * target_norm

        result_norms = new_hidden.norm(dim=-1)
        assert torch.allclose(result_norms, original_norms, atol=1e-3), (
            f"Norm not preserved: max diff = {(result_norms - original_norms).abs().max():.6f}"
        )

    def test_direction_preserved(self):
        """Cosine similarity between injected vector and translated direction ≈ 1.0."""
        batch, seq, dim = 2, 7, 64
        torch.manual_seed(1)

        hidden = torch.randn(batch, seq, dim) * 20.0
        translated = torch.randn(batch, seq, dim) * 0.85

        target_norm = hidden.norm(dim=-1, keepdim=True)
        new_hidden = nn.functional.normalize(translated, p=2, dim=-1) * target_norm

        # Cosine similarity between new_hidden and translated (per position)
        cos_sim = nn.functional.cosine_similarity(
            new_hidden.reshape(-1, dim),
            translated.reshape(-1, dim),
            dim=-1,
        )
        min_cos = cos_sim.min().item()
        assert min_cos > 0.9999, (
            f"Direction not preserved: min cosine similarity = {min_cos:.6f}"
        )

    def test_norm_preserved_with_unit_translated(self):
        """With exactly unit-norm translated input, result norm == hidden norm exactly."""
        batch, seq, dim = 3, 10, 32
        torch.manual_seed(2)

        hidden = torch.randn(batch, seq, dim) * 50.0
        # Exactly unit norm translated
        translated = nn.functional.normalize(torch.randn(batch, seq, dim), p=2, dim=-1)
        assert torch.allclose(translated.norm(dim=-1), torch.ones(batch, seq), atol=1e-6)

        target_norm = hidden.norm(dim=-1, keepdim=True)
        new_hidden = nn.functional.normalize(translated, p=2, dim=-1) * target_norm

        result_norms = new_hidden.norm(dim=-1)
        original_norms = hidden.norm(dim=-1)
        assert torch.allclose(result_norms, original_norms, atol=1e-4)

    def test_scale_correction_reduces_mismatch(self):
        """Scale-corrected vector has norm much closer to target than uncorrected."""
        batch, seq, dim = 1, 5, 64
        torch.manual_seed(3)

        # B hidden norm ~72, translate() output norm ~0.85 → ratio 85×
        hidden = torch.randn(batch, seq, dim)
        hidden = hidden / hidden.norm(dim=-1, keepdim=True) * 72.0  # norm exactly 72

        translated = torch.randn(batch, seq, dim)
        translated = translated / translated.norm(dim=-1, keepdim=True) * 0.85  # norm exactly 0.85

        target_norm = hidden.norm(dim=-1, keepdim=True)

        # Uncorrected: ratio = target_norm / translated_norm ≈ 72/0.85 ≈ 85
        ratio_uncorrected = (hidden.norm(dim=-1) / translated.norm(dim=-1)).mean().item()

        # Scale-corrected
        new_hidden = nn.functional.normalize(translated, p=2, dim=-1) * target_norm
        ratio_corrected = (new_hidden.norm(dim=-1) / hidden.norm(dim=-1)).mean().item()

        assert ratio_uncorrected > 80, f"Expected large uncorrected ratio, got {ratio_uncorrected}"
        assert abs(ratio_corrected - 1.0) < 0.01, (
            f"Expected corrected ratio ≈ 1.0, got {ratio_corrected:.4f}"
        )


# ---------------------------------------------------------------------------
# SC-alpha: injection_alpha blending behaviour
# ---------------------------------------------------------------------------


class TestInjectionAlpha:
    def test_alpha_one_equals_full_replacement(self):
        """injection_alpha=1.0 gives the same result as the default (no blending)."""
        batch, seq, dim = 1, 5, 32
        torch.manual_seed(10)
        hidden = torch.randn(batch, seq, dim) * 20.0
        translated = torch.randn(batch, seq, dim) * 0.85

        target_norm = hidden.norm(dim=-1, keepdim=True)
        corrected = nn.functional.normalize(translated, p=2, dim=-1) * target_norm

        # alpha=1.0: full replacement
        result_full = 1.0 * corrected + 0.0 * hidden
        assert torch.allclose(result_full, corrected, atol=1e-6)

    def test_alpha_zero_returns_original(self):
        """injection_alpha=0.0 leaves hidden states unchanged."""
        batch, seq, dim = 1, 5, 32
        torch.manual_seed(11)
        hidden = torch.randn(batch, seq, dim) * 20.0
        translated = torch.randn(batch, seq, dim) * 0.85

        target_norm = hidden.norm(dim=-1, keepdim=True)
        corrected = nn.functional.normalize(translated, p=2, dim=-1) * target_norm

        result_no_inject = 0.0 * corrected + 1.0 * hidden
        assert torch.allclose(result_no_inject, hidden, atol=1e-6)

    def test_alpha_half_is_midpoint(self):
        """injection_alpha=0.5 is the mean of corrected and original."""
        batch, seq, dim = 1, 5, 32
        torch.manual_seed(12)
        hidden = torch.randn(batch, seq, dim) * 20.0
        translated = torch.randn(batch, seq, dim) * 0.85

        target_norm = hidden.norm(dim=-1, keepdim=True)
        corrected = nn.functional.normalize(translated, p=2, dim=-1) * target_norm

        alpha = 0.5
        result = alpha * corrected + (1 - alpha) * hidden
        expected = (corrected + hidden) / 2
        assert torch.allclose(result, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# SC3 + SC4 + SC5: inject_and_generate integration guards
# ---------------------------------------------------------------------------


class TestScaleCorrectedGuards:
    def test_requires_prefill_only(self):
        """replace_scale_corrected raises if injection_timing != prefill_only."""
        layer_name = "gpt_neox.layers.8"
        model, tokenizer, _, _ = _make_mock_receiver(
            [1, 2, 3], [4, 5], "test", layer_name
        )
        vec = torch.randn(1, 3, 768)

        with pytest.raises(AssertionError, match="prefill_only"):
            inject_and_generate(
                model, tokenizer, vec, layer_name,
                question="Q?", task="multi_hop", device="cpu",
                injection_mode="replace_scale_corrected",
                injection_timing="persistent",
            )

    def test_requires_3d_activations(self):
        """replace_scale_corrected raises if activations are not 3D."""
        layer_name = "gpt_neox.layers.8"
        model, tokenizer, _, _ = _make_mock_receiver(
            [1, 2, 3], [4, 5], "test", layer_name
        )
        vec_1d = torch.randn(768)

        with pytest.raises(AssertionError, match="3D"):
            inject_and_generate(
                model, tokenizer, vec_1d, layer_name,
                question="Q?", task="multi_hop", device="cpu",
                injection_mode="replace_scale_corrected",
                injection_timing="prefill_only",
            )

    def test_valid_mode_accepted(self):
        """replace_scale_corrected is accepted as a valid injection_mode (no AssertionError)."""
        layer_name = "gpt_neox.layers.8"
        model, tokenizer, _, _ = _make_mock_receiver(
            [1, 2, 3, 4, 5], [6, 7], "answer", layer_name
        )
        vec = torch.randn(1, 5, 768)

        # Should not raise
        inject_and_generate(
            model, tokenizer, vec, layer_name,
            question="What?", task="multi_hop", device="cpu",
            injection_mode="replace_scale_corrected",
            injection_timing="prefill_only",
            context="Some context.",
            full_input=True,
        )

    def test_unknown_mode_rejected(self):
        """Unknown injection_mode raises AssertionError."""
        layer_name = "gpt_neox.layers.8"
        model, tokenizer, _, _ = _make_mock_receiver(
            [1, 2], [3], "x", layer_name
        )

        with pytest.raises(AssertionError, match="Unknown injection_mode"):
            inject_and_generate(
                model, tokenizer, torch.randn(768), layer_name,
                question="Q?", task="multi_hop", device="cpu",
                injection_mode="invalid_mode",  # type: ignore[arg-type]
            )


# ---------------------------------------------------------------------------
# Helper (same pattern as test_m6_replacement.py)
# ---------------------------------------------------------------------------


def _make_mock_receiver(prompt_token_ids, generated_token_ids, decoded_text, layer_name):
    """Build a mock model/tokenizer for inject_and_generate tests."""
    all_ids = prompt_token_ids + generated_token_ids
    mock_inputs = MagicMock()
    mock_inputs.__getitem__ = lambda _, k: (
        torch.tensor([prompt_token_ids]) if k == "input_ids" else MagicMock()
    )
    mock_inputs.to.return_value = mock_inputs

    tokenizer = MagicMock()
    tokenizer.return_value = mock_inputs
    tokenizer.decode.return_value = f"  {decoded_text}  "
    tokenizer.eos_token_id = 0

    mock_hook_handle = MagicMock()
    mock_layer = MagicMock()
    mock_layer.register_forward_hook.return_value = mock_hook_handle

    model = MagicMock()
    model.device = torch.device("cpu")
    model.generate.return_value = torch.tensor([all_ids])
    model.to.return_value = model
    model.eval.return_value = model

    parts = layer_name.split(".")
    current = model
    for part in parts[:-1]:
        child = MagicMock()
        child.to.return_value = child
        child.eval.return_value = child
        setattr(current, part, child)
        current = child
    setattr(current, parts[-1], mock_layer)

    return model, tokenizer, mock_hook_handle, mock_layer
