"""Unit tests for TranslationLayer and training utilities.

All tests use mock models and synthetic tensors — no real model weights are
loaded.  Tests verify:
  - TranslationLayer output shapes for all three architectures
  - L2 normalisation behaviour
  - Hook injection mechanism (additive residual)
  - Training loop convergence on a tiny synthetic problem
  - Checkpoint save / load round-trip
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from rosetta.translation.translation_layer import (
    TranslationLayer,
    inject_and_generate,
    load_translation_layer,
    train_translation_layer,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _random_acts(n: int, dim: int) -> torch.Tensor:
    """Return (n, dim) float tensor of random values."""
    return torch.randn(n, dim)


# ---------------------------------------------------------------------------
# 1. TranslationLayer: output shapes
# ---------------------------------------------------------------------------


class TestTranslationLayerShapes:
    def test_linear_same_dim(self):
        tl = TranslationLayer(768, 768, arch="linear")
        x = _random_acts(4, 768)
        out = tl(x)
        assert out.shape == (4, 768), f"Expected (4, 768), got {out.shape}"

    def test_linear_cross_dim(self):
        tl = TranslationLayer(768, 1024, arch="linear")
        x = _random_acts(4, 768)
        out = tl(x)
        assert out.shape == (4, 1024)

    def test_mlp_1hidden(self):
        tl = TranslationLayer(768, 1024, arch="mlp_1hidden", hidden_dim=512)
        x = _random_acts(4, 768)
        out = tl(x)
        assert out.shape == (4, 1024)

    def test_mlp_3hidden(self):
        tl = TranslationLayer(768, 1024, arch="mlp_3hidden", hidden_dim=1024)
        x = _random_acts(4, 768)
        out = tl(x)
        assert out.shape == (4, 1024)

    def test_single_vector_via_translate(self):
        """translate() accepts 1D input and returns 1D output."""
        tl = TranslationLayer(768, 1024, arch="mlp_1hidden")
        x = torch.randn(768)
        out = tl.translate(x)
        assert out.shape == (1024,)

    def test_batch_via_translate(self):
        """translate() accepts 2D batch and returns 2D."""
        tl = TranslationLayer(768, 1024, arch="mlp_1hidden")
        x = _random_acts(8, 768)
        out = tl.translate(x)
        assert out.shape == (8, 1024)

    def test_unknown_arch_raises(self):
        with pytest.raises(ValueError, match="Unknown arch"):
            TranslationLayer(768, 768, arch="transformer")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 2. Normalisation
# ---------------------------------------------------------------------------


class TestNormalisation:
    def test_normalise_static(self):
        x = torch.tensor([[3.0, 4.0]])
        normed = TranslationLayer.normalise(x)
        # ||[3,4]|| = 5, so normed = [0.6, 0.8]
        assert torch.allclose(normed, torch.tensor([[0.6, 0.8]]), atol=1e-5)

    def test_normalise_batch(self):
        x = _random_acts(16, 128)
        normed = TranslationLayer.normalise(x)
        norms = normed.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(16), atol=1e-5)

    def test_translate_normalises_input(self):
        """With normalize=True, translate() normalises x before forwarding."""
        tl = TranslationLayer(4, 4, arch="linear", normalize=True)
        # Use identity-like weights so we can inspect the transform
        nn.init.eye_(tl.net.proj.weight)  # type: ignore[union-attr]
        nn.init.zeros_(tl.net.proj.bias)  # type: ignore[union-attr]

        x = torch.tensor([[3.0, 4.0, 0.0, 0.0]])
        out = tl.translate(x)
        expected_normed = TranslationLayer.normalise(x)
        # forward on normed input with identity proj → output ≈ normed input
        assert torch.allclose(out, expected_normed, atol=1e-5)

    def test_translate_no_normalise(self):
        """With normalize=False, translate() passes x through unchanged."""
        tl = TranslationLayer(4, 4, arch="linear", normalize=False)
        nn.init.eye_(tl.net.proj.weight)  # type: ignore[union-attr]
        nn.init.zeros_(tl.net.proj.bias)  # type: ignore[union-attr]

        x = torch.tensor([[3.0, 4.0, 0.0, 0.0]])
        out = tl.translate(x)
        assert torch.allclose(out, x, atol=1e-5)


# ---------------------------------------------------------------------------
# 3. Training loop
# ---------------------------------------------------------------------------


class TestTrainingLoop:
    def test_loss_decreases_on_trivial_problem(self):
        """Training on a random linear mapping should reduce val loss."""
        dim = 32
        n = 200
        # Ground truth mapping: y = 2 * x (simple linear)
        acts_a = _random_acts(n, dim)
        acts_b = 2.0 * acts_a  # TranslationLayer should learn this

        tl = TranslationLayer(dim, dim, arch="linear", normalize=False)
        result = train_translation_layer(
            tl, acts_a, acts_b,
            epochs=20, batch_size=32, lr=1e-2, warmup_steps=5,
            val_fraction=0.1, device="cpu", seed=42,
        )
        assert result["val_losses"][0] > result["best_val_loss"], (
            "Val loss should improve during training"
        )

    def test_train_log_keys(self):
        """train_translation_layer returns expected keys."""
        dim = 16
        n = 50
        acts_a = _random_acts(n, dim)
        acts_b = _random_acts(n, dim)
        tl = TranslationLayer(dim, dim, arch="linear", normalize=False)
        result = train_translation_layer(
            tl, acts_a, acts_b,
            epochs=3, batch_size=16, lr=1e-3, warmup_steps=2,
            device="cpu", seed=0,
        )
        for key in ("train_losses", "val_losses", "best_epoch", "best_val_loss", "elapsed_sec"):
            assert key in result, f"Missing key: {key}"

    def test_train_loss_length_matches_epochs(self):
        dim = 8
        n = 40
        epochs = 5
        acts_a = _random_acts(n, dim)
        acts_b = _random_acts(n, dim)
        tl = TranslationLayer(dim, dim, arch="linear", normalize=False)
        result = train_translation_layer(
            tl, acts_a, acts_b,
            epochs=epochs, batch_size=16, lr=1e-3, warmup_steps=2,
            device="cpu",
        )
        assert len(result["train_losses"]) == epochs
        assert len(result["val_losses"]) == epochs


# ---------------------------------------------------------------------------
# 4. Checkpoint save / load
# ---------------------------------------------------------------------------


class TestCheckpoint:
    def test_save_and_load_round_trip(self, tmp_path):
        dim_src, dim_tgt = 64, 128
        tl = TranslationLayer(dim_src, dim_tgt, arch="mlp_1hidden", hidden_dim=32, normalize=True)

        acts_a = _random_acts(50, dim_src)
        acts_b = _random_acts(50, dim_tgt)
        ckpt_path = tmp_path / "test_ckpt.pt"

        train_translation_layer(
            tl, acts_a, acts_b,
            epochs=2, batch_size=16, lr=1e-3, warmup_steps=1,
            device="cpu", checkpoint_path=ckpt_path,
        )
        assert ckpt_path.exists(), "Checkpoint file not created"

        tl2 = load_translation_layer(ckpt_path, device="cpu")
        assert tl2.arch == "mlp_1hidden"
        assert tl2.dim_source == dim_src
        assert tl2.dim_target == dim_tgt
        assert tl2.normalize is True

        # Outputs should match
        x = _random_acts(4, dim_src)
        with torch.no_grad():
            out1 = tl(x)
            out2 = tl2(x)
        assert torch.allclose(out1, out2, atol=1e-6), "Loaded model outputs differ"


# ---------------------------------------------------------------------------
# 5. Hook injection
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
    # .to() and .eval() must return the *same* model object so that the nested
    # layer attributes we configure below are accessible after the chain.
    model.to.return_value = model
    model.eval.return_value = model

    # Set up the nested attribute path so layer resolution works.
    # layer_name = "gpt_neox.layers.8" → model.gpt_neox.layers.8 = mock_layer
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


class TestHookInjection:
    def test_inject_and_generate_calls_hook_registration(self):
        """inject_and_generate registers and removes a forward hook."""
        layer_name = "gpt_neox.layers.8"
        model, tokenizer, hook_handle, mock_layer = _make_mock_receiver(
            [1, 2, 3], [4, 5], "Paris", layer_name
        )
        vec = torch.randn(768)

        inject_and_generate(
            model, tokenizer, vec, layer_name,
            question="Where is France?",
            task="multi_hop",
            device="cpu",
        )

        mock_layer.register_forward_hook.assert_called_once()
        hook_handle.remove.assert_called_once()

    def test_inject_and_generate_output_stripped(self):
        """inject_and_generate returns decoded text with surrounding spaces stripped."""
        layer_name = "gpt_neox.layers.8"
        model, tokenizer, _, _ = _make_mock_receiver(
            [1, 2], [3, 4], "  France  ", layer_name
        )
        tokenizer.decode.return_value = "  France  "
        vec = torch.randn(768)

        result = inject_and_generate(
            model, tokenizer, vec, layer_name,
            question="Q?", task="multi_hop", device="cpu",
        )
        assert result == "France", f"Expected stripped output, got: {repr(result)}"

    def test_inject_and_generate_instruction_following_prompt(self):
        """instruction_following task uses 'Output:' anchor in receiver prompt."""
        layer_name = "gpt_neox.layers.8"
        model, tokenizer, _, _ = _make_mock_receiver(
            [1], [2, 3], "bonjour", layer_name
        )
        vec = torch.randn(768)

        inject_and_generate(
            model, tokenizer, vec, layer_name,
            question="Translate to French.",
            task="instruction_following",
            device="cpu",
        )

        # Verify the prompt passed to tokenizer contains "Output:"
        call_args = tokenizer.call_args
        prompt_text = call_args[0][0] if call_args[0] else call_args[1].get("text", "")
        assert "Output:" in prompt_text, f"'Output:' not in prompt: {prompt_text}"

    def test_inject_and_generate_qa_prompt(self):
        """multi_hop / knowledge_relay tasks use 'Q: ... A:' in receiver prompt."""
        layer_name = "gpt_neox.layers.8"
        model, tokenizer, _, _ = _make_mock_receiver(
            [1], [2], "Canada", layer_name
        )
        vec = torch.randn(768)

        inject_and_generate(
            model, tokenizer, vec, layer_name,
            question="What country?",
            task="knowledge_relay",
            device="cpu",
        )

        call_args = tokenizer.call_args
        prompt_text = call_args[0][0] if call_args[0] else call_args[1].get("text", "")
        assert "Q:" in prompt_text and "A:" in prompt_text, (
            f"Expected Q: ... A: in prompt, got: {prompt_text}"
        )
