"""
Activation extractor for arbitrary HuggingFace Transformer models.

Design rationale (see experiment guide §4.3.2):
- We extract *intermediate* layer activations, not the final logits.
- Final-layer representations are already formatted for next-token prediction;
  they carry less uncompressed semantic structure than middle layers.
- The extractor is model-agnostic: it works on any nn.Module with named layers.
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor


class ActivationExtractor:
    """Extracts hidden-state tensors from a named layer of any HuggingFace model.

    Uses PyTorch forward hooks so model weights never need to be modified.
    Hooks are registered on entry and cleanly removed on exit, making this
    safe to use in a `with` context.

    Args:
        model: Any HuggingFace causal-LM model (GPT-2, Pythia, etc.).
        layer_name: Dot-separated path to the target submodule, e.g.
            ``"transformer.h.6"`` for GPT-2 layer 6, or
            ``"gpt_neox.layers.6"`` for Pythia layer 6.

    Example:
        >>> extractor = ActivationExtractor(model, "transformer.h.6")
        >>> with extractor:
        ...     outputs = model(**inputs)
        ...     hidden = extractor.activation   # shape: (batch, seq_len, hidden_dim)
    """

    def __init__(self, model: torch.nn.Module, layer_name: str) -> None:
        self.model = model
        self.layer_name = layer_name
        self.activation: Tensor | None = None
        self._hook_handle = None

    # ------------------------------------------------------------------
    # Context manager interface
    # ------------------------------------------------------------------

    def __enter__(self) -> "ActivationExtractor":
        self._register_hook()
        return self

    def __exit__(self, *args) -> None:
        self._remove_hook()

    # ------------------------------------------------------------------
    # Manual interface (when context manager is inconvenient)
    # ------------------------------------------------------------------

    def register(self) -> "ActivationExtractor":
        """Register the hook. Call :meth:`remove` when done."""
        self._register_hook()
        return self

    def remove(self) -> None:
        """Remove the registered hook."""
        self._remove_hook()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_layer(self) -> torch.nn.Module:
        """Resolve dot-separated layer name to the actual submodule."""
        module = self.model
        for part in self.layer_name.split("."):
            try:
                module = getattr(module, part)
            except AttributeError:
                raise ValueError(
                    f"Layer '{self.layer_name}' not found in model. "
                    f"Failed at '{part}'. "
                    f"Available top-level attributes: {list(self.model._modules.keys())}"
                )
        return module

    def _make_hook(self) -> Callable:
        def hook(_module, _input, output):
            # output may be a tuple (hidden_states, ...) or a plain Tensor
            hidden = output[0] if isinstance(output, tuple) else output
            # Store detached copy so it survives backward passes
            self.activation = hidden.detach()

        return hook

    def _register_hook(self) -> None:
        if self._hook_handle is not None:
            return  # already registered
        layer = self._get_layer()
        self._hook_handle = layer.register_forward_hook(self._make_hook())

    def _remove_hook(self) -> None:
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None


# ---------------------------------------------------------------------------
# Utility: scan all layers and return their cosine similarity to a reference
# ---------------------------------------------------------------------------


@torch.no_grad()
def scan_layer_similarities(
    model_a: torch.nn.Module,
    model_b: torch.nn.Module,
    layer_names_a: list[str],
    layer_names_b: list[str],
    input_ids: Tensor,
    attention_mask: Tensor | None = None,
    device: str = "cpu",
) -> dict[tuple[str, str], float]:
    """Compare activations of paired layers between two models on identical input.

    Used during M0 to verify that GPT-2 and Pythia have *different* internal
    representations (expected mean cosine similarity << 1.0).

    Args:
        model_a: First model.
        model_b: Second model.
        layer_names_a: Layer names to extract from model_a.
        layer_names_b: Corresponding layer names from model_b (same length).
        input_ids: Token IDs (1, seq_len) — same input fed to both models.
        attention_mask: Optional attention mask.
        device: Torch device string.

    Returns:
        Dict mapping (layer_a_name, layer_b_name) → mean cosine similarity.
    """
    assert len(layer_names_a) == len(layer_names_b), (
        "layer_names_a and layer_names_b must have the same length"
    )

    input_ids = input_ids.to(device)
    attn = attention_mask.to(device) if attention_mask is not None else None

    extractors_a = [ActivationExtractor(model_a, n) for n in layer_names_a]
    extractors_b = [ActivationExtractor(model_b, n) for n in layer_names_b]

    # Register all hooks
    for e in extractors_a + extractors_b:
        e.register()

    # Forward passes
    model_a(input_ids=input_ids, attention_mask=attn)
    model_b(input_ids=input_ids, attention_mask=attn)

    # Remove hooks
    for e in extractors_a + extractors_b:
        e.remove()

    # Compute pairwise cosine similarities
    results: dict[tuple[str, str], float] = {}
    cos = torch.nn.CosineSimilarity(dim=-1)

    for ea, eb, na, nb in zip(extractors_a, extractors_b, layer_names_a, layer_names_b):
        act_a = ea.activation  # (1, seq, hidden_a)
        act_b = eb.activation  # (1, seq, hidden_b)

        if act_a is None or act_b is None:
            results[(na, nb)] = float("nan")
            continue

        # Pool over sequence dimension for a single similarity score
        a_pooled = act_a.mean(dim=1)  # (1, hidden_a)
        b_pooled = act_b.mean(dim=1)  # (1, hidden_b)

        # Pad shorter hidden dim with zeros to enable cosine comparison
        max_dim = max(a_pooled.shape[-1], b_pooled.shape[-1])
        if a_pooled.shape[-1] < max_dim:
            a_pooled = torch.nn.functional.pad(a_pooled, (0, max_dim - a_pooled.shape[-1]))
        if b_pooled.shape[-1] < max_dim:
            b_pooled = torch.nn.functional.pad(b_pooled, (0, max_dim - b_pooled.shape[-1]))

        similarity = cos(a_pooled, b_pooled).item()
        results[(na, nb)] = similarity

    return results
