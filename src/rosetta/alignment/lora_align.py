"""LoRA-based shared-basis alignment for cross-model activation transfer.

Goal (H2): Align the intermediate representations of Pythia-160M (sender) and
Pythia-410M (receiver) via LoRA fine-tuning so that the translation layer can
generalise across domains without retraining.

Design rationale:
- Both models receive LoRA adapters (rank=8) applied to attention QKV + output
  projections.  Only LoRA parameters are trained; original weights are frozen.
- Alignment loss: MSE( T(h_a(x)), h_b(x) ) where T is the *frozen* translation
  layer checkpoint from M4.  Freezing T isolates the LoRA contribution for a
  clean H2 test.
- Capability-retention proxy: cross-entropy language-modelling loss of model_b.
  Computing inference-based accuracy during training would be prohibitively slow
  on a single DirectML GPU.
- Training texts: context strings from test_enhanced.jsonl (multi-domain),
  supplemented by Wikitext if needed.

Usage::

    aligner = LoraAligner.from_config(
        model_a=model_a,
        model_b=model_b,
        translation_ckpt="results/phase1/checkpoints/m4_P1b_tier2_mlp_1hidden_n5000_s42.pt",
        device="privateuseone:0",
    )
    result = aligner.train(texts, layer_name_a="gpt_neox.layers.8",
                                   layer_name_b="gpt_neox.layers.16")
    aligner.save("results/phase1/checkpoints", seed=42)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DEFAULT_TARGET_MODULES = ["query_key_value", "dense"]  # GPT-NeoX / Pythia attention
_LAYER_A = "gpt_neox.layers.8"   # 67% of 12 layers
_LAYER_B = "gpt_neox.layers.16"  # 67% of 24 layers


def _get_peft_lora_config(rank: int = 8, alpha: int = 16,
                          target_modules: list[str] | None = None):
    """Build a PEFT LoraConfig for GPT-NeoX / Pythia models."""
    try:
        from peft import LoraConfig, TaskType
    except ImportError as e:
        raise ImportError("peft>=0.9.0 is required for LoRA alignment. "
                          "Install with: pip install peft>=0.9.0") from e

    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=alpha,
        target_modules=target_modules or _DEFAULT_TARGET_MODULES,
        lora_dropout=0.0,         # no dropout — alignment is deterministic
        bias="none",
        inference_mode=False,
    )


def _apply_lora(model: nn.Module, lora_config) -> nn.Module:
    """Wrap a model with PEFT LoRA adapters (in-place replacement)."""
    from peft import get_peft_model
    return get_peft_model(model, lora_config)


# ---------------------------------------------------------------------------
# Activation extraction hook
# ---------------------------------------------------------------------------

class _ActivationHook:
    """Single-shot hook that captures the *output* of a named module.

    Attaches to the target layer, captures on first forward pass, then
    detaches automatically.
    """

    def __init__(self) -> None:
        self._handle = None
        self.activation: Tensor | None = None

    def attach(self, module: nn.Module) -> "_ActivationHook":
        self._handle = module.register_forward_hook(self._hook)
        return self

    def _hook(self, _module, _input, output):
        # GPT-NeoX layers return tuples; take the hidden-state tensor.
        self.activation = output[0] if isinstance(output, tuple) else output
        self.detach()

    def detach(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


def _get_submodule(model: nn.Module, name: str) -> nn.Module:
    """Return nested submodule by dotted path (e.g. 'gpt_neox.layers.8')."""
    for part in name.split("."):
        model = getattr(model, part)
    return model


# ---------------------------------------------------------------------------
# Text dataset
# ---------------------------------------------------------------------------

class _TokenisedDataset(Dataset):
    """Pre-tokenised fixed-length chunks for alignment training."""

    def __init__(self, tokens: Tensor, chunk_size: int = 128) -> None:
        # tokens: 1-D long tensor of all token ids
        n_chunks = tokens.shape[0] // chunk_size
        self._data = tokens[: n_chunks * chunk_size].view(n_chunks, chunk_size)

    def __len__(self) -> int:
        return self._data.shape[0]

    def __getitem__(self, idx: int) -> Tensor:
        return self._data[idx]


def _load_alignment_texts(
    enhanced_jsonl: str | Path | None,
    corpus_dir: str | Path | None,
    n_texts: int = 1000,
) -> list[str]:
    """Collect alignment training texts from available sources.

    Priority:
    1. test_enhanced.jsonl context strings (multi-domain, avoids geo-overfitting)
    2. data/corpus/wikitext*.txt files (general coverage)
    """
    texts: list[str] = []

    # Source 1: multi-domain contexts from test_enhanced.jsonl
    if enhanced_jsonl and Path(enhanced_jsonl).exists():
        with open(enhanced_jsonl, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    ctx = obj.get("context", "")
                    if ctx:
                        texts.append(ctx)
                except json.JSONDecodeError:
                    continue
        logger.info("Loaded %d context strings from test_enhanced.jsonl", len(texts))

    # Source 2: wikitext corpus files for general coverage
    if len(texts) < n_texts and corpus_dir:
        corpus_path = Path(corpus_dir)
        for txt_file in sorted(corpus_path.glob("*.txt"))[:5]:
            with open(txt_file, encoding="utf-8", errors="replace") as f:
                raw = f.read()
            # Split into ~200-char chunks
            sentences = [s.strip() for s in raw.split("\n") if len(s.strip()) > 50]
            texts.extend(sentences)
            if len(texts) >= n_texts * 2:
                break
        logger.info("Total texts after corpus supplement: %d", len(texts))

    if not texts:
        logger.warning("No alignment texts found; using placeholder sentences.")
        texts = ["The quick brown fox jumps over the lazy dog."] * max(n_texts, 32)

    # Deduplicate and truncate
    seen: set[str] = set()
    unique: list[str] = []
    for t in texts:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return unique[:n_texts]


# ---------------------------------------------------------------------------
# Main aligner
# ---------------------------------------------------------------------------

@dataclass
class AlignmentResult:
    """Training result container."""

    loss_history: list[float] = field(default_factory=list)
    converged: bool = False
    epochs_run: int = 0
    checkpoint_a: str = ""
    checkpoint_b: str = ""


class LoraAligner:
    """LoRA alignment trainer for cross-model shared-basis learning.

    Trains LoRA adapters on both models jointly to minimise the MSE between
    T(h_a(x)) and h_b(x), where T is the frozen M4 translation layer.

    Args:
        model_a:        Sender model (Pythia-160M) with LoRA adapters attached.
        model_b:        Receiver model (Pythia-410M) with LoRA adapters attached.
        tokenizer_a:    Tokenizer for model_a.
        tokenizer_b:    Tokenizer for model_b (usually same as a for Pythia).
        translation_layer: Frozen M4 translation network (768→1024 MLP).
        layer_name_a:   Dotted module path to extract activations from model_a.
        layer_name_b:   Dotted module path to inject alignment target in model_b.
        lambda_align:   Weight of alignment loss relative to LM loss.
        device:         PyTorch device string.
    """

    def __init__(
        self,
        model_a: nn.Module,
        model_b: nn.Module,
        tokenizer_a,
        tokenizer_b,
        translation_layer: nn.Module,
        layer_name_a: str = _LAYER_A,
        layer_name_b: str = _LAYER_B,
        lambda_align: float = 0.01,
        device: str = "cpu",
    ) -> None:
        self.model_a = model_a
        self.model_b = model_b
        self.tok_a = tokenizer_a
        self.tok_b = tokenizer_b
        self.translation_layer = translation_layer
        self.layer_name_a = layer_name_a
        self.layer_name_b = layer_name_b
        self.lambda_align = lambda_align
        self.device = device

        # Freeze translation layer
        for p in self.translation_layer.parameters():
            p.requires_grad_(False)
        self.translation_layer.eval()
        self.translation_layer.to(device)

    @classmethod
    def from_config(
        cls,
        model_a: nn.Module,
        model_b: nn.Module,
        tokenizer_a,
        tokenizer_b,
        translation_ckpt: str | Path,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        target_modules: list[str] | None = None,
        lambda_align: float = 0.01,
        layer_name_a: str = _LAYER_A,
        layer_name_b: str = _LAYER_B,
        device: str = "cpu",
    ) -> "LoraAligner":
        """Build a LoraAligner from configs and a translation-layer checkpoint.

        Applies LoRA adapters to both models in-place.
        """
        from rosetta.translation.translation_layer import load_translation_layer

        lora_cfg = _get_peft_lora_config(
            rank=lora_rank, alpha=lora_alpha, target_modules=target_modules
        )

        logger.info("Applying LoRA (rank=%d, alpha=%d) to model_a ...", lora_rank, lora_alpha)
        model_a_lora = _apply_lora(model_a, lora_cfg)
        logger.info("Applying LoRA (rank=%d, alpha=%d) to model_b ...", lora_rank, lora_alpha)
        model_b_lora = _apply_lora(model_b, lora_cfg)

        logger.info("Loading (frozen) translation layer from %s", translation_ckpt)
        tl = load_translation_layer(Path(translation_ckpt), device=device)

        return cls(
            model_a=model_a_lora,
            model_b=model_b_lora,
            tokenizer_a=tokenizer_a,
            tokenizer_b=tokenizer_b,
            translation_layer=tl,
            layer_name_a=layer_name_a,
            layer_name_b=layer_name_b,
            lambda_align=lambda_align,
            device=device,
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        texts: list[str],
        epochs: int = 3,
        learning_rate: float = 2e-5,
        batch_size: int = 4,
        chunk_size: int = 128,
        convergence_tol: float = 0.05,
    ) -> AlignmentResult:
        """Run LoRA alignment training.

        For each batch of text chunks:
          1. Forward both models, capture layer activations via hooks.
          2. Compute L_align = MSE(T(h_a), h_b), averaged over sequence positions.
          3. Compute L_LM = cross-entropy of model_b on the same tokens.
          4. Backprop on L_total = L_align + lambda_align * L_LM.

        The translation layer T is frozen throughout; only LoRA params are updated.

        Args:
            texts:           List of plain-text strings for alignment.
            epochs:          Number of passes over the dataset.
            learning_rate:   AdamW learning rate.
            batch_size:      Number of chunks per gradient step.
            chunk_size:      Token length per chunk (shorter = faster).
            convergence_tol: |loss[-1] / loss[-2] - 1| < tol means converged.

        Returns:
            AlignmentResult with loss history and convergence flag.
        """
        result = AlignmentResult()

        # Tokenise all texts into a flat token stream (model_a tokenizer)
        logger.info("Tokenising %d texts (chunk_size=%d) ...", len(texts), chunk_size)
        all_ids: list[int] = []
        for text in texts:
            enc = self.tok_a(text, add_special_tokens=False)
            all_ids.extend(enc["input_ids"])

        token_tensor = torch.tensor(all_ids, dtype=torch.long)
        dataset = _TokenisedDataset(token_tensor, chunk_size=chunk_size)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        logger.info("Dataset: %d chunks × %d tokens = %d tokens total",
                    len(dataset), chunk_size, len(dataset) * chunk_size)

        # Collect LoRA parameters only
        lora_params_a = [p for n, p in self.model_a.named_parameters() if "lora_" in n]
        lora_params_b = [p for n, p in self.model_b.named_parameters() if "lora_" in n]
        all_lora_params = lora_params_a + lora_params_b
        n_lora = sum(p.numel() for p in all_lora_params)
        logger.info("Training %d LoRA parameters (%.1f K)", n_lora, n_lora / 1000)

        optimizer = torch.optim.AdamW(all_lora_params, lr=learning_rate, weight_decay=0.01)

        self.model_a.to(self.device)
        self.model_b.to(self.device)

        mse = nn.MSELoss()
        epoch_losses: list[float] = []
        lm_loss: Tensor = torch.tensor(0.0)

        for epoch in range(1, epochs + 1):
            self.model_a.train()
            self.model_b.train()
            batch_losses: list[float] = []
            t0 = time.time()

            for batch_tokens in loader:
                batch_tokens = batch_tokens.to(self.device)

                # ── Capture activations from both models ──────────────────
                hook_a = _ActivationHook()
                hook_b = _ActivationHook()
                hook_a.attach(_get_submodule(self.model_a, self.layer_name_a))
                hook_b.attach(_get_submodule(self.model_b, self.layer_name_b))

                # Forward pass model_a — gradients flow through LoRA_a params
                # via h_a → h_a_translated → align_loss.  No special context
                # needed; the default PyTorch context is gradient-enabled.
                _ = self.model_a(batch_tokens)

                # Forward pass model_b (need LM loss + activations)
                labels = batch_tokens.clone()
                out_b = self.model_b(batch_tokens, labels=labels)
                lm_loss = out_b.loss  # scalar; type already declared above loop

                # ── Compute alignment loss ────────────────────────────────
                h_a = hook_a.activation   # (batch, seq, 768)
                h_b = hook_b.activation   # (batch, seq, 1024)

                if h_a is None or h_b is None:
                    logger.warning("Activation hook did not fire — skipping batch.")
                    optimizer.zero_grad()
                    continue

                # Pool over sequence dimension → (batch*seq, dim).
                # Cast to float32: DirectML forward passes may produce float16
                # activations, while the translation layer weights are float32.
                tl_dtype = next(self.translation_layer.parameters()).dtype
                h_a_flat = h_a.reshape(-1, h_a.shape[-1]).to(tl_dtype)   # (batch*seq, 768)
                h_b_flat = h_b.reshape(-1, h_b.shape[-1]).to(tl_dtype)   # (batch*seq, 1024)

                h_a_translated = self.translation_layer(h_a_flat)  # (batch*seq, 1024)
                align_loss = mse(h_a_translated, h_b_flat.detach())

                loss = align_loss + self.lambda_align * lm_loss

                # ── Backprop ──────────────────────────────────────────────
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(all_lora_params, max_norm=1.0)
                optimizer.step()

                batch_losses.append(align_loss.item())

            epoch_loss = sum(batch_losses) / max(len(batch_losses), 1)
            epoch_losses.append(epoch_loss)
            elapsed = time.time() - t0
            logger.info("  Epoch %d/%d | align_loss=%.6f | lm_loss=%.4f | %.1fs",
                        epoch, epochs, epoch_loss, lm_loss.item(), elapsed)

        result.loss_history = epoch_losses
        result.epochs_run = epochs

        # Convergence check: last epoch loss within 5% of second-to-last
        if len(epoch_losses) >= 2:
            ratio = epoch_losses[-1] / (epoch_losses[-2] + 1e-9)
            result.converged = abs(ratio - 1.0) < convergence_tol
            converged_str = "CONVERGED" if result.converged else "NOT CONVERGED"
            logger.info("Convergence check: ratio=%.4f (tol=%.2f) → %s",
                        ratio, convergence_tol, converged_str)

        return result

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save(self, checkpoint_dir: str | Path, seed: int = 42,
             lora_rank: int = 8, epochs: int = 0) -> tuple[str, str]:
        """Save LoRA adapter weights for both models.

        Args:
            epochs: Number of training epochs run. Included in the filename
                    so each (rank, epochs, seed) combination maps to a unique
                    file — prevents silent overwriting across runs and satisfies
                    reproducibility requirements for open data.

        Returns:
            (path_a, path_b) — saved file paths.
        """
        ckpt_dir = Path(checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        epoch_tag = f"_e{epochs}" if epochs > 0 else ""
        tag = f"m4b_lora_tier2_r{lora_rank}{epoch_tag}_s{seed}"
        path_a = str(ckpt_dir / f"{tag}_a.pt")
        path_b = str(ckpt_dir / f"{tag}_b.pt")

        # Save only the LoRA delta weights (not full model)
        lora_state_a = {k: v for k, v in self.model_a.state_dict().items()
                        if "lora_" in k}
        lora_state_b = {k: v for k, v in self.model_b.state_dict().items()
                        if "lora_" in k}
        torch.save(lora_state_a, path_a)
        torch.save(lora_state_b, path_b)
        logger.info("Saved LoRA weights: %s / %s", path_a, path_b)
        return path_a, path_b

    def load(self, path_a: str | Path, path_b: str | Path,
             strict: bool = False) -> None:
        """Load LoRA delta weights into already-wrapped models.

        Args:
            strict: If False, missing/extra keys are silently ignored.
        """
        state_a = torch.load(str(path_a), map_location=self.device, weights_only=True)
        state_b = torch.load(str(path_b), map_location=self.device, weights_only=True)
        self.model_a.load_state_dict(state_a, strict=strict)
        self.model_b.load_state_dict(state_b, strict=strict)
        logger.info("Loaded LoRA weights from %s / %s", path_a, path_b)


# ---------------------------------------------------------------------------
# Convenience top-level function (for scripts)
# ---------------------------------------------------------------------------


def train_lora_alignment(
    model_a: nn.Module,
    model_b: nn.Module,
    tokenizer_a,
    tokenizer_b,
    translation_ckpt: str | Path,
    enhanced_jsonl: str | Path | None = None,
    corpus_dir: str | Path = "data/corpus",
    n_texts: int = 1000,
    lora_rank: int = 8,
    lora_alpha: int = 16,
    lambda_align: float = 0.01,
    epochs: int = 3,
    learning_rate: float = 2e-5,
    batch_size: int = 4,
    chunk_size: int = 128,
    layer_name_a: str = _LAYER_A,
    layer_name_b: str = _LAYER_B,
    device: str = "cpu",
    checkpoint_dir: str | Path = "results/phase1/checkpoints",
    seed: int = 42,
) -> AlignmentResult:
    """End-to-end LoRA alignment training.

    Loads texts (corpus-only by default — pass enhanced_jsonl=None to avoid
    contaminating the evaluation set), builds aligner, trains, saves checkpoints,
    and returns the result.

    Example::

        result = train_lora_alignment(
            model_a=model_a, model_b=model_b,
            tokenizer_a=tok, tokenizer_b=tok,
            translation_ckpt="results/phase1/checkpoints/m4_P1b_tier2_mlp_1hidden_n5000_s42.pt",
            device="privateuseone:0",
            seed=42,
        )
        print(result.loss_history)
    """
    import random
    import numpy as np

    # Reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    texts = _load_alignment_texts(enhanced_jsonl, corpus_dir, n_texts=n_texts)
    logger.info("Alignment training texts: %d", len(texts))

    aligner = LoraAligner.from_config(
        model_a=model_a,
        model_b=model_b,
        tokenizer_a=tokenizer_a,
        tokenizer_b=tokenizer_b,
        translation_ckpt=translation_ckpt,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lambda_align=lambda_align,
        layer_name_a=layer_name_a,
        layer_name_b=layer_name_b,
        device=device,
    )

    result = aligner.train(
        texts=texts,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        chunk_size=chunk_size,
    )

    path_a, path_b = aligner.save(checkpoint_dir, seed=seed, lora_rank=lora_rank,
                                   epochs=epochs)
    result.checkpoint_a = path_a
    result.checkpoint_b = path_b

    return result
