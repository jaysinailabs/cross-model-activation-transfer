"""
Translation layer networks for cross-model activation transfer.

Design rationale (experiment guide §4.3.2):
- We train a lightweight network to map model-A's intermediate activations into
  model-B's representation space.
- Three architectures (linear → MLP-1 → MLP-3) follow a simple-first strategy:
  escalate complexity only if simpler architectures are insufficient.
- Training objective: MSE on L2-normalised activations (mitigates scale
  differences between models of different sizes).
- Injection modes:
  - "replace": full replacement of model-B's hidden states at layer L with
    translated activations (guide §4.3.2 original specification).
  - "additive": residual injection; translated vector acts as a perturbation.
  - "replace_interpolate": linear interpolation between original and translated.
  - "replace_scale_corrected": like "replace" but L2-normalises the translated
    vector and rescales to match model-B's per-position hidden-state norm,
    eliminating the scale mismatch (M7 diagnostic).
- Supports both 2D (batch, dim) and 3D (batch, seq, dim) activations.
  Sequence-level (3D) is required for replacement injection.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Network architectures
# ---------------------------------------------------------------------------


class _LinearTranslation(nn.Module):
    """Single affine map: dim_source → dim_target."""

    def __init__(self, dim_source: int, dim_target: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim_source, dim_target, bias=True)

    def forward(self, x: Tensor) -> Tensor:  # noqa: D102
        return self.proj(x)


class _MLPTranslation(nn.Module):
    """MLP with configurable depth: dim_source → [hidden]×n_hidden → dim_target."""

    def __init__(
        self,
        dim_source: int,
        dim_target: int,
        hidden_dim: int,
        num_hidden: int,
        activation: Literal["gelu", "relu"] = "gelu",
    ) -> None:
        super().__init__()
        act_fn = nn.GELU() if activation == "gelu" else nn.ReLU()
        layers: list[nn.Module] = [nn.Linear(dim_source, hidden_dim), act_fn]
        for _ in range(num_hidden - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), act_fn]
        layers.append(nn.Linear(hidden_dim, dim_target))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:  # noqa: D102
        return self.net(x)


# ---------------------------------------------------------------------------
# Public TranslationLayer wrapper
# ---------------------------------------------------------------------------


class TranslationLayer(nn.Module):
    """Wraps a translation network with optional L2-normalisation of activations.

    This is the single entry-point for the M3 pipeline.  Callers should use
    :meth:`from_config` to build from ``configs/phase1.yaml`` parameters, or
    the constructor directly for custom experiments.

    Args:
        dim_source: Hidden dimension of sender model at extraction layer.
        dim_target: Hidden dimension of receiver model at injection layer.
        arch: Architecture variant — ``"linear"``, ``"mlp_1hidden"``,
            or ``"mlp_3hidden"``.
        hidden_dim: MLP hidden width (ignored for ``"linear"``).
        normalize: If True, L2-normalise *both* inputs and targets before
            computing MSE loss during training.  The normalisation is applied
            inside :meth:`normalise` rather than inside ``forward``, so
            inference vectors can be un-normalised and then the caller
            normalises separately if needed.  See :meth:`translate` which
            handles this automatically.

    Example:
        >>> tl = TranslationLayer(768, 1024, arch="mlp_1hidden")
        >>> out = tl(torch.randn(8, 768))   # (8, 1024)
    """

    def __init__(
        self,
        dim_source: int,
        dim_target: int,
        arch: Literal["linear", "mlp_1hidden", "mlp_3hidden"] = "mlp_1hidden",
        hidden_dim: int = 512,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.dim_source = dim_source
        self.dim_target = dim_target
        self.arch = arch
        self.hidden_dim = hidden_dim
        self.normalize = normalize

        if arch == "linear":
            self.net = _LinearTranslation(dim_source, dim_target)
        elif arch == "mlp_1hidden":
            self.net = _MLPTranslation(dim_source, dim_target, hidden_dim, num_hidden=1)
        elif arch == "mlp_3hidden":
            self.net = _MLPTranslation(dim_source, dim_target, hidden_dim, num_hidden=3)
        else:
            raise ValueError(f"Unknown arch '{arch}'. Choose: linear, mlp_1hidden, mlp_3hidden")

    # ------------------------------------------------------------------
    # Forward / translate
    # ------------------------------------------------------------------

    def forward(self, x: Tensor) -> Tensor:
        """Raw forward pass — no normalisation applied.

        Args:
            x: Source activations, shape ``(batch, dim_source)`` or
                ``(batch, seq, dim_source)`` for sequence-level translation.

        Returns:
            Translated activations with same leading dimensions as input.
        """
        if x.dim() == 3:
            batch, seq, dim = x.shape
            out_2d = self.net(x.reshape(batch * seq, dim))
            return out_2d.reshape(batch, seq, -1)
        return self.net(x)

    @staticmethod
    def normalise(x: Tensor) -> Tensor:
        """L2-normalise along the last dimension (unit-sphere projection)."""
        return nn.functional.normalize(x, p=2, dim=-1)

    def translate(self, x: Tensor) -> Tensor:
        """Forward pass with optional L2-normalisation (mirrors training behaviour).

        If ``self.normalize`` is True, normalises *x* before passing through
        the network.  The output is not re-normalised (the network should learn
        to produce unit-norm outputs when needed).

        Args:
            x: Source activations, shape ``(batch, dim_source)``,
                ``(dim_source,)`` for a single vector, or
                ``(batch, seq, dim_source)`` for sequence-level translation.

        Returns:
            Translated activations matching the input rank.
        """
        assert x.dim() in (1, 2, 3), f"Expected 1D/2D/3D input, got {x.dim()}D"
        squeezed = x.dim() == 1
        if squeezed:
            x = x.unsqueeze(0)
        x = x.float()  # ensure float32 (DML may produce float16)
        if self.normalize:
            x = self.normalise(x)
        out = self.forward(x)
        return out.squeeze(0) if squeezed else out

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        dim_source: int,
        dim_target: int,
        cfg: dict,
    ) -> "TranslationLayer":
        """Build from ``configs/phase1.yaml`` ``translation_layer`` block.

        Args:
            dim_source: Sender model hidden dim.
            dim_target: Receiver model hidden dim.
            cfg: The ``translation_layer`` sub-dict from the YAML config.

        Returns:
            Constructed :class:`TranslationLayer` instance.
        """
        arch = cfg.get("default_architecture", "mlp_1hidden")
        arch_cfg = cfg.get("architectures", {}).get(arch, {})
        hidden_dim = arch_cfg.get("hidden_dim", 512)
        normalize = cfg.get("normalize_activations", True)
        return cls(dim_source, dim_target, arch=arch, hidden_dim=hidden_dim, normalize=normalize)


# ---------------------------------------------------------------------------
# Activation pair extraction
# ---------------------------------------------------------------------------


@torch.no_grad()
def extract_activation_pairs(
    model_a: nn.Module,
    tokenizer_a,
    model_b: nn.Module,
    tokenizer_b,
    texts: list[str],
    layer_name_a: str,
    layer_name_b: str,
    device: str = "cpu",
    batch_size: int = 8,
    max_length: int = 256,
    pooling_mode: Literal["mean", "last_token", "sequence"] = "mean",
) -> tuple[Tensor, Tensor]:
    """Extract hidden states from paired layers of two models.

    Both models perform a forward pass on each text independently.  For
    ``"mean"`` and ``"last_token"`` modes, activations are pooled over the
    sequence dimension producing a single vector per sample.  For
    ``"sequence"`` mode, all valid (non-padding) positions are retained and
    flattened across samples into position-level pairs — useful for training
    sequence-level translation layers.

    Args:
        model_a: Sender model.
        tokenizer_a: Sender tokenizer.
        model_b: Receiver model.
        tokenizer_b: Receiver tokenizer.
        texts: List of plain text strings (same text fed to both models).
        layer_name_a: Dot-separated layer path in model_a (e.g. ``"gpt_neox.layers.8"``).
        layer_name_b: Dot-separated layer path in model_b (e.g. ``"gpt_neox.layers.16"``).
        device: Torch device string.
        batch_size: Number of texts processed per forward pass.
        max_length: Tokeniser max sequence length.
        pooling_mode: How to reduce the sequence dimension.
            ``"mean"`` (default): weighted mean over valid tokens.
            ``"last_token"``: the last non-padding token.
            ``"sequence"``: no pooling — return all valid positions flattened
            to ``(total_positions, hidden_dim)``.

    Returns:
        Tuple ``(acts_a, acts_b)``.  Shape is ``(n_texts, hidden_dim)`` for
        pooled modes, or ``(total_positions, hidden_dim)`` for sequence mode.
    """
    from rosetta.models.activation_extractor import ActivationExtractor

    model_a = model_a.to(device).eval()
    model_b = model_b.to(device).eval()

    acts_a_list: list[Tensor] = []
    acts_b_list: list[Tensor] = []

    def _pool(act: Tensor, mask: Tensor) -> Tensor:
        """Pool (batch, seq, hidden) → (batch, hidden) per pooling_mode."""
        if pooling_mode == "last_token":
            # mask: (batch, seq) — last valid position per sample
            last_pos = mask.sum(dim=1).long() - 1  # (batch,)
            return act[torch.arange(act.size(0), device=act.device), last_pos, :]
        # "mean": weighted mean over valid tokens
        mask_f = mask.unsqueeze(-1).float()
        return (act * mask_f).sum(1) / mask_f.sum(1)

    n = len(texts)
    for start in range(0, n, batch_size):
        batch_texts = texts[start : start + batch_size]
        if (start // batch_size) % 20 == 0:
            logger.info("Extracting activations: %d/%d", start, n)

        # ---- Model A ----
        enc_a = tokenizer_a(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)
        with ActivationExtractor(model_a, layer_name_a) as ext_a:
            model_a(**enc_a)
        act_a = ext_a.activation  # (batch, seq, hidden)

        # ---- Model B ----
        enc_b = tokenizer_b(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)
        with ActivationExtractor(model_b, layer_name_b) as ext_b:
            model_b(**enc_b)
        act_b = ext_b.activation  # (batch, seq, hidden)

        if pooling_mode == "sequence":
            # Keep per-position activations (skip pooling)
            mask_a = enc_a["attention_mask"]  # (batch, seq)
            mask_b = enc_b["attention_mask"]
            for i in range(act_a.size(0)):
                valid_len_a = int(mask_a[i].sum().item())
                valid_len_b = int(mask_b[i].sum().item())
                assert valid_len_a == valid_len_b, (
                    f"Seq length mismatch at sample {start + i}: "
                    f"A={valid_len_a}, B={valid_len_b}. "
                    "Same tokenizer required for sequence mode."
                )
                acts_a_list.append(act_a[i, :valid_len_a, :].cpu())
                acts_b_list.append(act_b[i, :valid_len_b, :].cpu())
        else:
            acts_a_list.append(_pool(act_a, enc_a["attention_mask"]).cpu())
            acts_b_list.append(_pool(act_b, enc_b["attention_mask"]).cpu())

    acts_a = torch.cat(acts_a_list, dim=0)
    acts_b = torch.cat(acts_b_list, dim=0)
    if pooling_mode == "sequence":
        logger.info(
            "Sequence-level extraction: %d total positions from %d samples",
            acts_a.shape[0], n,
        )
    return acts_a, acts_b


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train_translation_layer(
    tl: TranslationLayer,
    acts_a: Tensor,
    acts_b: Tensor,
    *,
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-4,
    warmup_steps: int = 100,
    val_fraction: float = 0.1,
    device: str = "cpu",
    checkpoint_path: str | Path | None = None,
    seed: int = 42,
) -> dict:
    """Train *tl* to minimise MSE between translated acts_a and acts_b.

    Training / validation split is performed by taking the last
    ``val_fraction`` of the data as validation (no shuffle of the split,
    but training data is shuffled each epoch).

    Args:
        tl: TranslationLayer to train (modified in place).
        acts_a: Source activations, shape ``(n, dim_source)``.
        acts_b: Target activations, shape ``(n, dim_target)``.
        epochs: Number of training epochs.
        batch_size: Training mini-batch size.
        lr: Peak learning rate (AdamW).
        warmup_steps: Linear warmup steps for the LR scheduler.
        val_fraction: Fraction of data held out for validation.
        device: Torch device string.
        checkpoint_path: If given, save best-val-loss checkpoint here.
        seed: Random seed for DataLoader shuffling.

    Returns:
        Dict with keys ``train_losses``, ``val_losses``, ``best_epoch``,
        ``best_val_loss``, ``elapsed_sec``.
    """
    torch.manual_seed(seed)
    tl = tl.to(device)

    # Cast activations to float32 for training stability
    # (DirectML may produce float16 activations)
    acts_a = acts_a.float()
    acts_b = acts_b.float()

    # --- Split ---
    n = len(acts_a)
    n_val = max(1, int(n * val_fraction))
    n_train = n - n_val
    acts_a_train, acts_a_val = acts_a[:n_train], acts_a[n_train:]
    acts_b_train, acts_b_val = acts_b[:n_train], acts_b[n_train:]

    # Optionally pre-normalise targets (mirrors translate() behaviour)
    if tl.normalize:
        acts_a_train = TranslationLayer.normalise(acts_a_train)
        acts_a_val = TranslationLayer.normalise(acts_a_val)
        acts_b_train = TranslationLayer.normalise(acts_b_train)
        acts_b_val = TranslationLayer.normalise(acts_b_val)

    train_ds = TensorDataset(acts_a_train.to(device), acts_b_train.to(device))
    val_ds = TensorDataset(acts_a_val.to(device), acts_b_val.to(device))
    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(tl.parameters(), lr=lr)
    total_steps = epochs * len(train_loader)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return max(0.0, 1.0 - (step - warmup_steps) / max(1, total_steps - warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    loss_fn = nn.MSELoss()

    train_losses: list[float] = []
    val_losses: list[float] = []
    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    t0 = time.time()
    step = 0

    for epoch in range(1, epochs + 1):
        # --- Train ---
        tl.train()
        epoch_loss = 0.0
        for src, tgt in train_loader:
            optimizer.zero_grad()
            pred = tl(src)
            loss = loss_fn(pred, tgt)
            loss.backward()
            nn.utils.clip_grad_norm_(tl.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item() * len(src)
            step += 1
        train_losses.append(epoch_loss / n_train)

        # --- Validate ---
        tl.eval()
        with torch.no_grad():
            val_loss = (
                sum(loss_fn(tl(src), tgt).item() * len(src) for src, tgt in val_loader) / n_val
            )
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in tl.state_dict().items()}

        if epoch % 5 == 0 or epoch == 1:
            logger.info(
                "Epoch %d/%d  train=%.5f  val=%.5f  best_val=%.5f (ep %d)",
                epoch,
                epochs,
                train_losses[-1],
                val_loss,
                best_val_loss,
                best_epoch,
            )

    # Restore best weights
    if best_state is not None:
        tl.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    # Save checkpoint
    if checkpoint_path is not None:
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": tl.state_dict(),
                "arch": tl.arch,
                "dim_source": tl.dim_source,
                "dim_target": tl.dim_target,
                "hidden_dim": tl.hidden_dim,
                "normalize": tl.normalize,
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "train_losses": train_losses,
                "val_losses": val_losses,
            },
            checkpoint_path,
        )
        logger.info("Checkpoint saved → %s", checkpoint_path)

    elapsed = time.time() - t0
    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "elapsed_sec": elapsed,
    }


def load_translation_layer(checkpoint_path: str | Path, device: str = "cpu") -> TranslationLayer:
    """Load a :class:`TranslationLayer` from a saved checkpoint.

    Args:
        checkpoint_path: Path to ``.pt`` file saved by :func:`train_translation_layer`.
        device: Torch device to load onto.

    Returns:
        Loaded and eval-mode :class:`TranslationLayer`.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    tl = TranslationLayer(
        dim_source=ckpt["dim_source"],
        dim_target=ckpt["dim_target"],
        arch=ckpt["arch"],
        hidden_dim=ckpt.get("hidden_dim", 512),
        normalize=ckpt["normalize"],
    )
    tl.load_state_dict(ckpt["state_dict"])
    tl.eval()
    return tl


# ---------------------------------------------------------------------------
# Injection helper
# ---------------------------------------------------------------------------


@torch.no_grad()
def inject_and_generate(
    model_b: nn.Module,
    tokenizer_b,
    translated_activations: Tensor,
    layer_name_b: str,
    question: str,
    task: str,
    device: str = "cpu",
    injection_scale: float = 1.0,
    injection_timing: Literal["persistent", "prefill_only"] = "persistent",
    injection_mode: Literal["additive", "replace_interpolate", "replace", "replace_scale_corrected"] = "additive",
    injection_alpha: float = 1.0,
    nl_context: str | None = None,
    context: str | None = None,
    full_input: bool = False,
    **gen_kwargs,
) -> str:
    """Generate model-B output with translated activation injected at layer L.

    Args:
        model_b: Receiver model (on ``device``).
        tokenizer_b: Receiver tokenizer.
        translated_activations: Translated activation tensor.
            For additive/replace_interpolate: ``(dim,)`` or ``(1, dim)``,
            broadcast to all token positions.
            For replace: ``(1, seq, dim)`` — full sequence-level replacement.
        layer_name_b: Dot-separated path to the injection layer in model_b.
        question: The question/instruction text fed to model_b.
        task: Task type — ``"multi_hop"``, ``"knowledge_relay"``, or
            ``"instruction_following"``.
        device: Torch device string.
        injection_scale: Multiplier for the injected vector (default 1.0).
            Ignored in ``"replace"`` mode (always 100% signal).
        injection_timing: When to fire the injection hook.
            ``"persistent"``: fires at every forward pass.
            ``"prefill_only"``: fires only during prompt prefill.
            Replace mode requires ``"prefill_only"``.
        injection_mode: How to combine translated activations with hidden state.
            ``"additive"`` (default): ``hidden += scale * vec``.
            ``"replace_interpolate"``: ``hidden = (1-scale)*hidden + scale*vec``.
            ``"replace"``: ``hidden = translated`` (guide §4.3.2 spec).
            ``"replace_scale_corrected"``: like ``"replace"`` but L2-normalises
            the translated vector and rescales to B's per-position hidden-state
            norm, eliminating scale mismatch (M7 diagnostic).
        injection_alpha: Blending coefficient for ``"replace_scale_corrected"``
            mode only.  ``1.0`` (default) = full replacement with the
            scale-corrected vector.  ``0.0`` = no injection (identity).
            Intermediate values linearly mix: ``α×corrected + (1-α)×original``.
        nl_context: Optional NL relay text to prepend to the receiver prompt.
        context: Context text for full-input mode (e.g. passage for multi_hop).
        full_input: If True, build prompt with context + question so that
            model B receives the same input as model A.
        **gen_kwargs: Passed to ``model_b.generate()``.

    Returns:
        Decoded generated text (new tokens only, stripped).
    """
    assert injection_mode in ("additive", "replace_interpolate", "replace", "replace_scale_corrected"), \
        f"Unknown injection_mode: {injection_mode}"
    if injection_mode in ("replace", "replace_scale_corrected"):
        assert injection_timing == "prefill_only", \
            "Replace mode requires prefill_only timing (auto-regressive steps have seq=1)"
    model_b = model_b.to(device).eval()

    # Build receiver prompt
    if nl_context is not None:
        # E2b combo condition: NL relay text + question + activation injection
        if task in ("multi_hop", "knowledge_relay"):
            prompt = f"{nl_context}\nQ: {question}\nA:"
        else:
            prompt = f"{nl_context}\nInstruction: {question}\nOutput:"
    elif full_input and context is not None:
        # M6 full-input mode: B receives same context+question as A
        if task in ("multi_hop", "knowledge_relay"):
            prompt = f"Context: {context}\nQ: {question}\nA:"
        else:
            prompt = f"Context: {context}\nInstruction: {question}\nOutput:"
    elif task in ("multi_hop", "knowledge_relay"):
        # Standard: question only — no relay text
        prompt = f"Q: {question}\nA:"
    else:
        prompt = f"Instruction: {question}\nOutput:"

    inputs = tokenizer_b(prompt, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]

    # Prepare injection activations
    activations = translated_activations.to(device)
    if injection_mode in ("replace", "replace_scale_corrected"):
        # Replace modes: need 3D (batch, seq, dim) activations
        if activations.dim() == 2:
            activations = activations.unsqueeze(0)  # (seq, dim) → (1, seq, dim)
        assert activations.dim() == 3, \
            f"Replace mode requires 3D (batch, seq, dim), got {activations.dim()}D"
    else:
        # Additive / replace_interpolate: (1, dim) broadcast to all positions
        if activations.dim() == 1:
            activations = activations.unsqueeze(0)
        if activations.dim() == 3:
            # If 3D given for non-replace mode, pool to last token
            activations = activations[:, -1, :]
        activations = activations * injection_scale  # scale applied once here

    # prefill_only mode: use a flag to inject only on the first (prefill) forward
    inject_done = [False]

    def _injection_hook(_module, _input, output):
        hidden = output[0] if isinstance(output, tuple) else output

        if injection_timing == "prefill_only":
            # hidden.shape[1] == 1 means a single-token auto-regressive step
            if inject_done[0] or hidden.shape[1] == 1:
                return output
            inject_done[0] = True

        if injection_mode == "replace":
            # Replacement injection — guide §4.3.2 L369 original specification
            assert activations.shape[1] == hidden.shape[1], (
                f"Seq length mismatch: translated={activations.shape[1]} "
                f"vs hidden={hidden.shape[1]}. Both models must receive "
                f"identical tokenized input for replace mode."
            )
            new_hidden = activations.to(dtype=hidden.dtype, device=hidden.device)
        elif injection_mode == "replace_scale_corrected":
            # Scale-corrected replacement — M7 diagnostic.
            # L2-normalises the translated direction, then rescales to B's natural
            # per-position norm.  This eliminates the 85× scale mismatch (translate()
            # output norm≈0.85 vs B's hidden norm≈72.4) while preserving direction.
            # target_norm must be read BEFORE hidden is overwritten.
            assert activations.shape[1] == hidden.shape[1], (
                f"Seq length mismatch: translated={activations.shape[1]} "
                f"vs hidden={hidden.shape[1]}. Both models must receive "
                f"identical tokenized input for replace mode."
            )
            target_norm = hidden.norm(dim=-1, keepdim=True)  # (batch, seq, 1)
            translated = activations.to(dtype=hidden.dtype, device=hidden.device)
            corrected = nn.functional.normalize(translated, p=2, dim=-1) * target_norm
            # injection_alpha=1.0 → full replacement; <1.0 → linear blend (Phase B)
            new_hidden = injection_alpha * corrected + (1.0 - injection_alpha) * hidden
        elif injection_mode == "replace_interpolate":
            # Linear interpolation: at scale=1 fully replaces with translated vec
            v = activations.unsqueeze(1).to(dtype=hidden.dtype, device=hidden.device)
            new_hidden = (1.0 - injection_scale) * hidden + v
        else:
            # additive (default, M3 original)
            v = activations.unsqueeze(1).to(dtype=hidden.dtype, device=hidden.device)
            new_hidden = hidden + v

        if isinstance(output, tuple):
            return (new_hidden,) + output[1:]
        return new_hidden

    # Resolve layer module
    module = model_b
    for part in layer_name_b.split("."):
        module = getattr(module, part)
    hook_handle = module.register_forward_hook(_injection_hook)

    try:
        default_kwargs = {
            "max_new_tokens": 64,
            "do_sample": False,
            "repetition_penalty": 1.3,
            "no_repeat_ngram_size": 3,
            "pad_token_id": tokenizer_b.eos_token_id,
        }
        default_kwargs.update(gen_kwargs)
        output_ids = model_b.generate(**inputs, **default_kwargs)
    finally:
        if hook_handle is not None:
            hook_handle.remove()

    new_ids = output_ids[0, prompt_len:]
    return tokenizer_b.decode(new_ids, skip_special_tokens=True).strip()
