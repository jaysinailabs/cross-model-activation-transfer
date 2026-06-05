"""
T5: Layer-wise CKA Analysis with Null Baseline.

Scientific rationale:
    CKA (Centered Kernel Alignment) is invariant to orthogonal transforms and
    isotropic scaling, making it more appropriate than cosine similarity for
    comparing representations across models with different hidden dimensions.

    A null baseline (random vectors of same shape) is computed alongside each
    CKA score to establish what "no structure" looks like, preventing
    over-interpretation of small positive CKA values.

Run:
    python -m rosetta.experiments.phase1.t5_cka_analysis

Reference: Kornblith et al. (2019) "Similarity of Neural Network Representations
    Revisited", ICML 2019.
"""

from __future__ import annotations

import os
from pathlib import Path

# Use project-local model cache (must precede transformers import)
_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = str(_PROJECT_ROOT / "models_cache")

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from rosetta.models.activation_extractor import ActivationExtractor  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RELATIVE_LAYERS = [0.25, 0.33, 0.5, 0.67, 0.83]  # extended scan vs T4

TIERS = {
    "tier1": {
        "sender": ("EleutherAI/pythia-160m", "gpt_neox.layers", 12),
        "receiver": ("EleutherAI/pythia-160m-deduped", "gpt_neox.layers", 12),
        "description": "Same arch + dim, different training data",
    },
    "tier2": {
        "sender": ("EleutherAI/pythia-160m", "gpt_neox.layers", 12),
        "receiver": ("EleutherAI/pythia-410m", "gpt_neox.layers", 24),
        "description": "Same arch, different size (768->1024)",
    },
}

# 20 diverse probe sentences covering factual, reasoning, negation, abstract domains
# (minimum for meaningful gram matrix; smoke test used only 3)
PROBE_SENTENCES = [
    # Factual / encyclopedic
    "The capital of France is Paris.",
    "Water boils at 100 degrees Celsius at standard pressure.",
    "The Earth orbits the Sun once every 365 days.",
    "Mount Everest is the highest mountain on Earth.",
    "Photosynthesis converts sunlight into chemical energy in plants.",
    # Reasoning / causal
    "If it rains, the ground becomes wet.",
    "Heating a gas at constant volume increases its pressure.",
    "The more you practice, the better you become.",
    "A heavier object falls at the same speed as a lighter one in a vacuum.",
    "Mixing blue and yellow paint produces green.",
    # Negation / contrast
    "The moon does not produce its own light.",
    "Humans cannot survive without oxygen for more than a few minutes.",
    "Antarctica is not a country; it is a continent.",
    "Gold does not rust when exposed to air or water.",
    "The sun does not orbit the Earth.",
    # Abstract / semantic
    "Justice requires treating similar cases similarly.",
    "Language is a system of symbols used for communication.",
    "Democracy is a form of government based on popular vote.",
    "Quantum mechanics describes physics at the atomic scale.",
    "The concept of infinity has no physical realization.",
]

N_RANDOM_TRIALS = 10  # null baseline: average over multiple random draws

# ---------------------------------------------------------------------------
# CKA implementation (linear kernel)
# ---------------------------------------------------------------------------


def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Compute linear CKA between representation matrices.

    Uses the feature-space formulation (Kornblith et al. 2019, Eq. 3):
        CKA(X, Y) = ||X^T Y||_F^2 / (||X^T X||_F * ||Y^T Y||_F)
    where X and Y are column-centered (mean over samples subtracted).

    This formulation handles different hidden dimensions (d_x != d_y)
    because X^T Y is (d_x, d_y) and both norms are scalars.

    Args:
        X: Activation matrix, shape (n_samples, dim_x). Will be centered.
        Y: Activation matrix, shape (n_samples, dim_y). Will be centered.

    Returns:
        CKA value in [0, 1]. Higher = more similar representational geometry.
    """
    # Upcast to float32: activations are float16, squaring overflows fp16 max (65504)
    X = X.float()
    Y = Y.float()
    # Center: subtract mean over samples for each feature
    X = X - X.mean(dim=0, keepdim=True)  # (n, d_x)
    Y = Y - Y.mean(dim=0, keepdim=True)  # (n, d_y)

    # Feature-space cross-covariance and auto-covariance matrices
    XtX = X.T @ X   # (d_x, d_x)
    YtY = Y.T @ Y   # (d_y, d_y)
    XtY = X.T @ Y   # (d_x, d_y)  — works for any d_x, d_y

    numerator = (XtY ** 2).sum()
    denominator = torch.sqrt((XtX ** 2).sum() * (YtY ** 2).sum())

    if denominator < 1e-10:
        return float("nan")
    return (numerator / denominator).item()


def null_baseline_cka(n: int, dim_x: int, dim_y: int, n_trials: int = N_RANDOM_TRIALS) -> float:
    """CKA between random Gaussian matrices (null hypothesis baseline).

    Uses gram-matrix CKA (sample-space) instead of feature-space CKA.
    When n << d (e.g. 20 sentences vs 768 hidden dim), the feature-space
    formula gives null≈1.0 regardless of correlation — a known degenerate
    regime. Gram-matrix CKA operates on (n×n) matrices and remains
    near 0 for uncorrelated random matrices regardless of hidden dim.

    Args:
        n: Number of samples (must match probe sentence count).
        dim_x: Hidden dimension of first model (unused in gram form).
        dim_y: Hidden dimension of second model (unused in gram form).
        n_trials: Number of random draws to average.

    Returns:
        Mean gram-CKA over random trials.
    """
    scores = []
    for _ in range(n_trials):
        X = torch.randn(n, dim_x)
        Y = torch.randn(n, dim_y)
        scores.append(gram_cka(X, Y))
    return sum(scores) / len(scores)


def gram_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Gram-matrix CKA: operates on (n×n) space, safe when n << d.

    CKA = <K_c, L_c>_F / (||K_c||_F * ||L_c||_F)
    where K = X X^T, L = Y Y^T, and K_c / L_c are doubly-centered gram matrices.
    """
    X = X.float()
    Y = Y.float()

    def center_gram(M: torch.Tensor) -> torch.Tensor:
        row = M.mean(dim=1, keepdim=True)
        col = M.mean(dim=0, keepdim=True)
        total = M.mean()
        return M - row - col + total

    K = center_gram(X @ X.T)
    L = center_gram(Y @ Y.T)
    num = (K * L).sum()
    denom = torch.sqrt((K ** 2).sum() * (L ** 2).sum())
    if denom < 1e-10:
        return float("nan")
    return (num / denom).item()


# ---------------------------------------------------------------------------
# Activation collection
# ---------------------------------------------------------------------------


@torch.no_grad()
def collect_activations(
    model: torch.nn.Module,
    tokenizer,
    layer_name: str,
    sentences: list[str],
    device: str,
) -> torch.Tensor:
    """Run model on each sentence, return mean-pooled activation matrix.

    Args:
        model: Loaded HF causal-LM.
        tokenizer: Corresponding tokenizer.
        layer_name: Target layer (e.g. "gpt_neox.layers.8").
        sentences: List of input sentences.
        device: Torch device.

    Returns:
        Tensor of shape (n_sentences, hidden_dim).
    """
    rows = []
    extractor = ActivationExtractor(model, layer_name)

    for sent in sentences:
        enc = tokenizer(sent, return_tensors="pt").to(device)
        with extractor, torch.no_grad():
            model(**enc)
        assert extractor.activation is not None
        # Mean-pool over sequence dim: (1, seq, hidden) -> (hidden,)
        rows.append(extractor.activation.mean(dim=1).squeeze(0).cpu())

    return torch.stack(rows)  # (n_sentences, hidden_dim)


# ---------------------------------------------------------------------------
# Per-tier analysis
# ---------------------------------------------------------------------------


def run_tier(tier_name: str, cfg: dict, device: str) -> None:
    """Run CKA analysis for one tier and print results."""
    s_id, s_prefix, s_n_layers = cfg["sender"]
    r_id, r_prefix, r_n_layers = cfg["receiver"]

    print(f"\n{'='*65}")
    print(f"  {tier_name.upper()}: {cfg['description']}")
    print(f"  Sender:   {s_id}  ({s_n_layers} layers, {_hidden_dim(s_id)} hidden)")
    print(f"  Receiver: {r_id}  ({r_n_layers} layers, {_hidden_dim(r_id)} hidden)")
    print(f"{'='*65}")

    tok_s = AutoTokenizer.from_pretrained(s_id)
    model_s = AutoModelForCausalLM.from_pretrained(s_id).to(device).eval()
    tok_s.pad_token = tok_s.eos_token

    tok_r = AutoTokenizer.from_pretrained(r_id)
    model_r = AutoModelForCausalLM.from_pretrained(r_id).to(device).eval()
    tok_r.pad_token = tok_r.eos_token

    dim_s = _hidden_dim(s_id)
    dim_r = _hidden_dim(r_id)
    n = len(PROBE_SENTENCES)

    print(f"\n  {'Depth':>6}  {'Layers':^18}  {'CKA':>8}  {'Null':>8}  {'Delta':>8}")
    print(f"  {'-'*6}  {'-'*18}  {'-'*8}  {'-'*8}  {'-'*8}")

    for rel in RELATIVE_LAYERS:
        s_idx = max(0, min(s_n_layers - 1, int(rel * s_n_layers)))
        r_idx = max(0, min(r_n_layers - 1, int(rel * r_n_layers)))
        s_layer = f"{s_prefix}.{s_idx}"
        r_layer = f"{r_prefix}.{r_idx}"

        act_s = collect_activations(model_s, tok_s, s_layer, PROBE_SENTENCES, device)
        act_r = collect_activations(model_r, tok_r, r_layer, PROBE_SENTENCES, device)

        cka_score = gram_cka(act_s, act_r)
        null_score = null_baseline_cka(n, dim_s, dim_r)
        delta = cka_score - null_score

        label = f"L{s_idx}/{s_n_layers} <-> L{r_idx}/{r_n_layers}"
        print(
            f"  {rel:>5.0%}  {label:^18}  {cka_score:>8.4f}  {null_score:>8.4f}  {delta:>+8.4f}"
        )

    del model_s, model_r
    if device != "cpu":
        torch.cuda.empty_cache()


def _hidden_dim(model_id: str) -> int:
    """Return known hidden dim for supported models."""
    known = {
        "EleutherAI/pythia-160m": 768,
        "EleutherAI/pythia-160m-deduped": 768,
        "EleutherAI/pythia-410m": 1024,
        "gpt2": 768,
    }
    return known.get(model_id, -1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 65)
    print("  T5: Layer-wise CKA Analysis with Null Baseline")
    print(f"  Probe sentences: {len(PROBE_SENTENCES)}")
    print(f"  Relative depths: {[f'{r:.0%}' for r in RELATIVE_LAYERS]}")
    print(f"  Null trials:     {N_RANDOM_TRIALS}")
    print("=" * 65)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    for tier_name, cfg in TIERS.items():
        run_tier(tier_name, cfg, device)

    print(f"\n{'='*65}")
    print("  T5 complete. Record CKA values in log.md and progress_log.md.")
    print("=" * 65)


if __name__ == "__main__":
    main()
