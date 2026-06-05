"""
T6: Procrustes Linear Alignment Baseline.

Scientific rationale:
    Orthogonal Procrustes finds the best-fit orthogonal mapping between two
    representation matrices of the same dimension, minimizing the Frobenius-norm
    residual ||A R - B||_F over all orthogonal matrices R.

    Alignment quality q = 1 - ||A R - B||_F / ||B||_F measures how well a
    *linear* orthogonal transform can align the two spaces.  This gives a
    theoretical upper bound for any translation layer that is constrained to
    orthogonal maps.  If even the optimal orthogonal alignment is poor, the
    true translation task is harder than a linear map can handle.

    A null baseline (random matrices, same shape) establishes the floor:
    what alignment quality do we get by chance?

    n=20 probe sentences are sufficient for Procrustes (it is an exact algebraic
    optimisation, not a statistical estimate like CKA).

Run:
    python -m rosetta.experiments.phase1.t6_procrustes

Reference: Schonemann (1966). "A generalized solution of the orthogonal
    Procrustes problem." Psychometrika 31(1): 1-10.
    scipy.linalg.orthogonal_procrustes documentation.
"""

from __future__ import annotations

import os
from pathlib import Path

# Use project-local model cache (must precede transformers import)
_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = str(_PROJECT_ROOT / "models_cache")

import torch  # noqa: E402
from scipy.linalg import orthogonal_procrustes  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from rosetta.models.activation_extractor import ActivationExtractor  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration (mirrors T5 probe set for comparability)
# ---------------------------------------------------------------------------

RELATIVE_LAYERS = [0.25, 0.33, 0.50, 0.67, 0.83]

TIERS = {
    "tier1": {
        "sender": ("EleutherAI/pythia-160m", "gpt_neox.layers", 12),
        "receiver": ("EleutherAI/pythia-160m-deduped", "gpt_neox.layers", 12),
        "description": "Same arch + dim, different training data",
        "dim_reduction": False,  # both 768 — no PCA needed
    },
    "tier2": {
        "sender": ("EleutherAI/pythia-160m", "gpt_neox.layers", 12),
        "receiver": ("EleutherAI/pythia-410m", "gpt_neox.layers", 24),
        "description": "Same arch, different size (768->1024)",
        "dim_reduction": True,  # 768 vs 1024 — PCA to min dim before aligning
    },
}

# Same 20 probe sentences as T5 (identical data -> directly comparable results)
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

N_NULL_TRIALS = 10  # average null baseline over multiple random draws

# ---------------------------------------------------------------------------
# Procrustes helpers
# ---------------------------------------------------------------------------


def pca_project(X: torch.Tensor, n_components: int) -> torch.Tensor:
    """Project X (n x d) onto its top-n_components principal components.

    Uses SVD of the mean-centered matrix.  Result shape: (n, n_components).
    Both X and the output are float32.

    Args:
        X: Activation matrix, shape (n, d).
        n_components: Number of principal components to retain.

    Returns:
        Projected matrix, shape (n, n_components).
    """
    X = X.float()
    X_c = X - X.mean(dim=0, keepdim=True)
    # economy SVD: U (n x k), S (k,), Vh (k x d)
    _, _, Vh = torch.linalg.svd(X_c, full_matrices=False)
    V_top = Vh[:n_components].T  # (d, n_components)
    return X_c @ V_top  # (n, n_components)


def procrustes_quality(A: torch.Tensor, B: torch.Tensor) -> float:
    """Align A onto B via orthogonal Procrustes and return alignment quality.

    Finds orthogonal R that minimises ||A R - B||_F, then returns:
        q = 1 - ||A R - B||_F / ||B||_F

    q = 1 means perfect alignment; q = 0 means residual equals ||B||_F;
    negative values possible if the unaligned residual is very large.

    Args:
        A: Source matrix (n, d) — float32.
        B: Target matrix (n, d) — float32. Must share shape with A.

    Returns:
        Alignment quality in (-inf, 1].
    """
    A_np = A.float().numpy()
    B_np = B.float().numpy()

    R, _ = orthogonal_procrustes(A_np, B_np)
    A_aligned = A_np @ R

    residual = float(((A_aligned - B_np) ** 2).sum() ** 0.5)
    b_norm = float((B_np ** 2).sum() ** 0.5)

    if b_norm < 1e-10:
        return float("nan")
    return 1.0 - residual / b_norm


def null_procrustes_quality(
    n: int, dim: int, n_trials: int = N_NULL_TRIALS
) -> float:
    """Mean Procrustes alignment quality for random Gaussian matrices.

    Args:
        n: Number of samples.
        dim: Shared hidden dimension (after any PCA projection).
        n_trials: Number of random draws to average.

    Returns:
        Mean alignment quality over random trials.
    """
    scores = []
    for _ in range(n_trials):
        A = torch.randn(n, dim)
        B = torch.randn(n, dim)
        scores.append(procrustes_quality(A, B))
    return sum(scores) / len(scores)


# ---------------------------------------------------------------------------
# Activation collection (identical to T5 for reproducibility)
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
        rows.append(extractor.activation.mean(dim=1).squeeze(0).cpu())

    return torch.stack(rows)  # (n_sentences, hidden_dim)


# ---------------------------------------------------------------------------
# Per-tier analysis
# ---------------------------------------------------------------------------


def run_tier(tier_name: str, cfg: dict, device: str) -> None:
    """Run Procrustes analysis for one tier and print results."""
    s_id, s_prefix, s_n_layers = cfg["sender"]
    r_id, r_prefix, r_n_layers = cfg["receiver"]
    needs_pca = cfg["dim_reduction"]

    print(f"\n{'='*65}")
    print(f"  {tier_name.upper()}: {cfg['description']}")
    print(f"  Sender:   {s_id}  ({s_n_layers} layers)")
    print(f"  Receiver: {r_id}  ({r_n_layers} layers)")
    if needs_pca:
        print("  Dim reduction: PCA to min(d_sender, d_receiver) before aligning")
    print(f"{'='*65}")

    tok_s = AutoTokenizer.from_pretrained(s_id)
    model_s = AutoModelForCausalLM.from_pretrained(s_id).to(device).eval()
    tok_s.pad_token = tok_s.eos_token

    tok_r = AutoTokenizer.from_pretrained(r_id)
    model_r = AutoModelForCausalLM.from_pretrained(r_id).to(device).eval()
    tok_r.pad_token = tok_r.eos_token

    n = len(PROBE_SENTENCES)

    print(f"\n  {'Depth':>6}  {'Layers':^22}  {'Quality':>8}  {'Null':>8}  {'Delta':>8}")
    print(f"  {'-'*6}  {'-'*22}  {'-'*8}  {'-'*8}  {'-'*8}")

    for rel in RELATIVE_LAYERS:
        s_idx = max(0, min(s_n_layers - 1, int(rel * s_n_layers)))
        r_idx = max(0, min(r_n_layers - 1, int(rel * r_n_layers)))
        s_layer = f"{s_prefix}.{s_idx}"
        r_layer = f"{r_prefix}.{r_idx}"

        act_s = collect_activations(model_s, tok_s, s_layer, PROBE_SENTENCES, device)
        act_r = collect_activations(model_r, tok_r, r_layer, PROBE_SENTENCES, device)

        # PCA projection for mismatched dims
        if needs_pca:
            target_dim = min(act_s.shape[1], act_r.shape[1])
            act_s_proc = pca_project(act_s, target_dim)
            act_r_proc = pca_project(act_r, target_dim)
        else:
            target_dim = act_s.shape[1]
            act_s_proc = act_s.float()
            act_r_proc = act_r.float()

        # Use actual projected dim (may be < target_dim when n < target_dim,
        # because economy SVD yields at most min(n, d) components)
        actual_dim = act_s_proc.shape[1]

        quality = procrustes_quality(act_s_proc, act_r_proc)
        null_q = null_procrustes_quality(n, actual_dim)
        delta = quality - null_q

        label = f"L{s_idx}/{s_n_layers} <-> L{r_idx}/{r_n_layers}"
        print(
            f"  {rel:>5.0%}  {label:^22}  {quality:>8.4f}  {null_q:>8.4f}  {delta:>+8.4f}"
        )

    del model_s, model_r
    if device != "cpu":
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 65)
    print("  T6: Procrustes Linear Alignment Baseline")
    print(f"  Probe sentences: {len(PROBE_SENTENCES)}")
    print(f"  Relative depths: {[f'{r:.0%}' for r in RELATIVE_LAYERS]}")
    print(f"  Null trials:     {N_NULL_TRIALS}")
    print("=" * 65)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    for tier_name, cfg in TIERS.items():
        run_tier(tier_name, cfg, device)

    print(f"\n{'='*65}")
    print("  T6 complete. Record alignment quality in log.md and progress_log.md.")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
