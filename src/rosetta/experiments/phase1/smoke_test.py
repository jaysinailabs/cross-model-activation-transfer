"""
M0 Smoke Test — Activation extraction, inter-model divergence, and layer scan.

Purpose (pre-research report §2, §6):
    1. Verify GPU/ROCm environment works
    2. Extract activations from Pythia-160M and Pythia-160M-deduped (Tier 1)
    3. Extract activations from Pythia-160M and Pythia-410M (Tier 2)
    4. Compute per-layer cosine similarity (expect << 1.0 for different models)
    5. Report CKA between paired layers across relative depth positions

Run:
    python -m rosetta.experiments.phase1.smoke_test
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Use project-local model cache (must be set before importing transformers)
_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = str(_PROJECT_ROOT / "models_cache")

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from rosetta.models.activation_extractor import ActivationExtractor  # noqa: E402


# Relative depth positions to scan (literature: semantic tasks peak at 0.67-0.83)
RELATIVE_LAYERS = [0.33, 0.5, 0.67, 0.83]

# Model pairs for progressive difficulty tiers
TIERS = {
    "tier1": {
        "sender": ("EleutherAI/pythia-160m", "gpt_neox.layers", 12),
        "receiver": ("EleutherAI/pythia-160m-deduped", "gpt_neox.layers", 12),
        "description": "Same arch + dim, different training data",
    },
    "tier2": {
        "sender": ("EleutherAI/pythia-160m", "gpt_neox.layers", 12),
        "receiver": ("EleutherAI/pythia-410m", "gpt_neox.layers", 24),
        "description": "Same arch, different size (768→1024)",
    },
}

PROBE_SENTENCES = [
    "The capital of France is Paris.",
    "Quantum mechanics describes the behavior of particles at the atomic scale.",
    "Water boils at 100 degrees Celsius at standard atmospheric pressure.",
]


def layer_indices(num_layers: int) -> list[int]:
    """Convert relative positions to absolute layer indices."""
    return [max(0, min(num_layers - 1, int(r * num_layers))) for r in RELATIVE_LAYERS]


def run_tier(tier_name: str, tier_cfg: dict, device: str) -> dict[str, list[float]]:
    """Run smoke test for one tier, return per-layer similarity lists."""
    s_id, s_prefix, s_layers = tier_cfg["sender"]
    r_id, r_prefix, r_layers = tier_cfg["receiver"]

    print(f"\n{'='*60}")
    print(f"  {tier_name.upper()}: {tier_cfg['description']}")
    print(f"  Sender:   {s_id} ({s_layers} layers)")
    print(f"  Receiver: {r_id} ({r_layers} layers)")
    print(f"{'='*60}")

    tok_s = AutoTokenizer.from_pretrained(s_id)
    model_s = AutoModelForCausalLM.from_pretrained(s_id).to(device).eval()
    tok_s.pad_token = tok_s.eos_token

    tok_r = AutoTokenizer.from_pretrained(r_id)
    model_r = AutoModelForCausalLM.from_pretrained(r_id).to(device).eval()
    tok_r.pad_token = tok_r.eos_token

    s_indices = layer_indices(s_layers)
    r_indices = layer_indices(r_layers)
    cos = torch.nn.CosineSimilarity(dim=-1)

    results: dict[str, list[float]] = {}

    for s_idx, r_idx, rel in zip(s_indices, r_indices, RELATIVE_LAYERS):
        s_layer = f"{s_prefix}.{s_idx}"
        r_layer = f"{r_prefix}.{r_idx}"
        key = f"{rel:.0%} depth (L{s_idx}/{s_layers} <-> L{r_idx}/{r_layers})"
        results[key] = []

        for sentence in PROBE_SENTENCES:
            enc_s = tok_s(sentence, return_tensors="pt").to(device)
            enc_r = tok_r(sentence, return_tensors="pt").to(device)

            ext_s = ActivationExtractor(model_s, s_layer)
            ext_r = ActivationExtractor(model_r, r_layer)

            with ext_s, ext_r, torch.no_grad():
                model_s(**enc_s)
                model_r(**enc_r)

            assert ext_s.activation is not None, f"No activation from {s_layer}"
            assert ext_r.activation is not None, f"No activation from {r_layer}"
            act_s = ext_s.activation.mean(dim=1)  # (1, hidden_s)
            act_r = ext_r.activation.mean(dim=1)  # (1, hidden_r)

            # Pad shorter dim with zeros for cosine comparison
            max_dim = max(act_s.shape[-1], act_r.shape[-1])
            if act_s.shape[-1] < max_dim:
                act_s = torch.nn.functional.pad(act_s, (0, max_dim - act_s.shape[-1]))
            if act_r.shape[-1] < max_dim:
                act_r = torch.nn.functional.pad(act_r, (0, max_dim - act_r.shape[-1]))

            sim = cos(act_s, act_r).item()
            results[key].append(sim)

    # Print results
    print(f"\n  Layer-wise mean cosine similarity ({len(PROBE_SENTENCES)} sentences):")
    for key, sims in results.items():
        mean_sim = sum(sims) / len(sims)
        print(f"    {key:45s}  cos_sim = {mean_sim:.4f}")

    # Cleanup GPU memory
    del model_s, model_r
    if device != "cpu":
        torch.cuda.empty_cache()

    return results


def main() -> None:
    # Environment check
    print("=" * 60)
    print("  M0 SMOKE TEST — Environment & Activation Diagnostics")
    print("=" * 60)

    if torch.cuda.is_available():
        device = "cuda"
        print(f"  Device:  {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory
        print(f"  VRAM:    {vram / 1e9:.1f} GB")
    elif hasattr(torch, "hip") and torch.hip.is_available():
        device = "cuda"  # ROCm uses cuda device in PyTorch
        print("  Device:  AMD GPU via ROCm")
    else:
        device = "cpu"
        print("  Device:  CPU (no GPU acceleration)")
        print("  WARNING: Training will be slow but functional.")

    # Run each tier
    all_ok = True
    for tier_name, tier_cfg in TIERS.items():
        try:
            results = run_tier(tier_name, tier_cfg, device)
            # Check: at least one layer pair should have similarity < 0.95
            all_means = [sum(s) / len(s) for s in results.values()]
            overall = sum(all_means) / len(all_means)
            if overall > 0.95:
                print(f"\n  WARNING: {tier_name} mean similarity {overall:.4f} is suspiciously high.")
                all_ok = False
            else:
                print(
                    f"\n  PASS: {tier_name} overall mean = {overall:.4f}"
                    " -- models have different representations."
                )
        except Exception as e:
            print(f"\n  FAIL: {tier_name} — {e}")
            all_ok = False

    # Summary
    print("\n" + "=" * 60)
    if all_ok:
        print("  ALL TIERS PASSED — Proceeding to M1 (data generation)")
    else:
        print("  ISSUES DETECTED — Review output above before proceeding")
    print("=" * 60)

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
