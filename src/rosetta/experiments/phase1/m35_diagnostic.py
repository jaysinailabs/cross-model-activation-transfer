"""M3.5: Diagnostic experiments — why does the translation layer ≈ null relay?

Implements five sequential diagnostic experiments (P0-P4) to identify root
causes of M3's pipeline failures, and determines the optimal injection
configuration for M4.

Experiment sequence (see plans/phase1/Project_Rosetta_M35_诊断实验指引_v3.md):
    P0-logits : Hook correctness — replace-mode self-stitch on CPU (prefill only)
    P0-generate: Injection timing comparison (persistent vs prefill-only,
                 additive vs replace) on full 160-sample multi_hop test set
    P1        : Extraction ablation (last-token vs mean-pool, with/without L2)
    P2        : Injection scale sweep on best P1 config
    P3        : Linear vs MLP-1 architecture comparison (merged with P1)

Results saved to:
    results/phase1/m35_{phase}_{config_tag}_{tier}_{task}_{seed}.json

Usage:
    # Run full diagnostic sequence:
    python -m rosetta.experiments.phase1.m35_diagnostic --device dml

    # Only P0 (hook verification):
    python -m rosetta.experiments.phase1.m35_diagnostic --phase p0 --device cpu

    # P1 ablation only:
    python -m rosetta.experiments.phase1.m35_diagnostic --phase p1 --device dml
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from rosetta.experiments.phase1.m2_baseline import (
    evaluate_instruction_following,
    evaluate_knowledge_relay,
    evaluate_multi_hop,
    get_device,
    load_model_and_tokenizer,
    load_test_split,  # noqa: F401 — re-exported for callers; used in main()
)
from rosetta.experiments.phase1.m3_translation import (
    _CHECKPOINTS_DIR,
    _DEFAULT_RELATIVE_LAYER,
    _RESULTS_DIR,
    _TIERS,
    load_train_texts,
    resolve_layer_name,
)
from rosetta.translation.translation_layer import (
    TranslationLayer,
    extract_activation_pairs,
    inject_and_generate,
    load_translation_layer,
    train_translation_layer,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_TIER = "tier1"  # default tier; overridden by --tier
_TASK = "multi_hop"  # clearest signal; null=0.144 (DirectML T0), M2=0.225
_SEED = 42
_N_TRAIN = 5000  # wikitext corpus size for P1 training
_LAYER_REL = _DEFAULT_RELATIVE_LAYER  # 0.67
_PREFIX = "m35"  # result file prefix; overridden by --prefix for M4
_NULL_BASELINE = 0.144  # null relay acc; overridden by --null-baseline for T2


# ---------------------------------------------------------------------------
# Shared evaluation helper
# ---------------------------------------------------------------------------


def _get_acc(metrics: dict) -> float:
    """Extract accuracy from metrics dict (contains_acc or compliance_rate)."""
    return metrics.get("contains_acc", metrics.get("compliance_rate", 0.0))


def _get_sender_text_and_question(sample: dict, task: str) -> tuple[str, str]:
    """Build sender input text and receiver question from sample dict."""
    if task == "instruction_following":
        sender_text = f"Instruction: {sample['instruction']}\nInput: {sample['input_text']}"
        question = sample["input_text"]
    elif task == "knowledge_relay":
        sender_text = sample["passage"]
        question = sample["question"]
    else:  # multi_hop
        sender_text = sample["context"]
        question = sample["question"]
    return sender_text, question


def _evaluate_samples(
    samples: list[dict],
    task: str,
    predicted: list[str],
) -> dict:
    """Compute task metric for a list of predictions.

    Returns a metrics dict and a list of per-sample eval records.
    """
    raw: list[dict] = []
    for sample, pred in zip(samples, predicted):
        if task == "multi_hop":
            result = evaluate_multi_hop(pred, sample["answer"])
        elif task == "knowledge_relay":
            result = evaluate_knowledge_relay(pred, sample["answer"])
        else:
            result = evaluate_instruction_following(pred, sample["constraints"])
        raw.append({"source": sample, "predicted": pred, "eval_result": result})

    if task == "instruction_following":
        n_ok = sum(1 for r in raw if r["eval_result"].get("compliant", False))
        metrics = {"compliance_rate": n_ok / len(raw), "n": len(raw)}
    else:
        n_ok = sum(1 for r in raw if r["eval_result"].get("correct", False))
        metrics = {
            "contains_acc": n_ok / len(raw),
            "exact_match_acc": sum(1 for r in raw if r["eval_result"].get("exact_match", False))
            / len(raw),
            "n": len(raw),
        }
    return {"metrics": metrics, "raw": raw}


def _build_per_sample(raw: list[dict], task: str) -> list[dict]:
    """Convert raw evaluation records to compact per-sample list for result files.

    Each entry: {"idx": int, "correct": bool, "generated": str, "gold": str}

    Args:
        raw: List of {"source": sample, "predicted": str, "eval_result": dict}
             as returned by _evaluate_samples.
        task: Task name ("multi_hop", "knowledge_relay", "instruction_following").

    Returns:
        List of compact per-sample dicts.
    """
    out = []
    for i, r in enumerate(raw):
        sample = r["source"]
        if task == "instruction_following":
            correct = bool(r["eval_result"].get("compliant", False))
            gold = str(sample.get("constraints", ""))
        else:
            correct = bool(r["eval_result"].get("correct", False))
            gold = str(sample.get("answer", ""))
        out.append(
            {
                "idx": i,
                "correct": correct,
                "generated": r["predicted"],
                "gold": gold,
            }
        )
    return out


def _save_result(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info("Saved: %s", path.name)


# ---------------------------------------------------------------------------
# P0-logits: hook correctness on CPU
# ---------------------------------------------------------------------------


def run_p0_logits(
    model: nn.Module,
    tokenizer,
    samples: list[dict],
    layer_idx: int,
    device: str,
    n_check: int = 20,
) -> bool:
    """Verify replace-mode self-stitch produces identical prefill logits.

    Args:
        model: Receiver model (should be on CPU for deterministic FP32).
        tokenizer: Receiver tokenizer.
        samples: Test samples to use as inputs.
        layer_idx: Layer index to hook (e.g. 8 for 0.67 × 12).
        device: Must be 'cpu' for reliable numerical comparison.
        n_check: Number of samples to check (default 20).

    Returns:
        True if all samples pass (max L1 diff < 1e-5), False otherwise.
    """
    logger.info("P0-logits: verifying replace self-stitch on %d samples (CPU)", n_check)
    assert device == "cpu", "P0-logits must run on CPU for deterministic FP32 comparison"

    model = model.to(device).eval()
    results = []

    for i, sample in enumerate(samples[:n_check]):
        prompt = f"Q: {sample['question']}\nA:"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # 1. Baseline logits
        with torch.no_grad():
            logits_base = model(**inputs).logits  # (1, seq, vocab)

        # 2. Capture layer activation
        captured: dict[str, Any] = {}

        def _capture(module, inp, output, _c=captured):
            h = output[0] if isinstance(output, tuple) else output
            _c["h"] = h.detach().clone()
            return output

        handle = model.gpt_neox.layers[layer_idx].register_forward_hook(_capture)
        with torch.no_grad():
            model(**inputs)
        handle.remove()

        # 3. Replace-inject: substitute layer output with itself
        def _replace(module, inp, output, _c=captured):
            h = output[0] if isinstance(output, tuple) else output
            new_h = _c["h"].to(dtype=h.dtype, device=h.device)
            return (new_h,) + output[1:] if isinstance(output, tuple) else new_h

        handle = model.gpt_neox.layers[layer_idx].register_forward_hook(_replace)
        with torch.no_grad():
            logits_inject = model(**inputs).logits
        handle.remove()

        # 4. Compare
        diff = (logits_base - logits_inject).abs().max().item()
        passed = diff < 1e-5
        results.append({"sample_idx": i, "max_l1_diff": diff, "pass": passed})
        if not passed:
            logger.warning("  Sample %d FAIL: max L1 diff = %.3e", i, diff)

    n_pass = sum(r["pass"] for r in results)
    max_diff = max(r["max_l1_diff"] for r in results)
    passed_all = n_pass == len(results)

    status = "PASS" if passed_all else "FAIL"
    logger.info(
        "P0-logits %s: %d/%d passed, max diff = %.3e",
        status,
        n_pass,
        len(results),
        max_diff,
    )

    out = {
        "experiment": "P0_logits",
        "device": device,
        "layer_idx": layer_idx,
        "n_checked": len(results),
        "n_pass": n_pass,
        "max_l1_diff": max_diff,
        "passed": passed_all,
        "per_sample": results,
    }
    _save_result(_RESULTS_DIR / f"{_PREFIX}_P0_logits_cpu.json", out)
    return passed_all


# ---------------------------------------------------------------------------
# P0-generate: injection timing comparison
# ---------------------------------------------------------------------------


def _run_injection_condition(
    model: nn.Module,
    tokenizer,
    samples: list[dict],
    layer_idx: int,
    device: str,
    injection_timing: str,
    injection_mode: str,
    captured_acts: dict[int, torch.Tensor],
) -> list[str]:
    """Run generate() for one injection condition on all samples.

    ``captured_acts[i]`` is the layer activation captured from a separate
    forward pass on sample i (used for self-stitch: inject model's own acts).

    injection_timing: 'persistent' | 'prefill_only'
    injection_mode  : 'additive' | 'replace'
    """
    model = model.to(device).eval()
    predictions = []

    for i, sample in enumerate(samples):
        prompt = f"Q: {sample['question']}\nA:"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        prompt_len = inputs["input_ids"].shape[1]

        captured_h = captured_acts[i].to(device)  # (1, seq, hidden)
        inject_done = [False]

        def _hook(module, inp, output, _h=captured_h, _done=inject_done):
            h = output[0] if isinstance(output, tuple) else output
            if injection_timing == "prefill_only":
                if _done[0] or h.shape[1] == 1:
                    return output
                _done[0] = True
            if injection_mode == "additive":
                # Self-stitch additive: hidden += hidden  → 2× hidden
                new_h = h + _h.to(dtype=h.dtype, device=h.device)[:, : h.shape[1], :]
            else:
                # Self-stitch replace: hidden = hidden (identity)
                new_h = _h.to(dtype=h.dtype, device=h.device)[:, : h.shape[1], :]
            return (new_h,) + output[1:] if isinstance(output, tuple) else new_h

        handle = model.gpt_neox.layers[layer_idx].register_forward_hook(_hook)
        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                repetition_penalty=1.3,
                no_repeat_ngram_size=3,
                pad_token_id=tokenizer.eos_token_id,
            )
        handle.remove()

        new_ids = out_ids[0, prompt_len:]
        predictions.append(tokenizer.decode(new_ids, skip_special_tokens=True).strip())

        if (i + 1) % 40 == 0:
            logger.info("  P0-generate: %d/%d", i + 1, len(samples))

    return predictions


def run_p0_generate(
    model: nn.Module,
    tokenizer,
    samples: list[dict],
    layer_idx: int,
    device: str,
) -> dict:
    """Run P0-generate: compare 4 injection conditions for self-stitch.

    Conditions:
      - baseline         : no injection
      - replace_prefill  : replace with own acts, prefill-only
      - additive_persistent: add own acts at every generate step (M3 original)
      - additive_prefill : add own acts at prefill only
    """
    logger.info("P0-generate: 4 conditions × %d samples", len(samples))
    model = model.to(device).eval()

    # Step 1: capture each sample's layer activation (prefill pass)
    logger.info("  Capturing self-activations for all samples...")
    captured_acts: dict[int, torch.Tensor] = {}
    for i, sample in enumerate(samples):
        prompt = f"Q: {sample['question']}\nA:"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        cap: dict[str, Any] = {}

        def _cap(module, inp, output, _c=cap):
            h = output[0] if isinstance(output, tuple) else output
            _c["h"] = h.detach().cpu().clone()
            return output

        handle = model.gpt_neox.layers[layer_idx].register_forward_hook(_cap)
        with torch.no_grad():
            model(**inputs)
        handle.remove()
        captured_acts[i] = cap["h"]

    # Step 2: baseline (no injection)
    logger.info("  Condition: baseline (no injection)...")
    baseline_preds = []
    for sample in samples:
        prompt = f"Q: {sample['question']}\nA:"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        prompt_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                repetition_penalty=1.3,
                no_repeat_ngram_size=3,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_ids = out_ids[0, prompt_len:]
        baseline_preds.append(tokenizer.decode(new_ids, skip_special_tokens=True).strip())

    conditions = {
        "baseline": baseline_preds,
    }
    for timing, mode, tag in [
        ("prefill_only", "replace", "replace_prefill"),
        ("persistent", "additive", "additive_persistent"),
        ("prefill_only", "additive", "additive_prefill"),
    ]:
        logger.info("  Condition: %s...", tag)
        preds = _run_injection_condition(
            model,
            tokenizer,
            samples,
            layer_idx,
            device,
            timing,
            mode,
            captured_acts,
        )
        conditions[tag] = preds

    # Evaluate all conditions
    results = {}
    for tag, preds in conditions.items():
        evaled = _evaluate_samples(samples, _TASK, preds)
        acc = _get_acc(evaled["metrics"])
        logger.info("  P0 condition %-25s acc = %.4f", tag, acc)
        results[tag] = {"metrics": evaled["metrics"], "n": len(samples)}
        # Save individual condition file
        out = {
            "experiment": "P0_generate",
            "config": {
                "condition": tag,
                "layer_idx": layer_idx,
                "tier": _TIER,
                "task": _TASK,
                "seed": _SEED,
            },
            "metric": evaled["metrics"],
            "n_samples": len(samples),
            "per_sample": _build_per_sample(evaled["raw"], _TASK),
        }
        fname = f"{_PREFIX}_P0_{tag}_{_TIER}_{_TASK}_{_SEED}.json"
        _save_result(_RESULTS_DIR / fname, out)

    # Summary
    summary = {
        "experiment": "P0_generate_summary",
        "layer_idx": layer_idx,
        "conditions": {k: v["metrics"] for k, v in results.items()},
    }
    _save_result(_RESULTS_DIR / f"{_PREFIX}_P0_summary_{_TIER}_{_TASK}.json", summary)
    return results


# ---------------------------------------------------------------------------
# P1: Extraction ablation (pooling_mode × normalize)
# ---------------------------------------------------------------------------


def run_p1_ablation(
    model_a: nn.Module,
    tokenizer_a,
    model_b: nn.Module,
    tokenizer_b,
    samples: list[dict],
    layer_name_a: str,
    layer_name_b: str,
    device: str,
    corpus_path: Path,
    injection_timing: str = "prefill_only",
    injection_mode: str = "additive",
    injection_scale: float = 0.1,
    full_input: bool = False,
    translation_ckpt_override: str | Path | None = None,
) -> dict[str, dict]:
    """Train and evaluate P1a / P1b / P1c translation layer variants.

    Variants (additive/replace_interpolate modes):
        P1a: last_token pooling + L2 normalization + MLP-1
        P1b: last_token pooling + no normalization  + MLP-1
        P1c: last_token pooling + no normalization  + Linear

    For ``injection_mode="replace"``, a single evaluation path is used with
    pre-trained checkpoint (M6 replace mode).

    Args:
        injection_timing: From P0 result ('persistent' or 'prefill_only').
        injection_mode: 'additive', 'replace_interpolate', or 'replace'.
        injection_scale: Scale to use for P1 evaluation (before P2 sweep).
        full_input: If True, model B receives context+question (same as A).
        translation_ckpt_override: Path to pre-trained checkpoint (required
            for replace mode).

    Returns:
        Dict mapping variant tag → {metrics, config}.
    """
    dim_a = _TIERS[_TIER]["sender_hidden_dim"]
    dim_b = _TIERS[_TIER]["receiver_hidden_dim"]
    mlp_hidden = max(dim_a, dim_b)  # C3: avoid bottleneck for T2 (768→1024)

    # ── M6 replace mode: single-path evaluation with pre-trained checkpoint ──
    if injection_mode == "replace":
        assert translation_ckpt_override is not None, (
            "Replace mode requires --translation-ckpt "
            "(M6 checkpoint trained in Phase C)"
        )
        logger.info("=== M6 replace mode: loading checkpoint %s ===", translation_ckpt_override)
        tl = load_translation_layer(str(translation_ckpt_override))
        tl = tl.to(device).eval()

        predictions: list[str] = []
        for sample in samples:
            sender_text, question = _get_sender_text_and_question(sample, _TASK)

            # Build full prompt (identical for A and B)
            if full_input:
                if _TASK in ("multi_hop", "knowledge_relay"):
                    a_input = f"Context: {sender_text}\nQ: {question}\nA:"
                else:
                    a_input = f"Context: {sender_text}\nInstruction: {question}\nOutput:"
            else:
                a_input = sender_text

            enc = tokenizer_a(
                a_input, return_tensors="pt", truncation=True, max_length=256,
            ).to(device)

            # Extract 3D activations from model A (no pooling)
            act_cache: dict[str, Any] = {}

            def _capture(_module, _inp, output, _c=act_cache):
                h = output[0] if isinstance(output, tuple) else output
                _c["h"] = h.detach()
                return output

            handle_a = model_a
            for part in layer_name_a.split("."):
                handle_a = getattr(handle_a, part)
            hook_h = handle_a.register_forward_hook(_capture)
            with torch.no_grad():
                model_a(**enc)
            hook_h.remove()

            raw_act = act_cache["h"]  # (1, seq, dim_a)
            translated = tl.translate(raw_act.to(device))  # (1, seq, dim_b)

            pred = inject_and_generate(
                model_b,
                tokenizer_b,
                translated,
                layer_name_b,
                question,
                _TASK,
                device=device,
                injection_timing="prefill_only",
                injection_mode="replace",
                context=sender_text if full_input else None,
                full_input=full_input,
            )
            predictions.append(pred)

        evaled = _evaluate_samples(samples, _TASK, predictions)
        acc = _get_acc(evaled["metrics"])
        logger.info("  M6_replace acc = %.4f", acc)
        return {
            "M6_replace": {
                "metrics": evaled["metrics"],
                "config": {
                    "injection_mode": "replace",
                    "injection_timing": "prefill_only",
                    "full_input": full_input,
                    "translation_ckpt": str(translation_ckpt_override),
                },
            }
        }

    # ── Standard P1 variants (additive / replace_interpolate) ──
    texts = load_train_texts(n_samples=_N_TRAIN, seed=_SEED, corpus_path=corpus_path)

    variants = [
        ("P1a", "last_token", True, "mlp_1hidden"),
        ("P1b", "last_token", False, "mlp_1hidden"),
        ("P1c", "last_token", False, "linear"),
    ]

    results: dict[str, dict] = {}

    for tag, pooling, normalize, arch in variants:
        logger.info("=== %s: pooling=%s  normalize=%s  arch=%s ===", tag, pooling, normalize, arch)

        # --- Extract activation pairs ---
        t0 = time.time()
        acts_a, acts_b = extract_activation_pairs(
            model_a,
            tokenizer_a,
            model_b,
            tokenizer_b,
            texts,
            layer_name_a,
            layer_name_b,
            device=device,
            pooling_mode=pooling,  # type: ignore[arg-type]
        )
        logger.info(
            "  Extraction done in %.1fs  shapes: %s / %s",
            time.time() - t0,
            acts_a.shape,
            acts_b.shape,
        )

        # Injection norm statistics (diagnostic)
        inject_norm_mean = acts_a.norm(dim=-1).mean().item()
        hidden_norm_sample = acts_b.norm(dim=-1).mean().item()
        inject_ratio_raw = inject_norm_mean / hidden_norm_sample if hidden_norm_sample > 0 else 0.0
        logger.info(
            "  act_a norm mean=%.2f  act_b norm mean=%.2f  ratio=%.4f",
            inject_norm_mean,
            hidden_norm_sample,
            inject_ratio_raw,
        )

        # --- Train translation layer (skip if checkpoint exists) ---
        ckpt = _CHECKPOINTS_DIR / f"{_PREFIX}_{tag}_{_TIER}_{arch}_n{_N_TRAIN}_s{_SEED}.pt"
        tl = TranslationLayer(dim_a, dim_b, arch=arch, hidden_dim=mlp_hidden, normalize=normalize)  # type: ignore[arg-type]
        if ckpt.exists():
            logger.info("  Checkpoint exists, loading: %s", ckpt.name)
            tl = load_translation_layer(ckpt)
            train_result = {"best_val_loss": float("nan"), "best_epoch": -1, "skipped": True}
        else:
            train_result = train_translation_layer(
                tl,
                acts_a,
                acts_b,
                epochs=30,
                batch_size=32,
                lr=1e-4,
                warmup_steps=100,
                device=device,
                checkpoint_path=ckpt,
                seed=_SEED,
            )
            logger.info(
                "  Train done: best_val_loss=%.6f at epoch %d",
                train_result["best_val_loss"],
                train_result["best_epoch"],
            )

        # --- Evaluate on test samples ---
        tl = tl.to(device).eval()
        predictions = []
        for sample in samples:
            sender_text, question = _get_sender_text_and_question(sample, _TASK)
            enc = tokenizer_a(
                sender_text,
                return_tensors="pt",
                truncation=True,
                max_length=256,
            ).to(device)

            act_cache: dict[str, Any] = {}

            def _capture(module, inp, output, _c=act_cache):
                h = output[0] if isinstance(output, tuple) else output
                _c["h"] = h.detach()
                return output

            handle_a = model_a
            for part in layer_name_a.split("."):
                handle_a = getattr(handle_a, part)
            hook_h = handle_a.register_forward_hook(_capture)
            with torch.no_grad():
                model_a(**enc)
            hook_h.remove()

            raw_act = act_cache["h"]  # (1, seq, dim_a)
            if pooling == "last_token":
                last_pos = enc["attention_mask"].sum(dim=1).long() - 1
                vec_a = raw_act[0, last_pos[0], :].unsqueeze(0)  # (1, dim_a)
            else:
                mask = enc["attention_mask"].unsqueeze(-1).float()
                vec_a = (raw_act * mask).sum(1) / mask.sum(1)

            # Translate
            translated = tl.translate(vec_a.to(device))  # (1, dim_b)

            # Generate with injection
            pred = inject_and_generate(
                model_b,
                tokenizer_b,
                translated,
                layer_name_b,
                question,
                _TASK,
                device=device,
                injection_scale=injection_scale,
                injection_timing=injection_timing,  # type: ignore[arg-type]
                injection_mode=injection_mode,  # type: ignore[arg-type]
            )
            predictions.append(pred)

        evaled = _evaluate_samples(samples, _TASK, predictions)
        acc = _get_acc(evaled["metrics"])
        logger.info("  %s acc = %.4f", tag, acc)

        config = {
            "pooling_mode": pooling,
            "normalize": normalize,
            "arch": arch,
            "injection_mode": injection_mode,
            "injection_timing": injection_timing,
            "injection_scale": injection_scale,
            "inject_norm_mean_raw": inject_norm_mean,
            "hidden_norm_mean": hidden_norm_sample,
            "inject_ratio_raw": inject_ratio_raw,
            "n_train": len(texts),
            "layer_relative": _LAYER_REL,
            "tier": _TIER,
            "task": _TASK,
            "seed": _SEED,
        }
        result_data = {
            "experiment": tag,
            "config": config,
            "metric": evaled["metrics"],
            "n_samples": len(samples),
            "null_baseline": _NULL_BASELINE,
            "train_result": {
                "best_val_loss": train_result["best_val_loss"],
                "best_epoch": train_result["best_epoch"],
            },
            "per_sample": _build_per_sample(evaled["raw"], _TASK),
        }
        fname = f"{_PREFIX}_{tag}_lasttoken_{'l2' if normalize else 'raw'}_{arch}_{_TIER}_{_TASK}_{_SEED}.json"
        _save_result(_RESULTS_DIR / fname, result_data)

        results[tag] = {"metrics": evaled["metrics"], "config": config}

    # Print comparison table
    print("\nP1 Ablation Results (multi_hop Tier1):")
    print(f"{'Variant':<8} {'Pooling':<12} {'Norm':<6} {'Arch':<12} {'Acc':>6}  {'Δnull':>6}")
    print("-" * 55)
    null = _NULL_BASELINE
    for tag, (pooling, normalize, arch) in zip(
        ["P1a", "P1b", "P1c"],
        [
            ("last_token", True, "mlp_1hidden"),
            ("last_token", False, "mlp_1hidden"),
            ("last_token", False, "linear"),
        ],
    ):
        if tag in results:
            acc = _get_acc(results[tag]["metrics"])
            print(
                f"{tag:<8} {pooling:<12} {'L2' if normalize else 'raw':<6} {arch:<12} {acc:>6.4f}  {acc-null:>+.4f}"
            )

    return results


# ---------------------------------------------------------------------------
# P2: Injection scale sweep
# ---------------------------------------------------------------------------


def run_p2_scale_sweep(
    model_a: nn.Module,
    tokenizer_a,
    model_b: nn.Module,
    tokenizer_b,
    samples: list[dict],
    layer_name_a: str,
    layer_name_b: str,
    device: str,
    corpus_path: Path,
    best_p1_tag: str,
    pooling_mode: str,
    normalize: bool,
    arch: str,
    injection_timing: str,
    injection_mode: str,
    scales: list[float] | None = None,
    translation_ckpt_override: str | Path | None = None,
) -> dict:
    """Sweep injection scale on the best P1 configuration.

    Scale ranges differ by normalization state (see experiment guide v3 §5.1).
    For each scale, records actual inject_norm and inject_ratio.

    Args:
        scales: Override the default scale list. If None, uses the full
            diagnostic sweep (8 scales). Pass e.g. [0.01] to evaluate a
            single known-good scale without retraining.
        translation_ckpt_override: If set, load this TranslationLayer
            checkpoint directly, bypassing the prefix-based lookup and
            silent retraining fallback. Required for M4c (T_new trained
            on LoRA-aligned activations).
    """
    if scales is None:
        scales_l2 = [0.0, 0.05, 0.1, 0.3, 0.5, 1.0, 3.0, 10.0]
        scales_raw = [0.0, 0.001, 0.005, 0.01, 0.03, 0.05, 0.1, 0.3]
        scales = scales_l2 if normalize else scales_raw

    logger.info("P2 scale sweep: %s  scales=%s", best_p1_tag, scales)

    # Load translation layer — three modes:
    #  1. Explicit override (--translation-ckpt): load directly, no fallback
    #  2. Prefix-based lookup: {_PREFIX}_{tag}_{tier}_{arch}_n{n}_s{seed}.pt
    #  3. Silent retrain: if no checkpoint found, train from scratch
    if translation_ckpt_override is not None:
        override_path = Path(translation_ckpt_override)
        if not override_path.exists():
            raise FileNotFoundError(
                f"--translation-ckpt path does not exist: {override_path}"
            )
        from rosetta.translation.translation_layer import load_translation_layer
        tl = load_translation_layer(override_path)
        logger.info("  Loaded translation layer from override: %s", override_path.name)
    else:
        # Primary: use current _PREFIX; fallback: m4_ (for enhance runs reusing M4 checkpoints)
        ckpt = _CHECKPOINTS_DIR / f"{_PREFIX}_{best_p1_tag}_{_TIER}_{arch}_n{_N_TRAIN}_s{_SEED}.pt"
        if not ckpt.exists():
            ckpt_m4 = _CHECKPOINTS_DIR / f"m4_{best_p1_tag}_{_TIER}_{arch}_n{_N_TRAIN}_s{_SEED}.pt"
            if ckpt_m4.exists():
                logger.info(
                    "  Checkpoint not found at %s, using M4 fallback: %s",
                    ckpt.name, ckpt_m4.name,
                )
                ckpt = ckpt_m4
        dim_a = _TIERS[_TIER]["sender_hidden_dim"]
        dim_b = _TIERS[_TIER]["receiver_hidden_dim"]
        mlp_hidden = max(dim_a, dim_b)  # C3: avoid bottleneck for T2

        if not ckpt.exists():
            logger.info("  Checkpoint not found, retraining...")
            texts = load_train_texts(n_samples=_N_TRAIN, seed=_SEED, corpus_path=corpus_path)
            acts_a, acts_b = extract_activation_pairs(
                model_a,
                tokenizer_a,
                model_b,
                tokenizer_b,
                texts,
                layer_name_a,
                layer_name_b,
                device=device,
                pooling_mode=pooling_mode,  # type: ignore[arg-type]
            )
            tl = TranslationLayer(dim_a, dim_b, arch=arch, hidden_dim=mlp_hidden, normalize=normalize)  # type: ignore[arg-type]
            train_translation_layer(
                tl,
                acts_a,
                acts_b,
                epochs=30,
                batch_size=32,
                lr=1e-4,
                warmup_steps=100,
                device=device,
                checkpoint_path=ckpt,
                seed=_SEED,
            )
        else:
            from rosetta.translation.translation_layer import load_translation_layer

            tl = load_translation_layer(ckpt)  # loads arch/dims from checkpoint

    tl = tl.to(device).eval()

    # Pre-compute translated vectors for all samples
    logger.info("  Computing translated vectors for all samples...")
    translated_vecs = []
    hidden_norms = []
    sample_texts = [_get_sender_text_and_question(s, _TASK) for s in samples]
    for sample, (sender_text, question) in zip(samples, sample_texts):
        enc = tokenizer_a(sender_text, return_tensors="pt", truncation=True, max_length=256).to(
            device
        )
        act_cache: dict[str, Any] = {}

        def _cap(module, inp, output, _c=act_cache):
            h = output[0] if isinstance(output, tuple) else output
            _c["h"] = h.detach()
            return output

        layer_mod = model_a
        for part in layer_name_a.split("."):
            layer_mod = getattr(layer_mod, part)
        hook = layer_mod.register_forward_hook(_cap)
        with torch.no_grad():
            model_a(**enc)
        hook.remove()

        raw_act = act_cache["h"]
        if pooling_mode == "last_token":
            last_pos = enc["attention_mask"].sum(dim=1).long() - 1
            vec_a = raw_act[0, last_pos[0], :].unsqueeze(0)
        else:
            mask = enc["attention_mask"].unsqueeze(-1).float()
            vec_a = (raw_act * mask).sum(1) / mask.sum(1)

        translated = tl.translate(vec_a.to(device))
        translated_vecs.append(translated)

        # Measure receiver hidden norm at injection layer
        recv_enc = tokenizer_b(f"Q: {question}\nA:", return_tensors="pt").to(device)
        h_cache: dict[str, Any] = {}

        def _h_cap(module, inp, output, _c=h_cache):
            h = output[0] if isinstance(output, tuple) else output
            _c["h"] = h.detach()
            return output

        layer_mod_b = model_b
        for part in layer_name_b.split("."):
            layer_mod_b = getattr(layer_mod_b, part)
        hook_b = layer_mod_b.register_forward_hook(_h_cap)
        with torch.no_grad():
            model_b(**recv_enc)
        hook_b.remove()
        hidden_norms.append(h_cache["h"].norm(dim=-1).mean().item())

    mean_hidden_norm = sum(hidden_norms) / len(hidden_norms)

    # Sweep scales
    sweep_results = []
    for scale in scales:
        logger.info("  scale=%.4f ...", scale)
        predictions = []
        inject_norms = []
        for (sample, (_, question)), t_vec in zip(zip(samples, sample_texts), translated_vecs):
            scaled = t_vec * scale
            inject_norms.append(scaled.norm().item())
            pred = inject_and_generate(
                model_b,
                tokenizer_b,
                t_vec,
                layer_name_b,
                question,
                _TASK,
                device=device,
                injection_scale=scale,
                injection_timing=injection_timing,  # type: ignore[arg-type]
                injection_mode=injection_mode,  # type: ignore[arg-type]
            )
            predictions.append(pred)

        evaled = _evaluate_samples(samples, _TASK, predictions)
        acc = _get_acc(evaled["metrics"])
        inject_norm_mean = sum(inject_norms) / len(inject_norms)
        inject_ratio = inject_norm_mean / mean_hidden_norm if mean_hidden_norm > 0 else 0.0
        logger.info(
            "    acc=%.4f  Δ=%+.4f  inject_norm=%.3f  ratio=%.4f",
            acc,
            acc - _NULL_BASELINE,
            inject_norm_mean,
            inject_ratio,
        )

        result_data = {
            "experiment": "P2",
            "config": {
                "p1_variant": best_p1_tag,
                "pooling_mode": pooling_mode,
                "normalize": normalize,
                "arch": arch,
                "injection_mode": injection_mode,
                "injection_timing": injection_timing,
                "injection_scale": scale,
                "inject_norm": inject_norm_mean,
                "hidden_norm": mean_hidden_norm,
                "inject_ratio": inject_ratio,
                "n_train": _N_TRAIN,
                "layer_relative": _LAYER_REL,
                "tier": _TIER,
                "task": _TASK,
                "seed": _SEED,
            },
            "metric": evaled["metrics"],
            "n_samples": len(samples),
            "null_baseline": _NULL_BASELINE,
            "per_sample": _build_per_sample(evaled["raw"], _TASK),
        }
        scale_tag = f"{scale:.4f}".rstrip("0").rstrip(".")
        fname = f"{_PREFIX}_P2_{best_p1_tag}_scale{scale_tag}_{_TIER}_{_TASK}_{_SEED}.json"
        _save_result(_RESULTS_DIR / fname, result_data)
        sweep_results.append(
            {
                "scale": scale,
                "acc": acc,
                "inject_norm": inject_norm_mean,
                "inject_ratio": inject_ratio,
            }
        )

    # Print sweep table
    print(f"\nP2 Scale Sweep Results ({best_p1_tag}, {_TASK} Tier1):")
    print(f"{'Scale':>8}  {'Acc':>6}  {'Δnull':>6}  {'InjectNorm':>10}  {'Ratio':>6}")
    print("-" * 48)
    for r in sweep_results:
        print(
            f"{r['scale']:>8.4f}  {r['acc']:>6.4f}  {r['acc']-_NULL_BASELINE:>+.4f}"
            f"  {r['inject_norm']:>10.3f}  {r['inject_ratio']:>6.4f}"
        )

    best = max(sweep_results, key=lambda x: x["acc"])
    logger.info("P2 best: scale=%.4f  acc=%.4f", best["scale"], best["acc"])
    return {"sweep": sweep_results, "best": best}


# ---------------------------------------------------------------------------
# E2b: NL + activation combo condition (completes 2x2 factorial design)
# ---------------------------------------------------------------------------


def run_e2b_combo(
    model_a: nn.Module,
    tokenizer_a,
    model_b: nn.Module,
    tokenizer_b,
    samples: list[dict],
    layer_name_a: str,
    layer_name_b: str,
    device: str,
    injection_scale: float = 0.01,
    injection_timing: str = "prefill_only",
    injection_mode: str = "additive",
    p1_tag: str = "P1b",
    arch: str = "mlp_1hidden",
    pooling_mode: str = "last_token",
    normalize: bool = False,
) -> dict:
    """Run NL-text + activation-injection combo condition (E2b).

    Completes the 2x2 factorial design:
        baseline (no NL, no act) | P2 (no NL, act)
        null relay (NL, no act)  | combo (NL, act)  ← this function

    The receiver sees: sender NL relay text + question + layer-16 activation
    injection. This allows decomposing the main effects of NL and activation.

    Args:
        injection_scale: Scale for activation injection (default 0.01, same as P2).
        p1_tag: P1 variant tag whose checkpoint to load (default P1b).
        arch: Translation layer architecture (default mlp_1hidden).

    Returns:
        Dict with metrics, config, per_sample.
    """
    # E2b reuses M4 main-experiment checkpoints (m4_ prefix), not enhance-prefix files.
    # The translation layer was trained during M4 P1; E2b only evaluates with NL context added.
    ckpt = _CHECKPOINTS_DIR / f"m4_{p1_tag}_{_TIER}_{arch}_n{_N_TRAIN}_s{_SEED}.pt"
    if not ckpt.exists():
        # Fallback to current prefix (for non-M4 runs)
        ckpt = _CHECKPOINTS_DIR / f"{_PREFIX}_{p1_tag}_{_TIER}_{arch}_n{_N_TRAIN}_s{_SEED}.pt"
    if not ckpt.exists():
        raise FileNotFoundError(
            f"Checkpoint not found. Expected m4_{p1_tag}_{_TIER}_{arch}_n{_N_TRAIN}_s{_SEED}.pt "
            f"in {_CHECKPOINTS_DIR}. Run P1 first to generate it."
        )
    tl = load_translation_layer(ckpt).to(device).eval()
    logger.info("E2b: loaded checkpoint %s", ckpt.name)

    predictions = []
    inject_norms: list[float] = []

    for sample in samples:
        sender_text, question = _get_sender_text_and_question(sample, _TASK)

        # Extract sender activation
        enc = tokenizer_a(sender_text, return_tensors="pt", truncation=True, max_length=256).to(
            device
        )
        act_cache: dict[str, Any] = {}

        def _cap(module, inp, output, _c=act_cache):  # noqa: ANN001
            h = output[0] if isinstance(output, tuple) else output
            _c["h"] = h.detach()
            return output

        layer_mod = model_a
        for part in layer_name_a.split("."):
            layer_mod = getattr(layer_mod, part)
        hook = layer_mod.register_forward_hook(_cap)
        with torch.no_grad():
            model_a(**enc)
        hook.remove()

        raw_act = act_cache["h"]
        if pooling_mode == "last_token":
            last_pos = enc["attention_mask"].sum(dim=1).long() - 1
            vec_a = raw_act[0, last_pos[0], :].unsqueeze(0)
        else:
            mask = enc["attention_mask"].unsqueeze(-1).float()
            vec_a = (raw_act * mask).sum(1) / mask.sum(1)

        translated = tl.translate(vec_a.to(device))
        inject_norms.append((translated * injection_scale).norm().item())

        # Generate: NL context + question + activation injection
        pred = inject_and_generate(
            model_b,
            tokenizer_b,
            translated,
            layer_name_b,
            question,
            _TASK,
            device=device,
            injection_scale=injection_scale,
            injection_timing=injection_timing,  # type: ignore[arg-type]
            injection_mode=injection_mode,  # type: ignore[arg-type]
            nl_context=sender_text,
        )
        predictions.append(pred)

    evaled = _evaluate_samples(samples, _TASK, predictions)
    acc = _get_acc(evaled["metrics"])
    inject_norm_mean = sum(inject_norms) / len(inject_norms)
    logger.info(
        "E2b combo acc=%.4f  Δ_vs_null=%+.4f  Δ_vs_baseline=%+.4f  Δ_vs_P2=%+.4f",
        acc,
        acc - _NULL_BASELINE,
        acc - 0.1125,  # T2 no-inject baseline
        acc - 0.1333,  # T2 P2 3-seed mean
    )

    result_data = {
        "experiment": "E2b_combo",
        "description": (
            "NL relay text + question + activation injection. "
            "Completes 2x2 factorial: NL(+/-) x activation(+/-). "
            "Reference values: baseline=0.1125, null_relay=0.094, P2_mean=0.1333."
        ),
        "config": {
            "p1_variant": p1_tag,
            "pooling_mode": pooling_mode,
            "normalize": normalize,
            "arch": arch,
            "injection_mode": injection_mode,
            "injection_timing": injection_timing,
            "injection_scale": injection_scale,
            "inject_norm_mean": inject_norm_mean,
            "nl_context": "sender_text (NL relay output)",
            "n_train": _N_TRAIN,
            "layer_relative": _LAYER_REL,
            "tier": _TIER,
            "task": _TASK,
            "seed": _SEED,
        },
        "metric": evaled["metrics"],
        "n_samples": len(samples),
        "null_baseline": _NULL_BASELINE,
        "factorial_2x2": {
            "baseline_no_nl_no_act": 0.1125,
            "null_relay_nl_no_act": 0.094,
            "P2_no_nl_act": 0.1333,
            "combo_nl_act": acc,
        },
        "per_sample": _build_per_sample(evaled["raw"], _TASK),
    }
    fname = f"{_PREFIX}_E2b_combo_scale{injection_scale}_{_TIER}_{_TASK}_{_SEED}.json"
    _save_result(_RESULTS_DIR / fname, result_data)
    return result_data


# ---------------------------------------------------------------------------
# LoRA loading helper (M4b)
# ---------------------------------------------------------------------------


def _load_lora_weights(model: torch.nn.Module, ckpt_path: str,
                       lora_rank: int = 8, lora_alpha: int = 16) -> torch.nn.Module:
    """Apply PEFT LoRA config and load saved delta weights into a model.

    This reconstructs the LoRA wrapper from config (same as training) then
    loads only the LoRA delta weights from the checkpoint, leaving base
    model weights untouched.

    Args:
        model:      Pre-loaded base model (no LoRA yet).
        ckpt_path:  Path to the .pt file saved by LoraAligner.save().
        lora_rank:  Must match the rank used during alignment training.
        lora_alpha: Must match the alpha used during alignment training.

    Returns:
        The model with LoRA adapters applied and delta weights loaded.
    """
    try:
        from peft import get_peft_model, LoraConfig, TaskType
    except ImportError:
        logger.error("peft>=0.9.0 required for LoRA evaluation. pip install peft")
        raise

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=["query_key_value", "dense"],
        lora_dropout=0.0,
        bias="none",
        inference_mode=True,
    )
    model_lora = get_peft_model(model, lora_cfg)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    # Load only LoRA keys (strict=False allows base weights to stay)
    model_lora.load_state_dict(state, strict=False)
    model_lora.eval()
    logger.info("Loaded LoRA weights from %s", ckpt_path)
    return model_lora


# ---------------------------------------------------------------------------
# Main: orchestrate phases
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="M3.5 / M4 Diagnostic Experiments")
    parser.add_argument(
        "--phase",
        default="all",
        choices=["all", "p0", "p0_logits", "p0_generate", "p1", "p2", "e2b"],
        help="Which phase(s) to run (default: all). "
        "p0_logits=CPU hook check only; p0_generate=4-condition comparison only; "
        "e2b=NL+activation combo condition (M4-Enhance E2b)",
    )
    parser.add_argument(
        "--task",
        default="multi_hop",
        choices=["multi_hop", "instruction_following", "knowledge_relay"],
        help="Evaluation task (default: multi_hop)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--tier",
        default="tier1",
        choices=["tier1", "tier2"],
        help="Experiment tier: tier1 (160M/160M-deduped) or tier2 (160M/410M)",
    )
    parser.add_argument(
        "--prefix",
        default="m35",
        help="Result file prefix (default: m35; use m4 for M4 experiments)",
    )
    parser.add_argument("--device", default="auto", help="Compute device: cpu / dml / cuda / auto")
    parser.add_argument(
        "--corpus-path",
        type=Path,
        default=Path("data/corpus/wikitext103_clean.txt"),
        help="Path to Wikitext-103 corpus for P1 training",
    )
    parser.add_argument(
        "--injection-timing",
        default="prefill_only",
        choices=["persistent", "prefill_only"],
        help="Injection timing for P1/P2 (override P0 decision)",
    )
    parser.add_argument(
        "--injection-mode",
        default="additive",
        choices=["additive", "replace_interpolate", "replace"],
        help="Injection mode for P1/P2. 'replace' = guide §4.3.2 original spec.",
    )
    parser.add_argument(
        "--full-input",
        action="store_true",
        help="M6: model B receives full context+question (same as model A). "
        "Required for replace mode to ensure matching sequence lengths.",
    )
    parser.add_argument(
        "--p1-scale",
        type=float,
        default=0.1,
        help="Injection scale for P1 evaluation (before P2 sweep)",
    )
    parser.add_argument(
        "--p2-variant", default="P1b", help="P1 variant to use for P2 scale sweep (default: P1b)"
    )
    parser.add_argument(
        "--p2-single-scale",
        type=float,
        default=None,
        help="Run P2 at a single scale instead of the full sweep. "
        "E.g. --p2-single-scale 0.01 evaluates only scale=0.01.",
    )
    parser.add_argument(
        "--null-baseline",
        type=float,
        default=None,
        help="Null relay accuracy for delta calculations. "
        "If not set, uses 0.144 for tier1 (from M3.5 T0). "
        "Must be set explicitly for tier2 after running null relay.",
    )
    parser.add_argument(
        "--test-file",
        default=None,
        help="Override test data path (e.g. for pilot sets). "
        "If not set, uses the default test.jsonl for the selected task.",
    )
    parser.add_argument(
        "--lora-a-ckpt",
        default=None,
        help="Path to LoRA delta weights for model_a (sender). "
        "If set, applies LoRA adapters to model_a before evaluation (M4b).",
    )
    parser.add_argument(
        "--lora-b-ckpt",
        default=None,
        help="Path to LoRA delta weights for model_b (receiver). "
        "If set, applies LoRA adapters to model_b before evaluation (M4b).",
    )
    parser.add_argument(
        "--translation-ckpt",
        default=None,
        help="Path to a pre-trained TranslationLayer checkpoint (.pt). "
        "If set, P2 uses this checkpoint directly instead of the default "
        "prefix-based lookup or silent retraining. Required for M4c "
        "(T_new trained on LoRA-aligned activations).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit test samples to first N (for smoke testing). "
        "E.g. --max-samples 5 runs only 5 samples for quick validation.",
    )
    parser.add_argument(
        "--reverse",
        action="store_true",
        help="Reverse communication direction: sender=receiver_model, "
        "receiver=sender_model. For B5 bidirectional experiment "
        "(e.g. 410M→160M instead of 160M→410M).",
    )
    args = parser.parse_args()

    # C1/C4: Apply --tier, --prefix, --task, --seed to module-level defaults
    global _TIER, _PREFIX, _TASK, _SEED
    _TIER = args.tier
    _PREFIX = args.prefix
    _TASK = args.task
    _SEED = args.seed

    # Null baseline: require explicit value for tier2
    global _NULL_BASELINE
    if args.null_baseline is not None:
        _NULL_BASELINE = args.null_baseline
    elif _TIER == "tier1":
        _NULL_BASELINE = 0.144
    else:
        _NULL_BASELINE = 0.0  # placeholder until T2 null relay is measured
        logger.warning("No --null-baseline set for %s. Delta calculations will use 0.0.", _TIER)

    if args.device == "auto":
        device = str(get_device())
    elif args.device == "dml":
        try:
            import torch_directml  # type: ignore[import]

            device = str(torch_directml.device())
        except ImportError:
            logger.error(
                "--device dml requires torch-directml. Install with: pip install torch-directml"
            )
            return
    else:
        device = args.device

    logger.info("Device: %s", device)
    logger.info("Phase: %s", args.phase)

    # Load tier config (with optional reversal for B5 bidirectional experiment)
    tier_cfg = _TIERS[_TIER]
    if args.reverse:
        tier_cfg = {
            "sender_id": tier_cfg["receiver_id"],
            "receiver_id": tier_cfg["sender_id"],
            "sender_hidden_dim": tier_cfg["receiver_hidden_dim"],
            "receiver_hidden_dim": tier_cfg["sender_hidden_dim"],
            "sender_num_layers": tier_cfg["receiver_num_layers"],
            "receiver_num_layers": tier_cfg["sender_num_layers"],
            "sender_layer_prefix": tier_cfg["receiver_layer_prefix"],
            "receiver_layer_prefix": tier_cfg["sender_layer_prefix"],
        }
        logger.info("--reverse: sender=%s → receiver=%s",
                     tier_cfg["sender_id"], tier_cfg["receiver_id"])
    layer_name_a = resolve_layer_name(
        tier_cfg["sender_layer_prefix"], tier_cfg["sender_num_layers"], _LAYER_REL
    )
    layer_name_b = resolve_layer_name(
        tier_cfg["receiver_layer_prefix"], tier_cfg["receiver_num_layers"], _LAYER_REL
    )
    layer_idx_b = int(_LAYER_REL * tier_cfg["receiver_num_layers"])
    logger.info("Sender layer: %s  Receiver layer: %s", layer_name_a, layer_name_b)

    # Load test data
    if args.test_file:
        from rosetta.experiments.phase1.m2_baseline import load_jsonl
        samples = load_jsonl(args.test_file)
        logger.info("Loaded %d test samples from %s", len(samples), args.test_file)
    else:
        samples = load_test_split(_TASK)
        logger.info("Loaded %d test samples", len(samples))

    if args.max_samples is not None and args.max_samples < len(samples):
        samples = samples[: args.max_samples]
        logger.info("Truncated to %d samples (--max-samples)", len(samples))

    # ── P0-logits (CPU hook correctness check) ──────────────────────────
    if args.phase in ("all", "p0", "p0_logits"):
        logger.info("Loading receiver model for P0-logits (CPU)...")
        model_b_cpu, tok_b_cpu = load_model_and_tokenizer(tier_cfg["receiver_id"], device="cpu")
        passed = run_p0_logits(model_b_cpu, tok_b_cpu, samples, layer_idx_b, "cpu")
        if not passed:
            logger.error("P0-logits FAILED — hook has a bug. Stopping all experiments.")
            return
        if args.phase == "p0_logits":
            logger.info(
                "P0-logits complete. Re-run with --phase p0_generate --device dml for generate comparison."
            )
            return

    # ── P0-generate (4-condition injection comparison) ───────────────────
    if args.phase in ("all", "p0", "p0_generate"):
        logger.info("P0-logits PASSED. Running P0-generate on device=%s...", device)
        model_b, tok_b = load_model_and_tokenizer(tier_cfg["receiver_id"], device=device)
        if args.lora_b_ckpt:
            model_b = _load_lora_weights(model_b, args.lora_b_ckpt)
        p0_results = run_p0_generate(model_b, tok_b, samples, layer_idx_b, device)

        # Determine recommended timing from P0
        rep_acc = _get_acc(p0_results.get("replace_prefill", {}).get("metrics", {}))
        bas_acc = _get_acc(p0_results.get("baseline", {}).get("metrics", {}))
        logger.info(
            "P0 summary: baseline=%.4f  replace_prefill=%.4f  diff=%.4f",
            bas_acc,
            rep_acc,
            rep_acc - bas_acc,
        )
        if args.phase in ("p0", "p0_generate"):
            return

    # ── P1 ──────────────────────────────────────────────────────────────
    if args.phase in ("all", "p1"):
        logger.info("Loading models for P1...")
        model_a, tok_a = load_model_and_tokenizer(tier_cfg["sender_id"], device=device)
        # Always (re)load model_b here — P0 may have used CPU copy (model_b_cpu)
        model_b, tok_b = load_model_and_tokenizer(tier_cfg["receiver_id"], device=device)

        p1_results = run_p1_ablation(
            model_a,
            tok_a,
            model_b,
            tok_b,
            samples,
            layer_name_a,
            layer_name_b,
            device,
            corpus_path=args.corpus_path,
            injection_timing=args.injection_timing,
            injection_mode=args.injection_mode,
            injection_scale=args.p1_scale,
            full_input=args.full_input,
            translation_ckpt_override=args.translation_ckpt,
        )

        # Check if any variant has positive signal
        best_p1 = max(p1_results.items(), key=lambda x: _get_acc(x[1]["metrics"]))
        best_acc = _get_acc(best_p1[1]["metrics"])
        delta = best_acc - _NULL_BASELINE
        logger.info(
            "P1 best: %s  acc=%.4f  Δ=%+.4f",
            best_p1[0],
            best_acc,
            delta,
        )
        if delta < 0.02:
            logger.warning(
                "P1 best Δ=%.4f < 0.02 threshold. Proceeding to P2 scale sweep anyway.", delta
            )

        if args.phase == "p1":
            return

    # ── P2 ──────────────────────────────────────────────────────────────
    if args.phase in ("all", "p2"):
        logger.info("Running P2 scale sweep on variant %s...", args.p2_variant)
        # Always load models — ensures they are bound regardless of prior phases
        model_a, tok_a = load_model_and_tokenizer(tier_cfg["sender_id"], device=device)
        model_b, tok_b = load_model_and_tokenizer(tier_cfg["receiver_id"], device=device)
        if args.lora_a_ckpt:
            model_a = _load_lora_weights(model_a, args.lora_a_ckpt)
        if args.lora_b_ckpt:
            model_b = _load_lora_weights(model_b, args.lora_b_ckpt)

        # Determine config from P1 variant tag
        p1_configs = {
            "P1a": ("last_token", True, "mlp_1hidden"),
            "P1b": ("last_token", False, "mlp_1hidden"),
            "P1c": ("last_token", False, "linear"),
        }
        pooling, normalize, arch = p1_configs.get(
            args.p2_variant, ("last_token", False, "mlp_1hidden")
        )

        p2_scales = [args.p2_single_scale] if args.p2_single_scale is not None else None
        run_p2_scale_sweep(
            model_a,
            tok_a,
            model_b,
            tok_b,
            samples,
            layer_name_a,
            layer_name_b,
            device,
            corpus_path=args.corpus_path,
            best_p1_tag=args.p2_variant,
            pooling_mode=pooling,
            normalize=normalize,
            arch=arch,
            injection_timing=args.injection_timing,
            injection_mode=args.injection_mode,
            scales=p2_scales,
            translation_ckpt_override=args.translation_ckpt,
        )

    # ── E2b: NL + activation combo (M4-Enhance) ─────────────────────────
    if args.phase == "e2b":
        logger.info("Running E2b NL+activation combo condition (M4-Enhance)...")
        model_a, tok_a = load_model_and_tokenizer(tier_cfg["sender_id"], device=device)
        model_b, tok_b = load_model_and_tokenizer(tier_cfg["receiver_id"], device=device)

        p1_configs = {
            "P1a": ("last_token", True, "mlp_1hidden"),
            "P1b": ("last_token", False, "mlp_1hidden"),
            "P1c": ("last_token", False, "linear"),
        }
        pooling, normalize, arch = p1_configs.get(
            args.p2_variant, ("last_token", False, "mlp_1hidden")
        )

        result = run_e2b_combo(
            model_a,
            tok_a,
            model_b,
            tok_b,
            samples,
            layer_name_a,
            layer_name_b,
            device,
            injection_scale=args.p1_scale,
            injection_timing=args.injection_timing,
            injection_mode=args.injection_mode,
            p1_tag=args.p2_variant,
            arch=arch,
            pooling_mode=pooling,
            normalize=normalize,
        )
        acc = _get_acc(result["metric"])
        logger.info("E2b done. acc=%.4f  2x2 table: %s", acc, result["factorial_2x2"])


if __name__ == "__main__":
    main()
