"""Final clean-rerun runner for the H1 activation-transfer paper package.

This module is intentionally separate from the historical M6/M7 runners.  It
keeps the final paper protocol explicit: clean-eval input, shared tokenization
settings, per-sample outputs, and control conditions that can be summarized by
the paper workspace scripts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import time
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any, Literal, Sequence

import torch
import torch.nn.functional as F

from rosetta.experiments.phase1.m2_baseline import (
    evaluate_instruction_following,
    evaluate_knowledge_relay,
    evaluate_multi_hop,
    get_device,
    load_jsonl,
    load_model_and_tokenizer,
)
from rosetta.experiments.phase1.m3_translation import (
    _DEFAULT_RELATIVE_LAYER,
    _TIERS,
    resolve_layer_name,
)
from rosetta.experiments.phase1.m35_diagnostic import _get_sender_text_and_question
from rosetta.translation.nl_relay import build_receiver_prompt, build_sender_prompt
from rosetta.translation.translation_layer import load_translation_layer

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[4]
_DEFAULT_TEST_FILE = _ROOT / "data" / "tasks" / "multi_hop_reasoning" / "clean_eval.jsonl"
_DEFAULT_OUTPUT_DIR = _ROOT / "papers" / "h1_activation_transfer" / "results" / "final"
_DEFAULT_CKPT_DIR = _ROOT / "results" / "phase1" / "checkpoints"
_TASK = "multi_hop"
_SCHEMA_VERSION = "h1_final_runner.v1"

_TRANSLATION_CONDITIONS = {
    "additive",
    "replace",
    "scale_corrected",
    "best_alpha",
    "shuffled_translation",
    "shuffled_translation_strict_matched",
}
_DETERMINISTIC_CONDITIONS = {"no_inject", "nl_relay", "b_to_b_self_inject", "zero_replacement"}
_ALL_CONDITIONS = (
    "no_inject",
    "nl_relay",
    "additive",
    "replace",
    "scale_corrected",
    "best_alpha",
    "b_to_b_self_inject",
    "same_norm_random",
    "shuffled_translation",
    "shuffled_translation_strict_matched",
    "zero_replacement",
)
_MAIN_CONDITIONS = (
    "no_inject",
    "nl_relay",
    "additive",
    "replace",
    "scale_corrected",
    "best_alpha",
)
_CONTROL_CONDITIONS = (
    "b_to_b_self_inject",
    "same_norm_random",
    "shuffled_translation",
    "zero_replacement",
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve_device(device_arg: str):
    if device_arg == "auto":
        return get_device()
    return torch.device(device_arg)


def _direction_cfg(tier: str, direction: Literal["fwd", "rev"]) -> dict[str, Any]:
    cfg = dict(_TIERS[tier])
    if direction == "fwd":
        return cfg
    return {
        "sender_id": cfg["receiver_id"],
        "receiver_id": cfg["sender_id"],
        "sender_hidden_dim": cfg["receiver_hidden_dim"],
        "receiver_hidden_dim": cfg["sender_hidden_dim"],
        "sender_num_layers": cfg["receiver_num_layers"],
        "receiver_num_layers": cfg["sender_num_layers"],
        "sender_layer_prefix": cfg["receiver_layer_prefix"],
        "receiver_layer_prefix": cfg["sender_layer_prefix"],
    }


def _resolve_layers(cfg: dict[str, Any], layer_rel: float) -> tuple[str, str]:
    layer_a = resolve_layer_name(cfg["sender_layer_prefix"], cfg["sender_num_layers"], layer_rel)
    layer_b = resolve_layer_name(
        cfg["receiver_layer_prefix"], cfg["receiver_num_layers"], layer_rel
    )
    return layer_a, layer_b


def _module_by_name(model: torch.nn.Module, layer_name: str) -> torch.nn.Module:
    module = model
    for part in layer_name.split("."):
        module = getattr(module, part)
    return module


def _generation_kwargs(max_new_tokens: int) -> dict[str, Any]:
    return {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "repetition_penalty": 1.3,
        "no_repeat_ngram_size": 3,
    }


def _encode(tokenizer, prompt: str, device, max_length: int):
    return tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    ).to(device)


def _input_ids_list(inputs) -> list[int]:
    return inputs["input_ids"][0].detach().cpu().tolist()


def _full_input_prompt(sample: dict[str, Any], task: str) -> tuple[str, str, str]:
    sender_text, question = _get_sender_text_and_question(sample, task)
    if task in ("multi_hop", "knowledge_relay"):
        prompt = f"Context: {sender_text}\nQ: {question}\nA:"
    else:
        prompt = f"Context: {sender_text}\nInstruction: {question}\nOutput:"
    return prompt, sender_text, question


def _gold_value(sample: dict[str, Any], task: str) -> Any:
    if task in ("multi_hop", "knowledge_relay"):
        return sample.get("answer", "")
    return sample.get("constraints", [])


def _evaluate_prediction(prediction: str, sample: dict[str, Any], task: str) -> dict[str, Any]:
    if task == "multi_hop":
        return evaluate_multi_hop(prediction, str(sample["answer"]))
    if task == "knowledge_relay":
        return evaluate_knowledge_relay(prediction, str(sample["answer"]))
    return evaluate_instruction_following(prediction, list(sample["constraints"]))


def _sample_id(sample: dict[str, Any], idx: int) -> str:
    for key in ("clean_eval_id", "id", "sample_id"):
        if key in sample:
            return str(sample[key])
    return f"idx:{idx}"


def _record_prediction(
    sample: dict[str, Any],
    idx: int,
    task: str,
    prediction: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    eval_result = _evaluate_prediction(prediction, sample, task)
    record = {
        "idx": idx,
        "sample_id": _sample_id(sample, idx),
        "gold": _gold_value(sample, task),
        "prediction": prediction,
        "legacy_contains": bool(eval_result.get("correct", False)),
        "legacy_exact_match": bool(eval_result.get("exact_match", False)),
        "eval_result": eval_result,
    }
    if "source_eval_index" in sample:
        record["source_eval_index"] = sample["source_eval_index"]
    if "source_component_file" in sample:
        record["source_component_file"] = sample["source_component_file"]
    if extra:
        record.update(extra)
    return record


def _aggregate_metrics(per_sample: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(per_sample)
    if n == 0:
        return {"n": 0, "contains_acc": 0.0, "exact_match_acc": 0.0}
    return {
        "n": n,
        "contains_acc": sum(1 for r in per_sample if r.get("legacy_contains")) / n,
        "exact_match_acc": sum(1 for r in per_sample if r.get("legacy_exact_match")) / n,
    }


def _mean_of(per_sample: list[dict[str, Any]], key: str) -> float | None:
    vals = [r[key] for r in per_sample if isinstance(r.get(key), (int, float))]
    return float(mean(vals)) if vals else None


def _aggregate_diagnostics(per_sample: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "mean_prompt_len_a": _mean_of(per_sample, "prompt_len_a"),
        "mean_prompt_len_b": _mean_of(per_sample, "prompt_len_b"),
        "mean_translated_norm": _mean_of(per_sample, "translated_norm_mean"),
        "mean_b_hidden_norm": _mean_of(per_sample, "b_hidden_norm_mean"),
        "mean_injected_norm": _mean_of(per_sample, "injected_norm_mean"),
        "seq_len_mismatch_count": sum(1 for r in per_sample if r.get("seq_len_match") is False),
        "token_mismatch_count": sum(1 for r in per_sample if r.get("token_ids_match") is False),
        "shuffle_self_fallback_count": sum(
            1 for r in per_sample if r.get("shuffle_self_fallback") is True
        ),
    }


@torch.no_grad()
def _generate_plain(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    device,
    max_length: int,
    gen_kwargs: dict[str, Any],
) -> tuple[str, int, list[int]]:
    model = model.to(device).eval()
    inputs = _encode(tokenizer, prompt, device, max_length)
    prompt_len = inputs["input_ids"].shape[1]
    kwargs = dict(gen_kwargs)
    kwargs.setdefault("pad_token_id", tokenizer.eos_token_id)
    output_ids = model.generate(**inputs, **kwargs)
    new_ids = output_ids[0, prompt_len:]
    return (
        tokenizer.decode(new_ids, skip_special_tokens=True).strip(),
        prompt_len,
        _input_ids_list(inputs),
    )


@torch.no_grad()
def _extract_layer_hidden(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    layer_name: str,
    device,
    max_length: int,
) -> tuple[torch.Tensor, int, list[int]]:
    model = model.to(device).eval()
    inputs = _encode(tokenizer, prompt, device, max_length)
    captured: dict[str, torch.Tensor] = {}

    def _capture(_module, _input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured["hidden"] = hidden.detach().clone()
        return output

    handle = _module_by_name(model, layer_name).register_forward_hook(_capture)
    try:
        model(**inputs)
    finally:
        handle.remove()
    if "hidden" not in captured:
        raise RuntimeError(f"Layer hook did not capture activations for {layer_name}")
    return captured["hidden"], inputs["input_ids"].shape[1], _input_ids_list(inputs)


def _tensor_norm_mean(x: torch.Tensor) -> float:
    return float(x.float().norm(dim=-1).mean().detach().cpu().item())


@torch.no_grad()
def _generate_with_layer_injection(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    layer_name: str,
    activations: torch.Tensor,
    injection_mode: Literal["additive", "replace", "replace_scale_corrected"],
    device,
    max_length: int,
    gen_kwargs: dict[str, Any],
    injection_scale: float = 1.0,
    injection_alpha: float = 1.0,
    expected_input_ids: list[int] | None = None,
    strict_token_match: bool = True,
) -> tuple[str, dict[str, Any]]:
    """Generate from a fixed prompt with a prefill-only layer hook.

    This duplicates the small hook path from ``inject_and_generate`` so the
    final runner can own the exact prompt string and tokenization arguments.
    Replacement-style conditions are protocol violations if prompt lengths or
    token IDs diverge.
    """

    model = model.to(device).eval()
    inputs = _encode(tokenizer, prompt, device, max_length)
    input_ids = _input_ids_list(inputs)
    prompt_len = inputs["input_ids"].shape[1]
    token_ids_match = expected_input_ids is None or input_ids == expected_input_ids

    if strict_token_match and expected_input_ids is not None and not token_ids_match:
        raise ValueError("A/B token IDs differ under the shared full-input prompt")

    acts = activations.detach().to(device)
    if injection_mode == "additive":
        if acts.dim() == 1:
            acts = acts.unsqueeze(0)
        if acts.dim() == 3:
            acts = acts[:, -1, :]
        if acts.dim() != 2:
            raise ValueError(f"Additive injection expects 1D/2D/3D activations, got {acts.dim()}D")
    else:
        if acts.dim() == 2:
            acts = acts.unsqueeze(0)
        if acts.dim() != 3:
            raise ValueError(f"Replacement injection expects 3D activations, got {acts.dim()}D")
        if acts.shape[1] != prompt_len:
            raise ValueError(
                f"Seq length mismatch: activations={acts.shape[1]} receiver_prompt={prompt_len}"
            )

    diagnostics: dict[str, Any] = {
        "prompt_len_b": prompt_len,
        "token_ids_match": token_ids_match,
        "seq_len_match": bool(acts.dim() != 3 or acts.shape[1] == prompt_len),
        "translated_norm_mean": _tensor_norm_mean(acts),
    }
    inject_done = False

    def _hook(_module, _input, output):
        nonlocal inject_done
        hidden = output[0] if isinstance(output, tuple) else output
        if inject_done or hidden.shape[1] == 1:
            return output
        inject_done = True
        diagnostics["b_hidden_norm_mean"] = _tensor_norm_mean(hidden)

        if injection_mode == "additive":
            vector = (
                (acts * injection_scale).unsqueeze(1).to(dtype=hidden.dtype, device=hidden.device)
            )
            new_hidden = hidden + vector
            diagnostics["injected_norm_mean"] = _tensor_norm_mean(vector)
        elif injection_mode == "replace":
            new_hidden = acts.to(dtype=hidden.dtype, device=hidden.device)
            diagnostics["injected_norm_mean"] = _tensor_norm_mean(new_hidden)
        else:
            target_norm = hidden.norm(dim=-1, keepdim=True)
            translated = acts.to(dtype=hidden.dtype, device=hidden.device)
            corrected = F.normalize(translated, p=2, dim=-1) * target_norm
            new_hidden = injection_alpha * corrected + (1.0 - injection_alpha) * hidden
            diagnostics["injected_norm_mean"] = _tensor_norm_mean(new_hidden)

        if isinstance(output, tuple):
            return (new_hidden,) + output[1:]
        return new_hidden

    handle = _module_by_name(model, layer_name).register_forward_hook(_hook)
    try:
        kwargs = dict(gen_kwargs)
        kwargs.setdefault("pad_token_id", tokenizer.eos_token_id)
        output_ids = model.generate(**inputs, **kwargs)
    finally:
        handle.remove()
    if not inject_done:
        raise RuntimeError("Injection hook did not fire during prompt prefill")

    new_ids = output_ids[0, prompt_len:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip(), diagnostics


def _checkpoint_path(ckpt_dir: Path, direction: str, seed: int) -> Path:
    return ckpt_dir / f"m6_translation_{direction}_seed{seed}.pt"


def _load_samples(test_file: Path, max_samples: int | None) -> list[dict[str, Any]]:
    samples = load_jsonl(test_file)
    if max_samples is not None and max_samples > 0:
        return samples[:max_samples]
    return samples


def _base_result(
    *,
    condition: str,
    direction: str,
    seed: int | None,
    deterministic: bool,
    tier: str,
    task: str,
    test_file: Path,
    clean_eval_hash: str,
    samples: list[dict[str, Any]],
    cfg: dict[str, Any],
    layer_a: str | None,
    layer_b: str,
    device,
    max_length: int,
    layer_rel: float,
    gen_kwargs: dict[str, Any],
    sender_gen_kwargs: dict[str, Any] | None = None,
    checkpoint: Path | None = None,
    injection: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result = {
        "schema_version": _SCHEMA_VERSION,
        "created_at": _utc_now(),
        "condition": condition,
        "direction": direction,
        "seed": seed,
        "deterministic": deterministic,
        "config": {
            "tier": tier,
            "task": task,
            "test_file": str(test_file),
            "clean_eval_hash": clean_eval_hash,
            "clean_eval_version": samples[0].get("clean_eval_version") if samples else None,
            "n_samples": len(samples),
            "max_length": max_length,
            "layer_rel": layer_rel,
            "device": str(device),
            "generation": gen_kwargs,
            "sender_generation": sender_gen_kwargs,
        },
        "models": {
            "model_a": cfg.get("sender_id"),
            "model_b": cfg.get("receiver_id"),
            "layer_a": layer_a,
            "layer_b": layer_b,
        },
        "checkpoint": str(checkpoint) if checkpoint is not None else None,
        "injection": injection or {"mode": "none"},
    }
    if deterministic:
        result["deterministic_reason"] = (
            "greedy decoding with do_sample=false; this condition does not consume a "
            "seed-dependent translation checkpoint or random control."
        )
    return result


def _save_result(result: dict[str, Any], output_dir: Path) -> Path:
    condition = result["condition"]
    direction = result["direction"]
    seed = result.get("seed")
    deterministic = result.get("deterministic", False)
    seed_tag = "deterministic" if deterministic else f"seed{seed}"
    alpha = result.get("injection", {}).get("alpha")
    alpha_tag = ""
    if alpha is not None and condition in {
        "best_alpha",
        "scale_corrected",
        "shuffled_translation",
        "shuffled_translation_strict_matched",
    }:
        alpha_tag = f"_alpha{str(alpha).replace('.', 'p')}"
    out_path = output_dir / f"h1_{condition}_{direction}_{seed_tag}{alpha_tag}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    logger.info("Saved %s", out_path)
    return out_path


def _finish_result(result: dict[str, Any], per_sample: list[dict[str, Any]]) -> dict[str, Any]:
    result["per_sample"] = per_sample
    result["metrics"] = _aggregate_metrics(per_sample)
    result["diagnostics"] = _aggregate_diagnostics(per_sample)
    return result


def run_no_inject(
    *,
    samples: list[dict[str, Any]],
    model_b,
    tokenizer_b,
    cfg: dict[str, Any],
    layer_b: str,
    args: argparse.Namespace,
    direction: str,
    clean_eval_hash: str,
    device,
) -> dict[str, Any]:
    gen_kwargs = _generation_kwargs(args.max_new_tokens)
    result = _base_result(
        condition="no_inject",
        direction=direction,
        seed=None,
        deterministic=True,
        tier=args.tier,
        task=args.task,
        test_file=args.test_file,
        clean_eval_hash=clean_eval_hash,
        samples=samples,
        cfg=cfg,
        layer_a=None,
        layer_b=layer_b,
        device=device,
        max_length=args.max_length,
        layer_rel=args.layer_rel,
        gen_kwargs=gen_kwargs,
    )
    per_sample = []
    t0 = time.time()
    for idx, sample in enumerate(samples):
        prompt, _sender_text, _question = _full_input_prompt(sample, args.task)
        prediction, prompt_len, _ids = _generate_plain(
            model_b, tokenizer_b, prompt, device, args.max_length, gen_kwargs
        )
        per_sample.append(
            _record_prediction(
                sample,
                idx,
                args.task,
                prediction,
                {"prompt_len_b": prompt_len, "prompt_mode": "full_input"},
            )
        )
    result["elapsed_sec"] = round(time.time() - t0, 3)
    return _finish_result(result, per_sample)


def run_nl_relay(
    *,
    samples: list[dict[str, Any]],
    model_a,
    tokenizer_a,
    model_b,
    tokenizer_b,
    cfg: dict[str, Any],
    layer_a: str,
    layer_b: str,
    args: argparse.Namespace,
    direction: str,
    clean_eval_hash: str,
    device,
) -> dict[str, Any]:
    sender_gen_kwargs = _generation_kwargs(args.sender_max_new_tokens)
    receiver_gen_kwargs = _generation_kwargs(args.max_new_tokens)
    result = _base_result(
        condition="nl_relay",
        direction=direction,
        seed=None,
        deterministic=True,
        tier=args.tier,
        task=args.task,
        test_file=args.test_file,
        clean_eval_hash=clean_eval_hash,
        samples=samples,
        cfg=cfg,
        layer_a=layer_a,
        layer_b=layer_b,
        device=device,
        max_length=args.max_length,
        layer_rel=args.layer_rel,
        gen_kwargs=receiver_gen_kwargs,
        sender_gen_kwargs=sender_gen_kwargs,
        injection={"mode": "natural_language"},
    )
    per_sample = []
    t0 = time.time()
    for idx, sample in enumerate(samples):
        _sender_text, question = _get_sender_text_and_question(sample, args.task)
        sender_prompt = build_sender_prompt(sample)
        relay, sender_prompt_len, _sender_ids = _generate_plain(
            model_a, tokenizer_a, sender_prompt, device, args.max_length, sender_gen_kwargs
        )
        receiver_prompt = build_receiver_prompt(relay, question, task=args.task)
        prediction, receiver_prompt_len, _receiver_ids = _generate_plain(
            model_b, tokenizer_b, receiver_prompt, device, args.max_length, receiver_gen_kwargs
        )
        per_sample.append(
            _record_prediction(
                sample,
                idx,
                args.task,
                prediction,
                {
                    "relay": relay,
                    "sender_prompt_len": sender_prompt_len,
                    "prompt_len_b": receiver_prompt_len,
                    "prompt_mode": "nl_relay",
                },
            )
        )
    result["elapsed_sec"] = round(time.time() - t0, 3)
    return _finish_result(result, per_sample)


def _translated_activation_for_sample(
    *,
    sample: dict[str, Any],
    model_a,
    tokenizer_a,
    tl,
    layer_a: str,
    args: argparse.Namespace,
    device,
) -> tuple[str, torch.Tensor, int, list[int], str]:
    prompt, _sender_text, question = _full_input_prompt(sample, args.task)
    raw_act, prompt_len_a, input_ids_a = _extract_layer_hidden(
        model_a, tokenizer_a, prompt, layer_a, device, args.max_length
    )
    translated = tl.translate(raw_act).detach()
    return prompt, translated, prompt_len_a, input_ids_a, question


def _same_length_derangement(
    prompt_lengths: dict[int, int],
    seed: int,
    strict: bool,
) -> tuple[dict[int, int], int]:
    buckets: dict[int, list[int]] = {}
    for idx, prompt_len in prompt_lengths.items():
        buckets.setdefault(prompt_len, []).append(idx)

    rng = random.Random(seed)
    mapping: dict[int, int] = {}
    fallback_count = 0
    for _prompt_len, indices in buckets.items():
        shuffled = list(indices)
        rng.shuffle(shuffled)
        if len(shuffled) == 1:
            if strict:
                raise ValueError(
                    "Cannot build strict same-length shuffled control with singleton buckets"
                )
            mapping[shuffled[0]] = shuffled[0]
            fallback_count += 1
            continue
        shifted = shuffled[1:] + shuffled[:1]
        for src, dst in zip(shuffled, shifted, strict=True):
            mapping[src] = dst
    return mapping, fallback_count


def _prepare_shuffled_cache(
    *,
    samples: list[dict[str, Any]],
    model_a,
    tokenizer_a,
    tokenizer_b,
    tl,
    layer_a: str,
    args: argparse.Namespace,
    device,
    seed: int,
    drop_singletons: bool = False,
) -> tuple[dict[int, dict[str, Any]], dict[int, int], int, int]:
    cache: dict[int, dict[str, Any]] = {}
    lengths: dict[int, int] = {}
    for idx, sample in enumerate(samples):
        prompt, translated, prompt_len_a, input_ids_a, question = _translated_activation_for_sample(
            sample=sample,
            model_a=model_a,
            tokenizer_a=tokenizer_a,
            tl=tl,
            layer_a=layer_a,
            args=args,
            device=device,
        )
        inputs_b = _encode(tokenizer_b, prompt, device, args.max_length)
        input_ids_b = _input_ids_list(inputs_b)
        prompt_len_b = inputs_b["input_ids"].shape[1]
        if args.strict_token_match and input_ids_a != input_ids_b:
            raise ValueError("A/B token IDs differ while preparing shuffled cache")
        if translated.shape[1] != prompt_len_b:
            raise ValueError(
                f"Translated seq length {translated.shape[1]} != receiver prompt {prompt_len_b}"
            )
        cache[idx] = {
            "prompt": prompt,
            "question": question,
            "translated": translated.detach().cpu(),
            "prompt_len_a": prompt_len_a,
            "input_ids_a": input_ids_a,
            "prompt_len_b": prompt_len_b,
            "token_ids_match": input_ids_a == input_ids_b,
        }
        lengths[idx] = prompt_len_b

    if drop_singletons:
        buckets: dict[int, list[int]] = {}
        for idx, prompt_len in lengths.items():
            buckets.setdefault(prompt_len, []).append(idx)
        eligible = {idx for indices in buckets.values() if len(indices) >= 2 for idx in indices}
        excluded_count = len(cache) - len(eligible)
        cache = {idx: item for idx, item in cache.items() if idx in eligible}
        lengths = {idx: prompt_len for idx, prompt_len in lengths.items() if idx in eligible}
        mapping, fallback_count = _same_length_derangement(lengths, seed, strict=True)
    else:
        excluded_count = 0
        mapping, fallback_count = _same_length_derangement(lengths, seed, args.strict_shuffle)

    return cache, mapping, fallback_count, excluded_count


def run_translation_condition(
    *,
    condition: str,
    direction: str,
    seed: int,
    samples: list[dict[str, Any]],
    model_a,
    tokenizer_a,
    model_b,
    tokenizer_b,
    cfg: dict[str, Any],
    layer_a: str,
    layer_b: str,
    args: argparse.Namespace,
    clean_eval_hash: str,
    device,
) -> dict[str, Any]:
    ckpt = _checkpoint_path(args.ckpt_dir, direction, seed)
    if not ckpt.exists():
        raise FileNotFoundError(f"Translation checkpoint not found: {ckpt}")

    tl = load_translation_layer(ckpt, device="cpu").to(device).eval()
    gen_kwargs = _generation_kwargs(args.max_new_tokens)
    alpha = None
    injection_scale = None
    if condition == "additive":
        injection_mode = "additive"
        injection_scale = args.injection_scale
    elif condition == "replace":
        injection_mode = "replace"
    elif condition == "scale_corrected":
        injection_mode = "replace_scale_corrected"
        alpha = args.alpha
    elif condition == "best_alpha":
        injection_mode = "replace_scale_corrected"
        alpha = args.best_alpha
    elif condition in ("shuffled_translation", "shuffled_translation_strict_matched"):
        injection_mode = (
            "replace" if args.shuffle_injection_mode == "replace" else "replace_scale_corrected"
        )
        alpha = args.alpha if injection_mode == "replace_scale_corrected" else None
    else:
        raise ValueError(f"Unsupported translation condition: {condition}")

    result = _base_result(
        condition=condition,
        direction=direction,
        seed=seed,
        deterministic=False,
        tier=args.tier,
        task=args.task,
        test_file=args.test_file,
        clean_eval_hash=clean_eval_hash,
        samples=samples,
        cfg=cfg,
        layer_a=layer_a,
        layer_b=layer_b,
        device=device,
        max_length=args.max_length,
        layer_rel=args.layer_rel,
        gen_kwargs=gen_kwargs,
        checkpoint=ckpt,
        injection={
            "mode": injection_mode,
            "timing": "prefill_only",
            "scale": injection_scale,
            "alpha": alpha,
            "shuffle_injection_mode": (
                args.shuffle_injection_mode
                if condition in ("shuffled_translation", "shuffled_translation_strict_matched")
                else None
            ),
            "strict_matched_subset": condition == "shuffled_translation_strict_matched",
        },
    )
    per_sample = []
    t0 = time.time()
    shuffled_cache: dict[int, dict[str, Any]] | None = None
    shuffle_mapping: dict[int, int] = {}
    shuffle_fallback_count = 0
    shuffle_excluded_count = 0
    if condition in ("shuffled_translation", "shuffled_translation_strict_matched"):
        shuffled_cache, shuffle_mapping, shuffle_fallback_count, shuffle_excluded_count = (
            _prepare_shuffled_cache(
                samples=samples,
                model_a=model_a,
                tokenizer_a=tokenizer_a,
                tokenizer_b=tokenizer_b,
                tl=tl,
                layer_a=layer_a,
                args=args,
                device=device,
                seed=seed,
                drop_singletons=condition == "shuffled_translation_strict_matched",
            )
        )
        result["config"]["n_samples"] = len(shuffled_cache)
        result["config"]["source_clean_eval_n_samples"] = len(samples)
        result["injection"]["shuffle_excluded_singleton_count"] = shuffle_excluded_count

    eval_indices = (
        sorted(shuffled_cache) if shuffled_cache is not None else list(range(len(samples)))
    )

    for idx in eval_indices:
        sample = samples[idx]
        if shuffled_cache is not None:
            current = shuffled_cache[idx]
            donor_idx = shuffle_mapping[idx]
            donor = shuffled_cache[donor_idx]
            prompt = current["prompt"]
            translated = donor["translated"].to(device)
            prompt_len_a = current["prompt_len_a"]
            input_ids_a = current["input_ids_a"]
            extra_shuffle = {
                "shuffle_donor_idx": donor_idx,
                "shuffle_self_fallback": donor_idx == idx,
                "shuffle_fallback_count_run": shuffle_fallback_count,
                "shuffle_excluded_singleton_count_run": shuffle_excluded_count,
            }
        else:
            prompt, translated, prompt_len_a, input_ids_a, _question = (
                _translated_activation_for_sample(
                    sample=sample,
                    model_a=model_a,
                    tokenizer_a=tokenizer_a,
                    tl=tl,
                    layer_a=layer_a,
                    args=args,
                    device=device,
                )
            )
            extra_shuffle = {}

        prediction, diagnostics = _generate_with_layer_injection(
            model_b,
            tokenizer_b,
            prompt,
            layer_b,
            translated,
            injection_mode=injection_mode,
            device=device,
            max_length=args.max_length,
            gen_kwargs=gen_kwargs,
            injection_scale=args.injection_scale,
            injection_alpha=alpha if alpha is not None else 1.0,
            expected_input_ids=input_ids_a,
            strict_token_match=args.strict_token_match,
        )
        per_sample.append(
            _record_prediction(
                sample,
                idx,
                args.task,
                prediction,
                {
                    "prompt_len_a": prompt_len_a,
                    "prompt_mode": "full_input",
                    **diagnostics,
                    **extra_shuffle,
                },
            )
        )

    result["elapsed_sec"] = round(time.time() - t0, 3)
    return _finish_result(result, per_sample)


def run_self_inject(
    *,
    samples: list[dict[str, Any]],
    model_b,
    tokenizer_b,
    cfg: dict[str, Any],
    layer_b: str,
    args: argparse.Namespace,
    direction: str,
    clean_eval_hash: str,
    device,
) -> dict[str, Any]:
    gen_kwargs = _generation_kwargs(args.max_new_tokens)
    result = _base_result(
        condition="b_to_b_self_inject",
        direction=direction,
        seed=None,
        deterministic=True,
        tier=args.tier,
        task=args.task,
        test_file=args.test_file,
        clean_eval_hash=clean_eval_hash,
        samples=samples,
        cfg={**cfg, "sender_id": cfg["receiver_id"]},
        layer_a=layer_b,
        layer_b=layer_b,
        device=device,
        max_length=args.max_length,
        layer_rel=args.layer_rel,
        gen_kwargs=gen_kwargs,
        injection={"mode": "replace", "timing": "prefill_only", "source": "receiver_self"},
    )
    per_sample = []
    t0 = time.time()
    for idx, sample in enumerate(samples):
        prompt, _sender_text, _question = _full_input_prompt(sample, args.task)
        own_hidden, prompt_len_b, input_ids_b = _extract_layer_hidden(
            model_b, tokenizer_b, prompt, layer_b, device, args.max_length
        )
        prediction, diagnostics = _generate_with_layer_injection(
            model_b,
            tokenizer_b,
            prompt,
            layer_b,
            own_hidden,
            injection_mode="replace",
            device=device,
            max_length=args.max_length,
            gen_kwargs=gen_kwargs,
            expected_input_ids=input_ids_b,
            strict_token_match=True,
        )
        per_sample.append(
            _record_prediction(
                sample,
                idx,
                args.task,
                prediction,
                {"prompt_len_b": prompt_len_b, "prompt_mode": "full_input", **diagnostics},
            )
        )
    result["elapsed_sec"] = round(time.time() - t0, 3)
    return _finish_result(result, per_sample)


def run_random_same_norm(
    *,
    samples: list[dict[str, Any]],
    model_b,
    tokenizer_b,
    cfg: dict[str, Any],
    layer_b: str,
    args: argparse.Namespace,
    direction: str,
    seed: int,
    clean_eval_hash: str,
    device,
) -> dict[str, Any]:
    gen_kwargs = _generation_kwargs(args.max_new_tokens)
    result = _base_result(
        condition="same_norm_random",
        direction=direction,
        seed=seed,
        deterministic=False,
        tier=args.tier,
        task=args.task,
        test_file=args.test_file,
        clean_eval_hash=clean_eval_hash,
        samples=samples,
        cfg={**cfg, "sender_id": "random_direction"},
        layer_a=None,
        layer_b=layer_b,
        device=device,
        max_length=args.max_length,
        layer_rel=args.layer_rel,
        gen_kwargs=gen_kwargs,
        injection={
            "mode": "replace_scale_corrected",
            "timing": "prefill_only",
            "source": "random_direction",
            "alpha": 1.0,
        },
    )
    per_sample = []
    t0 = time.time()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    for idx, sample in enumerate(samples):
        prompt, _sender_text, _question = _full_input_prompt(sample, args.task)
        receiver_hidden, prompt_len_b, input_ids_b = _extract_layer_hidden(
            model_b, tokenizer_b, prompt, layer_b, device, args.max_length
        )
        random_acts = torch.randn(
            tuple(receiver_hidden.shape),
            generator=generator,
            dtype=torch.float32,
        )
        prediction, diagnostics = _generate_with_layer_injection(
            model_b,
            tokenizer_b,
            prompt,
            layer_b,
            random_acts,
            injection_mode="replace_scale_corrected",
            device=device,
            max_length=args.max_length,
            gen_kwargs=gen_kwargs,
            injection_alpha=1.0,
            expected_input_ids=input_ids_b,
            strict_token_match=True,
        )
        per_sample.append(
            _record_prediction(
                sample,
                idx,
                args.task,
                prediction,
                {"prompt_len_b": prompt_len_b, "prompt_mode": "full_input", **diagnostics},
            )
        )
    result["elapsed_sec"] = round(time.time() - t0, 3)
    return _finish_result(result, per_sample)


def run_zero_replacement(
    *,
    samples: list[dict[str, Any]],
    model_b,
    tokenizer_b,
    cfg: dict[str, Any],
    layer_b: str,
    args: argparse.Namespace,
    direction: str,
    clean_eval_hash: str,
    device,
) -> dict[str, Any]:
    gen_kwargs = _generation_kwargs(args.max_new_tokens)
    result = _base_result(
        condition="zero_replacement",
        direction=direction,
        seed=None,
        deterministic=True,
        tier=args.tier,
        task=args.task,
        test_file=args.test_file,
        clean_eval_hash=clean_eval_hash,
        samples=samples,
        cfg={**cfg, "sender_id": "zero_vector"},
        layer_a=None,
        layer_b=layer_b,
        device=device,
        max_length=args.max_length,
        layer_rel=args.layer_rel,
        gen_kwargs=gen_kwargs,
        injection={"mode": "replace", "timing": "prefill_only", "source": "zero_vector"},
    )
    per_sample = []
    t0 = time.time()
    for idx, sample in enumerate(samples):
        prompt, _sender_text, _question = _full_input_prompt(sample, args.task)
        receiver_hidden, prompt_len_b, input_ids_b = _extract_layer_hidden(
            model_b, tokenizer_b, prompt, layer_b, device, args.max_length
        )
        zeros = torch.zeros_like(receiver_hidden)
        prediction, diagnostics = _generate_with_layer_injection(
            model_b,
            tokenizer_b,
            prompt,
            layer_b,
            zeros,
            injection_mode="replace",
            device=device,
            max_length=args.max_length,
            gen_kwargs=gen_kwargs,
            expected_input_ids=input_ids_b,
            strict_token_match=True,
        )
        per_sample.append(
            _record_prediction(
                sample,
                idx,
                args.task,
                prediction,
                {"prompt_len_b": prompt_len_b, "prompt_mode": "full_input", **diagnostics},
            )
        )
    result["elapsed_sec"] = round(time.time() - t0, 3)
    return _finish_result(result, per_sample)


def _expand_conditions(conditions: Sequence[str]) -> list[str]:
    expanded: list[str] = []
    for condition in conditions:
        if condition == "all":
            expanded.extend(_ALL_CONDITIONS)
        elif condition == "main":
            expanded.extend(_MAIN_CONDITIONS)
        elif condition == "controls":
            expanded.extend(_CONTROL_CONDITIONS)
        else:
            expanded.append(condition)
    deduped = []
    for condition in expanded:
        if condition not in deduped:
            deduped.append(condition)
    return deduped


def _run_direction(args: argparse.Namespace, direction: str, conditions: list[str]) -> list[Path]:
    device = _resolve_device(args.device)
    samples = _load_samples(args.test_file, args.max_samples)
    clean_eval_hash = _sha256_file(args.test_file)
    cfg = _direction_cfg(args.tier, direction)  # type: ignore[arg-type]
    layer_a, layer_b = _resolve_layers(cfg, args.layer_rel)
    logger.info(
        "Direction %s: %s -> %s  layers %s -> %s  n=%d",
        direction,
        cfg["sender_id"],
        cfg["receiver_id"],
        layer_a,
        layer_b,
        len(samples),
    )

    needs_sender = bool(set(conditions) & (_TRANSLATION_CONDITIONS | {"nl_relay"}))
    logger.info("Loading receiver model: %s", cfg["receiver_id"])
    model_b, tokenizer_b = load_model_and_tokenizer(cfg["receiver_id"], device)
    model_a = tokenizer_a = None
    if needs_sender:
        logger.info("Loading sender model: %s", cfg["sender_id"])
        model_a, tokenizer_a = load_model_and_tokenizer(cfg["sender_id"], device)

    written: list[Path] = []
    deterministic_done: set[str] = set()
    for condition in conditions:
        if condition in _DETERMINISTIC_CONDITIONS and condition in deterministic_done:
            continue
        if condition == "no_inject":
            result = run_no_inject(
                samples=samples,
                model_b=model_b,
                tokenizer_b=tokenizer_b,
                cfg=cfg,
                layer_b=layer_b,
                args=args,
                direction=direction,
                clean_eval_hash=clean_eval_hash,
                device=device,
            )
            written.append(_save_result(result, args.output_dir))
            deterministic_done.add(condition)
        elif condition == "nl_relay":
            if model_a is None or tokenizer_a is None:
                raise RuntimeError("nl_relay requires sender model")
            result = run_nl_relay(
                samples=samples,
                model_a=model_a,
                tokenizer_a=tokenizer_a,
                model_b=model_b,
                tokenizer_b=tokenizer_b,
                cfg=cfg,
                layer_a=layer_a,
                layer_b=layer_b,
                args=args,
                direction=direction,
                clean_eval_hash=clean_eval_hash,
                device=device,
            )
            written.append(_save_result(result, args.output_dir))
            deterministic_done.add(condition)
        elif condition in _TRANSLATION_CONDITIONS:
            if model_a is None or tokenizer_a is None:
                raise RuntimeError(f"{condition} requires sender model")
            for seed in args.seeds:
                result = run_translation_condition(
                    condition=condition,
                    direction=direction,
                    seed=seed,
                    samples=samples,
                    model_a=model_a,
                    tokenizer_a=tokenizer_a,
                    model_b=model_b,
                    tokenizer_b=tokenizer_b,
                    cfg=cfg,
                    layer_a=layer_a,
                    layer_b=layer_b,
                    args=args,
                    clean_eval_hash=clean_eval_hash,
                    device=device,
                )
                written.append(_save_result(result, args.output_dir))
        elif condition == "b_to_b_self_inject":
            result = run_self_inject(
                samples=samples,
                model_b=model_b,
                tokenizer_b=tokenizer_b,
                cfg=cfg,
                layer_b=layer_b,
                args=args,
                direction=direction,
                clean_eval_hash=clean_eval_hash,
                device=device,
            )
            written.append(_save_result(result, args.output_dir))
            deterministic_done.add(condition)
        elif condition == "same_norm_random":
            for seed in args.seeds:
                result = run_random_same_norm(
                    samples=samples,
                    model_b=model_b,
                    tokenizer_b=tokenizer_b,
                    cfg=cfg,
                    layer_b=layer_b,
                    args=args,
                    direction=direction,
                    seed=seed,
                    clean_eval_hash=clean_eval_hash,
                    device=device,
                )
                written.append(_save_result(result, args.output_dir))
        elif condition == "zero_replacement":
            result = run_zero_replacement(
                samples=samples,
                model_b=model_b,
                tokenizer_b=tokenizer_b,
                cfg=cfg,
                layer_b=layer_b,
                args=args,
                direction=direction,
                clean_eval_hash=clean_eval_hash,
                device=device,
            )
            written.append(_save_result(result, args.output_dir))
            deterministic_done.add(condition)
        else:
            raise ValueError(f"Unknown condition: {condition}")

    del model_b
    if model_a is not None:
        del model_a
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return written


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run final H1 clean-rerun conditions with per-sample outputs."
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["main"],
        choices=list(_ALL_CONDITIONS) + ["main", "controls", "all"],
        help="Conditions to run. Aliases: main, controls, all.",
    )
    parser.add_argument("--tier", default="tier2", choices=sorted(_TIERS.keys()))
    parser.add_argument(
        "--task", default=_TASK, choices=["multi_hop", "knowledge_relay", "instruction_following"]
    )
    parser.add_argument("--directions", nargs="+", default=["fwd"], choices=["fwd", "rev"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    parser.add_argument("--test-file", type=Path, default=_DEFAULT_TEST_FILE)
    parser.add_argument("--ckpt-dir", type=Path, default=_DEFAULT_CKPT_DIR)
    parser.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--layer-rel", type=float, default=_DEFAULT_RELATIVE_LAYER)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--sender-max-new-tokens", type=int, default=128)
    parser.add_argument("--injection-scale", type=float, default=0.01)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--best-alpha", type=float, default=0.30)
    parser.add_argument(
        "--shuffle-injection-mode",
        choices=["scale_corrected", "replace"],
        default="scale_corrected",
    )
    parser.add_argument(
        "--strict-token-match",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require identical A/B input token IDs for replacement-style conditions.",
    )
    parser.add_argument(
        "--strict-shuffle",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fail shuffled_translation if any same-length bucket has only one sample.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not args.test_file.exists():
        raise FileNotFoundError(f"Test file not found: {args.test_file}")
    conditions = _expand_conditions(args.conditions)
    logger.info("H1 final runner schema=%s conditions=%s", _SCHEMA_VERSION, conditions)
    written: list[Path] = []
    for direction in args.directions:
        written.extend(_run_direction(args, direction, conditions))
    logger.info("Done. Wrote %d result files.", len(written))
    for path in written:
        logger.info("  %s", path)


if __name__ == "__main__":
    main()
