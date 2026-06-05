"""Reproducibility guard for the H1 clean evaluation builder.

Both the builder script (``papers/h1_activation_transfer/scripts/build_clean_eval.py``)
and the multi-hop task data (``data/tasks/multi_hop_reasoning/*.jsonl``) live
outside the A0 narrow-scope save point: scripts ship in the C-round paper-repo
commit, and the data tier is intentionally never committed. The test therefore
skips gracefully on a clean checkout / CI; it is intended as an integration-
style reproducibility check on a fully populated working tree.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

EXPECTED_CLEAN_HASH = "504e077cf17433e22967c86e98d321532d4e803dbe24d96af14c7e8ecdd0dcbb"
EXPECTED_CLEAN_N = 396


def _load_build_clean_eval_module():
    path = Path("papers/h1_activation_transfer/scripts/build_clean_eval.py")
    if not path.is_file():
        pytest.skip(
            f"H1 paper clean-eval builder not yet committed at {path}; "
            "runnable once papers/h1_activation_transfer/ ships in C-round."
        )
    spec = importlib.util.spec_from_file_location("h1_build_clean_eval", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_clean_eval_rebuilds_to_frozen_hash(tmp_path: Path):
    train_path = Path("data/tasks/multi_hop_reasoning/train.jsonl")
    eval_path = Path("data/tasks/multi_hop_reasoning/test_enhanced.jsonl")
    if not (train_path.is_file() and eval_path.is_file()):
        pytest.skip(
            "H1 multi-hop task data not present "
            f"({train_path}, {eval_path}); data/ tier is never committed, "
            "rerun on a fully populated working tree."
        )
    builder = _load_build_clean_eval_module()
    out_path = tmp_path / "clean_eval.jsonl"
    report_path = tmp_path / "leakage_report.json"
    manifest_path = tmp_path / "clean_eval_manifest.json"

    manifest = builder.build_clean_eval(
        train_path=train_path,
        eval_path=eval_path,
        out_path=out_path,
        report_path=report_path,
        manifest_path=manifest_path,
    )

    assert manifest["clean_summary"]["n_rows"] == EXPECTED_CLEAN_N
    assert manifest["acceptance"]["passed"] is True
    assert manifest["clean_eval_sha256"] == EXPECTED_CLEAN_HASH
    assert builder.file_sha256(out_path) == EXPECTED_CLEAN_HASH
