"""Unit tests for the Mac / RAM method recommender (no MLX required)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

# Repo-root tests/ so pytest does not import veloxquant_mlx/__init__.py (needs mlx).
_MOD_PATH = (
    Path(__file__).resolve().parents[2]
    / "veloxquant_mlx"
    / "tools"
    / "mac_recommender.py"
)
_SPEC = importlib.util.spec_from_file_location("mac_recommender", _MOD_PATH)
assert _SPEC and _SPEC.loader
_mod = importlib.util.module_from_spec(_SPEC)
sys.modules["mac_recommender"] = _mod
_SPEC.loader.exec_module(_mod)

RecommendRequest = _mod.RecommendRequest
estimate_kv_fp16_mb = _mod.estimate_kv_fp16_mb
recommend = _mod.recommend
ruleset_dict = _mod.ruleset_dict


def test_estimate_kv_fp16_mb_mistral_like():
    # 2 * 32 * 8 * 128 * 2048 * 2 bytes = 256 MiB
    mb = estimate_kv_fp16_mb(32, 8, 128, 2048)
    assert abs(mb - 256.0) < 0.5


def test_everyday_default_is_rvq():
    r = recommend(
        RecommendRequest(chip="M4", ram_gb=48, model_class="7B", goal="everyday")
    )
    assert r.method == "turboquant_rvq"
    assert r.knobs["bit_width_inlier"] == 1
    assert abs(r.key_accounting_ratio - 7.5) < 1e-6
    assert r.resident_savings_likely is False


def test_max_key_accounting_is_vecinfer():
    r = recommend(
        RecommendRequest(
            chip="M2", ram_gb=32, model_class="3B", goal="max_key_accounting"
        )
    )
    assert r.method == "vecinfer"
    assert abs(r.key_accounting_ratio - 16.0) < 1e-6


def test_constant_memory_is_eviction():
    r = recommend(
        RecommendRequest(
            chip="M1", ram_gb=8, model_class="3B", goal="constant_memory"
        )
    )
    assert r.method == "streaming_llm"
    assert r.resident_savings_likely is True


def test_tight_ram_warns_for_large_model():
    r = recommend(
        RecommendRequest(chip="M1", ram_gb=8, model_class="7B", goal="everyday")
    )
    assert any("headroom" in w.lower() or "tight" in w.lower() for w in r.warnings)


def test_32b_on_24gb_warns():
    r = recommend(
        RecommendRequest(chip="M4", ram_gb=24, model_class="32B", goal="everyday")
    )
    assert any("headroom" in w.lower() for w in r.warnings)


def test_invalid_ram_raises():
    with pytest.raises(ValueError):
        recommend(
            RecommendRequest(chip="M4", ram_gb=12, model_class="3B", goal="everyday")
        )


def test_ruleset_export():
    rs = ruleset_dict()
    assert rs["version"] == 1
    assert "everyday" in rs["defaults"]
    assert rs["defaults"]["everyday"]["method"] == "turboquant_rvq"
