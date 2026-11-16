from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LANDING = ROOT / "landing"


def test_landing_uses_measured_quality_claims_with_named_baselines():
    homepage = (LANDING / "index.html").read_text(encoding="utf-8").lower()
    benchmarks = (LANDING / "benchmarks.html").read_text(encoding="utf-8").lower()
    scripts = (LANDING / "assets" / "main.js").read_text(encoding="utf-8").lower()

    assert "bit-exact" not in homepage
    assert "bit-exact" not in scripts
    assert "sign accuracy" not in homepage
    assert "sign accuracy" not in benchmarks
    assert "signacc" not in scripts

    assert "perplexity change" in homepage
    assert "&lt; 0.8 pt vs fp16" in homepage
    assert "13.63 with rvq vs 13.57 with uncompressed fp16 k/v" in homepage
    assert "< 0.8 pt vs fp16" in scripts
    assert "< 0.4 pt vs fp16" in scripts
    assert "< 1.4 pt vs fp16" in scripts
    assert "full-attention reference" in benchmarks
    assert "22.77 with fp16" in benchmarks
    assert "81.30 with kivi 2-bit" in benchmarks
    assert "q-filters: 8.48 ppl; naive eviction: 13.36 ppl" in benchmarks
    assert "q-filters: 16.31 ppl; naive eviction: 23.65 ppl" in benchmarks
    assert "q-filters: 25.93 ppl; naive eviction: 31.27 ppl" in benchmarks
