import yaml

from src.data.generate_workforce import generate
from src.quality.checks import run_quality_checks

CFG = yaml.safe_load(open("configs/features.yaml", encoding="utf-8"))


def test_clean_data_passes():
    df = generate(300, seed=4)
    report = run_quality_checks(df, CFG)
    assert report["passed"] is True


def test_out_of_range_is_flagged():
    df = generate(300, seed=5)
    df.loc[0, "engagement_score"] = 9.0  # invalid (>5)
    report = run_quality_checks(df, CFG)
    assert report["passed"] is False
    assert any(c["check"] == "range:engagement_score" and not c["ok"] for c in report["checks"])
