from src.data.generate_workforce import generate
from src.features.build_features import FEATURE_COLUMNS, build_features


def test_build_features_shape_and_columns():
    raw = generate(200, seed=1)
    feats = build_features(raw)
    assert len(feats) == 200
    for col in FEATURE_COLUMNS:
        assert col in feats.columns
    assert "attrition_180d" in feats.columns


def test_tenure_bucket_is_ordinal_range():
    raw = generate(200, seed=2)
    feats = build_features(raw)
    assert feats["tenure_bucket"].between(0, 4).all()


def test_mobility_flag_is_binary():
    raw = generate(200, seed=3)
    feats = build_features(raw)
    assert set(feats["mobility_flag"].unique()).issubset({0, 1})
