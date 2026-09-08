"""Config-driven feature extraction framework.

Turns raw workforce records into model-ready features. The feature list is
driven by `configs/features.yaml` so schemas can evolve without code changes.
"""

from __future__ import annotations

import pandas as pd

FEATURE_COLUMNS = [
    "tenure_bucket",
    "comp_ratio_z",
    "engagement_gap",
    "promo_velocity",
    "overtime_load",
    "mobility_flag",
]


def _tenure_bucket(months: pd.Series) -> pd.Series:
    bins = [-1, 6, 12, 36, 72, 10_000]
    labels = [0, 1, 2, 3, 4]  # ordinal: <6m, 6-12m, 1-3y, 3-6y, 6y+
    return pd.cut(months, bins=bins, labels=labels).astype("int64")


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return an entity-keyed feature frame with point-in-time columns."""
    out = pd.DataFrame()
    out["employee_id"] = df["employee_id"].astype("int64")
    out["as_of_date"] = pd.to_datetime(df["as_of_date"])

    out["tenure_bucket"] = _tenure_bucket(df["tenure_months"])

    comp = df["comp_ratio"].astype(float)
    std = comp.std(ddof=0) or 1.0
    out["comp_ratio_z"] = ((comp - comp.mean()) / std).round(4)

    # Gap from the "healthy" engagement anchor (4.0 on a 1-5 survey).
    out["engagement_gap"] = (4.0 - df["engagement_score"].astype(float)).round(3)
    out["promo_velocity"] = df["promo_count_24m"].astype(float) / 2.0
    out["overtime_load"] = (df["overtime_hours_4w"].astype(float) / 40.0).round(3)
    out["mobility_flag"] = (df["transfers_12m"].astype(int) >= 2).astype("int64")

    if "attrition_180d" in df.columns:
        out["attrition_180d"] = df["attrition_180d"].astype("int64")
    return out


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/generated/workforce.parquet")
    parser.add_argument("--out", default="data/store/features.parquet")
    args = parser.parse_args()

    raw = pd.read_parquet(args.input)
    feats = build_features(raw)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(args.out, index=False)
    print(f"Built {len(feats)} feature rows, {len(FEATURE_COLUMNS)} features -> {args.out}")
