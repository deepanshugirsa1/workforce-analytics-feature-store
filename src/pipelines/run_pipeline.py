"""End-to-end pipeline: generate -> quality -> features -> store -> train.

Mirrors how a people-science model gets productionized: validated raw data,
a config-driven feature layer, an offline/online feature store, and a trained
model artifact ready for the serving API.
"""

from __future__ import annotations

import argparse

import yaml

from src.data.generate_workforce import generate
from src.features.build_features import build_features
from src.features.store import FeatureStore
from src.quality.checks import run_quality_checks
from src.serving.model import train


def run(rows: int, config: str) -> dict:
    cfg = yaml.safe_load(open(config, encoding="utf-8"))

    raw = generate(rows)
    quality = run_quality_checks(raw, cfg)
    if not quality["passed"]:
        raise SystemExit(f"data quality gate failed: {quality}")

    feats = build_features(raw)
    store = FeatureStore()
    store.materialize(feats)

    metrics = train(store.get_training_frame())
    return {"rows": rows, "quality_passed": quality["passed"], "model": metrics}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=5000)
    parser.add_argument("--config", default="configs/features.yaml")
    args = parser.parse_args()
    result = run(args.rows, args.config)
    print("pipeline complete:")
    print(f"  rows            : {result['rows']}")
    print(f"  quality passed  : {result['quality_passed']}")
    print(f"  model AUC       : {result['model']['auc']}")
    print("  online store + model artifact ready for serving API")


if __name__ == "__main__":
    main()
