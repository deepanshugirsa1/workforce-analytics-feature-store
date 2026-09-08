"""Data-quality and validation checks with a config-driven, evolvable schema."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """Population Stability Index for numeric drift detection."""
    quantiles = np.linspace(0, 1, bins + 1)
    cuts = np.unique(expected.quantile(quantiles).values)
    if len(cuts) < 3:
        return 0.0
    e = np.histogram(expected, bins=cuts)[0] / max(len(expected), 1)
    a = np.histogram(actual, bins=cuts)[0] / max(len(actual), 1)
    e = np.clip(e, 1e-6, None)
    a = np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def run_quality_checks(df: pd.DataFrame, cfg: dict, baseline: pd.DataFrame | None = None) -> dict:
    """Return a structured report; `passed=False` if any hard check fails."""
    q = cfg.get("quality", {})
    report: dict = {"passed": True, "checks": []}

    def record(name: str, ok: bool, detail: str) -> None:
        report["checks"].append({"check": name, "ok": bool(ok), "detail": detail})
        if not ok:
            report["passed"] = False

    for col in q.get("non_null", []):
        if col not in df.columns:
            record(f"non_null:{col}", False, "column missing")
            continue
        nulls = int(df[col].isna().sum())
        record(f"non_null:{col}", nulls == 0, f"{nulls} nulls")

    for col, (lo, hi) in q.get("ranges", {}).items():
        if col not in df.columns:
            record(f"range:{col}", False, "column missing")
            continue
        bad = int(((df[col] < lo) | (df[col] > hi)).sum())
        record(f"range:{col}", bad == 0, f"{bad} out-of-range")

    if baseline is not None:
        threshold = q.get("drift_psi_threshold", 0.2)
        for col in df.select_dtypes("number").columns:
            if col in baseline.columns:
                psi = _psi(baseline[col].dropna(), df[col].dropna())
                record(f"drift:{col}", psi <= threshold, f"psi={psi:.3f}")

    return report


if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/generated/workforce.parquet")
    parser.add_argument("--config", default="configs/features.yaml")
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    rep = run_quality_checks(df, cfg)
    print(f"quality passed={rep['passed']}")
    for c in rep["checks"]:
        print(f"  [{'ok' if c['ok'] else 'FAIL'}] {c['check']}: {c['detail']}")
