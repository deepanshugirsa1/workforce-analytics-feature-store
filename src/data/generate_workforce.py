"""Synthetic, privacy-safe workforce data generator.

No real employee data is used. Records are fully synthetic and model
voice-of-employee, compensation, tenure, and mobility signals that a
people-science team would use for attrition-risk modeling.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ORGS = ["Benefits", "Compensation", "Recruiting", "PeopleTech", "Culture", "VoiceOfEmployee"]


def generate(n: int = 5000, seed: int = 7, as_of: str = "2026-06-01") -> pd.DataFrame:
    """Generate `n` synthetic employee-period records with an attrition label."""
    rng = np.random.default_rng(seed)

    tenure = rng.gamma(shape=2.0, scale=18.0, size=n).clip(0, 600)
    comp_ratio = rng.normal(1.0, 0.12, size=n).clip(0.5, 2.0)
    engagement = rng.normal(3.6, 0.8, size=n).clip(1.0, 5.0)
    manager_span = rng.integers(1, 15, size=n)
    promo = rng.poisson(0.4, size=n).clip(0, 5)
    transfers = rng.poisson(0.3, size=n).clip(0, 6)
    overtime = rng.gamma(2.0, 4.0, size=n).clip(0, 80)

    # Latent attrition propensity: low engagement, low comp, high overtime,
    # very short or stagnant tenure raise risk. Kept interpretable on purpose.
    logit = (
        -1.2
        + 0.9 * (engagement < 3.0)
        + 0.8 * (comp_ratio < 0.9)
        + 0.6 * (overtime > 30)
        + 0.5 * (tenure < 12)
        - 0.4 * (promo > 0)
        + 0.3 * (transfers >= 2)
    )
    prob = 1.0 / (1.0 + np.exp(-logit))
    attrition = (rng.random(n) < prob).astype(int)

    return pd.DataFrame(
        {
            "employee_id": np.arange(1, n + 1),
            "as_of_date": pd.Timestamp(as_of),
            "org": rng.choice(ORGS, size=n),
            "tenure_months": tenure.round(1),
            "comp_ratio": comp_ratio.round(3),
            "engagement_score": engagement.round(2),
            "manager_span": manager_span,
            "promo_count_24m": promo,
            "transfers_12m": transfers,
            "overtime_hours_4w": overtime.round(1),
            "attrition_180d": attrition,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default="data/generated/workforce.parquet")
    args = parser.parse_args()

    df = generate(args.rows, args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"Wrote {len(df)} synthetic workforce rows -> {out}")
    print(f"Attrition base rate: {df['attrition_180d'].mean():.1%}")


if __name__ == "__main__":
    main()
