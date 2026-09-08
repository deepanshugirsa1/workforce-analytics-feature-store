"""Minimal offline + online feature store.

- Offline store: partitioned Parquet used for point-in-time training pulls.
- Online store: low-latency key-value snapshot (JSON here; DynamoDB/Redis in
  Future Scope) used by the real-time inference API.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .build_features import FEATURE_COLUMNS


class FeatureStore:
    def __init__(self, root: str = "data/store") -> None:
        self.root = Path(root)
        self.offline_path = self.root / "features.parquet"
        self.online_path = self.root / "online.json"

    # ---- offline (training) ----
    def materialize(self, features: pd.DataFrame) -> None:
        """Persist the offline table and refresh the online snapshot."""
        self.root.mkdir(parents=True, exist_ok=True)
        features.to_parquet(self.offline_path, index=False)
        latest = (
            features.sort_values("as_of_date")
            .groupby("employee_id")
            .tail(1)
            .set_index("employee_id")[FEATURE_COLUMNS]
        )
        self.online_path.write_text(json.dumps({str(k): v for k, v in latest.to_dict("index").items()}))

    def get_training_frame(self) -> pd.DataFrame:
        return pd.read_parquet(self.offline_path)

    # ---- online (serving) ----
    def get_online_features(self, employee_id: int) -> dict | None:
        if not self.online_path.exists():
            return None
        table = json.loads(self.online_path.read_text())
        return table.get(str(employee_id))
