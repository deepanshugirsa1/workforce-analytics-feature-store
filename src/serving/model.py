"""Attrition-risk model: train from the offline store, persist, and score."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd  # noqa: F401  (used for typing / named-frame scoring)
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src.features.build_features import FEATURE_COLUMNS

MODEL_PATH = Path("artifacts/attrition_model.joblib")
LABEL = "attrition_180d"


def train(features: pd.DataFrame, model_path: Path = MODEL_PATH) -> dict:
    X = features[FEATURE_COLUMNS]
    y = features[LABEL]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=7, stratify=y)
    clf = GradientBoostingClassifier(random_state=7)
    clf.fit(X_tr, y_tr)
    auc = float(roc_auc_score(y_te, clf.predict_proba(X_te)[:, 1]))
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": clf, "features": FEATURE_COLUMNS}, model_path)
    return {"auc": round(auc, 4), "n_train": len(X_tr), "n_test": len(X_te)}


def load(model_path: Path = MODEL_PATH):
    if not model_path.exists():
        raise FileNotFoundError(f"model not found at {model_path}; run the pipeline first")
    return joblib.load(model_path)


def predict_one(bundle: dict, feats: dict) -> float:
    cols = bundle["features"]
    row = pd.DataFrame([{c: feats[c] for c in cols}], columns=cols)
    return float(bundle["model"].predict_proba(row)[0, 1])


if __name__ == "__main__":
    from src.features.store import FeatureStore

    fs = FeatureStore()
    metrics = train(fs.get_training_frame())
    print(f"trained attrition model: {metrics}")
