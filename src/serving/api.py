"""FastAPI serving layer that productionizes the attrition model.

Exposes real-time single inference (pulls features from the online store) and
batch inference for downstream analytics consumption.
"""

from __future__ import annotations

from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.features.build_features import FEATURE_COLUMNS
from src.features.store import FeatureStore
from src.serving import model as model_mod

app = FastAPI(title="Workforce Attrition Serving API", version="0.1.0")
_store = FeatureStore()
_bundle = None


def _get_bundle():
    global _bundle
    if _bundle is None:
        _bundle = model_mod.load()
    return _bundle


class FeatureVector(BaseModel):
    tenure_bucket: int
    comp_ratio_z: float
    engagement_gap: float
    promo_velocity: float
    overtime_load: float
    mobility_flag: int


class BatchRequest(BaseModel):
    employee_ids: List[int]


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "features": FEATURE_COLUMNS}


@app.get("/predict/{employee_id}")
def predict_realtime(employee_id: int) -> dict:
    """Real-time inference: fetch online features by entity key, then score."""
    feats = _store.get_online_features(employee_id)
    if feats is None:
        raise HTTPException(status_code=404, detail="employee not in online store")
    risk = model_mod.predict_one(_get_bundle(), feats)
    return {"employee_id": employee_id, "attrition_risk": round(risk, 4)}


@app.post("/predict")
def predict_features(vec: FeatureVector) -> dict:
    """Real-time inference from an explicit feature vector."""
    risk = model_mod.predict_one(_get_bundle(), vec.model_dump())
    return {"attrition_risk": round(risk, 4)}


@app.post("/predict/batch")
def predict_batch(req: BatchRequest) -> dict:
    """Batch inference over a list of entity keys."""
    bundle = _get_bundle()
    out = []
    for eid in req.employee_ids:
        feats = _store.get_online_features(eid)
        if feats is None:
            out.append({"employee_id": eid, "attrition_risk": None})
        else:
            out.append({"employee_id": eid, "attrition_risk": round(model_mod.predict_one(bundle, feats), 4)})
    return {"results": out}
