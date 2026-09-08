# Workforce Analytics Feature Store + Model Serving

An end-to-end **people-science data platform**: it turns workforce signals
(tenure, compensation, voice-of-employee engagement, mobility, overtime) into a
governed **feature store** and **productionizes an attrition-risk model** behind
a **serving API** with both **batch and real-time inference**.

Built to mirror how an interdisciplinary people-science team takes a model from
notebook to reliable production system — with data-quality gates, an
offline/online feature store, and confidential-by-design synthetic data.

> **Status: ~55% complete** — the full pipeline (generate → quality gate →
> features → store → train → serve) runs today. Streaming ingestion, DynamoDB
> online store, and Airflow orchestration are planned (see Future Scope).

## What works today (55%)

- **Config-driven feature framework** (`configs/features.yaml`) so schemas evolve without code changes
- **Offline + online feature store** with point-in-time training pulls and a low-latency serving snapshot
- **Data-quality & validation** gate: non-null, range, and PSI drift checks
- **Attrition-risk model** (gradient boosting) trained from the offline store with AUC reporting
- **FastAPI serving layer** — real-time single inference, feature-vector inference, and batch inference
- **Privacy-safe synthetic data** — no real employee records; confidential-by-design
- Unit tests for the feature and quality layers

## Quick start

```bash
pip install -r requirements.txt

# 1) run the end-to-end pipeline (generate -> quality -> features -> store -> train)
python -m src.pipelines.run_pipeline --rows 5000

# 2) serve the model (real-time + batch inference)
uvicorn src.serving.api:app --reload
#   GET  /health
#   GET  /predict/{employee_id}      -> real-time, features pulled from online store
#   POST /predict                    -> real-time from an explicit feature vector
#   POST /predict/batch              -> batch inference over many entity keys

pytest -q
```

## Architecture (current)

```
synthetic workforce data
        │
   data-quality gate (non-null, range, PSI drift)
        │
   config-driven feature framework
        │
   feature store ── offline (Parquet, point-in-time) ─► model training (GBM, AUC)
        └────────── online (KV snapshot) ─────────────► FastAPI serving
                                                          ├─ real-time inference
                                                          └─ batch inference
```

## Mapping to a people-science platform

| Platform need | Where it lives |
|---|---|
| Standardized metrics/features | `configs/features.yaml` + `src/features/build_features.py` |
| Model productionization & serving | `src/serving/model.py`, `src/serving/api.py` |
| Batch + real-time inference | `POST /predict/batch`, `GET /predict/{id}` |
| Data quality & validation | `src/quality/checks.py` |
| Confidential data handling | fully synthetic generator, no PII |

## Future scope (remaining ~45%)

| Area | Planned work |
|------|----------------|
| Ingestion | Kafka/Kinesis streaming of survey + HRIS events into the store |
| Online store | DynamoDB / Redis backend with TTL and versioned feature groups |
| Orchestration | Airflow DAGs for scheduled materialization + retraining |
| Monitoring | Model + feature drift dashboards, alerting, SLA tracking |
| Governance | Fine-grained access policies, audit logging, lineage catalog |
| Fairness | Subgroup calibration and bias checks before any decision surfacing |

## Stack

Python, pandas, scikit-learn, FastAPI, PyArrow, PyYAML; Kafka/Kinesis, DynamoDB, Airflow (planned)

## License

MIT
