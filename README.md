# AeroPredict KLIA

**AeroPredict KLIA** is a machine learning-enabled application that predicts the probability that a **KLIA (Kuala Lumpur International Airport) flight departure will be delayed by 15 minutes or more**. The model automatically trains incrementally every week.

---

## For Travellers

**[→ Open the app](https://klia-flight-delay-predictor.streamlit.app/)**

Select your airline, destination, and departure time to get an instant delay risk estimate.

*Note: web app might take a while to start up as I'm using free tier web service*

---

## For Developers

### How It Works

```
Neon Postgres (postgres database)
    ↓  weekly via GitHub Actions
Incremental model training
    ↓  bundle stored in Neon model_registry
FastAPI on Render  ←→  Streamlit Cloud
```

- **Data:** Raw flight departure records stored in Neon Postgres (`departures` table)
- **Model:** XGBoost ensemble (`rsub_xgb`); trained incrementally with each weekly run calling `partial_fit` on new rows only; hyperparameters tuned offline with Optuna
- **Features:** 21 engineered features (cascade delay rates, smoothed airline/route rates, temporal cyclics, congestion); top_k selected per bootstrap probe (default: 20, tunable via Optuna)
- **Threshold:** Decision threshold tuned offline by Optuna (F1-optimised), stored statically in `config.yaml`; applied unchanged during production update runs
- **Serving:** FastAPI on Render loads the latest model bundle from Neon on startup; hot-swaps every 30 minutes without restart
- **Frontend:** Streamlit Cloud: thin UI only, calls the API

---

### API Reference

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| GET | `/healthz` | None | Health check; returns model version |
| POST | `/v1/public/predict` | None | Delay probability — **no API key required**, rate-limited |
| GET | `/v1/options` | ✅ | Available airlines and destinations |
| POST | `/v1/predict` | ✅ | Delay probability for a flight |
| GET | `/v1/model` | ✅ | Model metadata and feature info |
| GET | `/v1/drift` | ✅ | Evidently drift monitoring summary |

Authenticated endpoints require the header:
```
X-API-Key: your-api-key
```

---

#### Public Endpoint (no API key)

**POST `/v1/public/predict`** (rate-limited)

**Request body:**
```json
{
  "airline": "AirAsia",
  "destination": "Singapore",
  "scheduled_departure": "2026-10-10T08:30",
  "aircraft": "A320"
}
```

`aircraft` is optional.

**Response:**
```json
{
  "delay_probability": 0.7123,
  "predicted_delayed": true,
  "threshold": 0.61,
  "risk": "high",
  "model_version": 12,
  "known_airline": true,
  "known_route": true
}
```

`risk` is one of `low`, `elevated`, or `high`. `known_airline` / `known_route` flag whether the model has seen this airline or route before — if false, the estimate falls back to the base delay rate and is less reliable.

**curl:**
```bash
curl -X POST https://klia-flight-delay-predictor.onrender.com/v1/public/predict \
  -H "Content-Type: application/json" \
  -d '{
    "airline": "AirAsia",
    "destination": "Singapore",
    "scheduled_departure": "2026-10-10T08:30"
  }'
```

**Python:**
```python
import requests

response = requests.post(
    "https://klia-flight-delay-predictor.onrender.com/v1/public/predict",
    json={
        "airline": "AirAsia",
        "destination": "Singapore",
        "scheduled_departure": "2026-10-10T08:30",
        "aircraft": "A320",        # optional
    },
)
print(response.json())
# {'delay_probability': 0.7123, 'predicted_delayed': True, 'risk': 'high', ...}
```

---

### Local Setup

**Prerequisites:** Python 3.11+, a [Neon](https://neon.tech) Postgres database.

```bash
# 1. Clone
git clone https://github.com/realnoobestnoob/KLIA-Flight-Delay-Predictor
cd KLIA-Flight-Delay-Predictor

# 2. Install training dependencies
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-train.txt

# 3. Set environment variables
cp .env.example .env
# Edit .env — add DATABASE_URL and optionally RAPIDAPI_KEY

# 4. Verify connectivity
python -m klia.jobs.check

# 5. Bootstrap (first run only — probes features, replays all history)
python -m klia.jobs.update --bootstrap

# 6. Start API
uvicorn klia.api.app:app --reload

# 7. Start Streamlit (separate terminal)
streamlit run streamlit_app/app.py
```

**Subsequent runs** (after bootstrap):
```bash
python -m klia.jobs.update   # trains on new rows only; skips if no new data
```

---

### Environment Variables

| Variable | Where | Required | Description |
|----------|-------|----------|-------------|
| `DATABASE_URL` | API + update job | ✅ | Neon **direct** (non-pooled) connection string |
| `KLIA_API_KEY` | API + Streamlit | Recommended | Shared secret for API auth (`X-API-Key` header) |
| `KLIA_API_URL` | Streamlit only | ✅ | FastAPI base URL |

---

### Docker (API only)

```bash
docker build -t klia-api .
docker run -p 8000:8000 -e DATABASE_URL=... -e KLIA_API_KEY=... klia-api
```

The `Dockerfile` and `render.yaml` are configured for Render deployment out of the box.

---

### Automated Training (GitHub Actions)

The workflow `.github/workflows/update.yml` runs every Monday at 04:30 MYT.

**Setup:**
1. Go to your repo → Settings → Secrets → Actions
2. Add secret: `DATABASE_URL` = your Neon direct connection string
3. Enable the workflow under Actions → "Weekly model update"

The job skips training automatically if no new rows are detected beyond the watermark. Logs are written to `etl_runs` in Neon.

---

### Key CLI Flags

```bash
# Tune hyperparameters (XGBoost params, top_k, decision threshold) with Optuna
python -m klia.jobs.tune                        # fetch from Neon
python -m klia.jobs.tune --trials 50            # override trial count
python -m klia.jobs.tune --sample 20000         # override sample row count
python -m klia.jobs.tune --csv data/departures.csv

# Full retrain from scratch (also re-runs feature selection probe)
python -m klia.jobs.update --bootstrap

# Process without saving (useful for debugging)
python -m klia.jobs.update --dry-run

# Limit rows processed (quick smoke test)
python -m klia.jobs.update --max-rows 1000

# Run against a local CSV instead of Neon
python -m klia.jobs.update --csv data/departures.csv
```

> Run `--bootstrap` whenever you add or remove features in `klia/features/state.py`, switch model type, or apply new hyperparameters from `tune.py`. Normal incremental runs will silently ignore new features until bootstrap is re-run. After tuning, uncomment `decision_threshold` in `config.yaml` before running `--bootstrap`.

---

### Project Structure

```
klia/
├── api/          # FastAPI app, bundle cache, and public router (public.py)
├── etl/          # Row validation and time parsing
├── features/     # FeatureState and FeatureSelector
├── jobs/         # update.py (training entry point), tune.py (Optuna tuning), check.py (connectivity)
├── model/        # Bundle, incremental model, MLflow wrapper
├── monitoring/   # Evidently drift detection
├── store/        # PostgresStore (Neon) and FileStore (local/CSV)
└── data/         # Dataset (not latest)
streamlit_app/    # Streamlit frontend (UI only)
config/           # config.yaml (all tunables, no secrets)
.github/workflows # update.yml (weekly training), keep_alive.yml (Streamlit ping)
artifacts/        # pretrained_bundle.gz (bootstrap output); delete before fresh retrain
notebooks/        # 01_model_selection_eda.ipynb
tests/            # test_api, test_features, test_update_job, test_validate
```

---

### Troubleshooting

| Problem | Fix |
|---------|-----|
| `503 no model available` | Run `python -m klia.jobs.update --bootstrap` first |
| `429 rate limit exceeded` on `/v1/public/predict` | Max 30 requests/min per IP; use the authenticated `/v1/predict` for higher volume |
| `401 invalid or missing X-API-Key` | Set `KLIA_API_KEY` in Render env and Streamlit secrets |
| `another update job is already running` | Use Neon **direct** URL, not pooled; delete stale advisory lock if job crashed |
| New features not taking effect | Re-run `--bootstrap`; `feature_names` locks on first `partial_fit` |
| Bundle size error on Neon | Reduce ensemble size in `config.yaml`; monitor `bundle_mb` in run logs |
| Streamlit airline dropdown missing entries | Expected — dropdown deduplicates bracket suffixes client-side; model uses full name internally |
