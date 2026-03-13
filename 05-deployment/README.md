# 05 – Model Deployment

Deploy the VenturePulse startup traction model as a REST API using FastAPI + MLflow.

---
## What You Will Do
1. Start MLflow
2. Train the model (logs Random Forest artifact + creates `run_id.txt`)
3. Start the API service (`python app.py`)
4. Explore manually via Swagger (`/predict` in browser)
5. Test the API automatically (`pytest -q test_api.py`)

---
## Folder Structure
```
05-deployment/
├── train.py        # Train & log Random Forest artifact (creates run_id.txt)
├── app.py          # FastAPI service (loads model from MLflow via run_id.txt)
├── test_api.py     # Pytest to exercise /health and /predict endpoints
├── run_id.txt      # Generated MLflow run ID (after training)
└── README.md       # This guide
```

---
## 1. Start MLflow tracking server (New VSCode terminal)
Activate Environment
```powershell
.venv\Scripts\activate
```
You should see `(.venv)` at the start of the prompt.

Go to the folder:
```powershell
cd 05-deployment
```

And run:
```powershell
mlflow server --host 127.0.0.1 --port 5000
```
Open http://localhost:5000 (leave running).

---
## 2. Train the Model (New VSCode terminal)
```powershell
.venv\Scripts\activate
```
You should see `(.venv)` at the start of the prompt.

```powershell
cd 05-deployment
```

And run:
```powershell
python train.py
```
Example output:
```
=== VenturePulse Training ===
📥 Loading data...
✓ Loaded 30,000 rows | Success rate: 25.0%
✓ Train: 21,000 | Val: 4,500 | Test: 4,500
🔧 Preparing features...
✓ Features: 47
🚀 Training model...
✓ Precision@100: 38.0%  AUC: 0.705  F1: 0.378  Acc: 0.642
💾 Saved run_id.txt (run: <RUN_ID>)
🖥  View MLflow UI: http://localhost:5000
✅ Training complete. Next: python app.py
```
This writes `run_id.txt` and logs ONE artifact: `model` (the trained Random Forest).

---
## 3. Start the API (New VSCode terminal)
```powershell
.venv\Scripts\activate
cd 05-deployment
python app.py
```
Sample output:
```
[startup] Loaded model from run: <RUN_ID>
INFO:     Uvicorn running on http://0.0.0.0:9696 (Press CTRL+C to quit)
```
Keep this terminal open.

---
## 4. Test via Browser
Open http://localhost:9696/docs
1. Click `POST /predict` → Try it out
2. Use this payload:
```json
{
  "sector": "Fintech",
  "location": "NYC",
  "funding_stage": "Seed",
  "founder_background": "Technical",
  "team_diversity": "High",
  "initial_funding_amount": 500000,
  "team_size": 8,
  "months_since_founded": 12,
  "capital_efficiency": 0.65,
  "market_timing": 0.72,
  "pmf_score": 0.58,
  "cac_efficiency": 0.45,
  "burn_rate": 50000,
  "network_strength": 0.60,
  "founder_commitment": 0.85,
  "technical_moat": 0.55,
  "revenue_growth": 0.30,
  "has_customers": 1,
  "prev_experience": 0.70,
  "investor_quality": 0.65,
  "tier1_location": 1,
  "hot_sector": 1
}
```
3. Execute → Response shows `prediction`, `probability`, `confidence`, `flagged_for_review`, `model_version`.

Sample output:
```json
{
  "prediction": 1,
  "probability": 0.6421,
  "confidence": "High",
  "flagged_for_review": false,
  "model_version": "<RUN_ID>"
}
```

---
## 5. Automated API Test (New VSCode terminal)
```powershell
.venv\Scripts\activate
cd 05-deployment
pytest -q test_api.py
```
Expect `2 passed`.

---
## API Endpoints
| Endpoint | Purpose |
|----------|---------|
| /        | Welcome / model info |
| /health  | Service + model status |
| /predict | Make a prediction |
| /docs    | Swagger UI |

---
## Troubleshooting
| Issue | Fix |
|-------|-----|
| "run_id.txt not found" | Run `python train.py` first |
| MLflow connection error | Start server (step 1) |
| Module not found | Activate env: `.venv\Scripts\activate` |
| API not responding | Ensure Uvicorn running from `python app.py` |

---
## Behind the Scenes
- `train.py`: loads `../data/raw/venturepulse_dataset.csv` → stratified 70/15/15 split → one-hot encodes categoricals → trains Random Forest (n_estimators=100, max_depth=10, class_weight=balanced) → logs params, metrics (Precision@100, AUC, F1) and ONE artifact `model` → writes `run_id.txt`.
- `app.py`: on startup reads `run_id.txt` → loads model from MLflow → `/predict` builds feature DataFrame and returns prediction + probability + confidence + flagged_for_review flag.
- **Fairness note:** EdTech and Singapore predictions return `flagged_for_review: true` based on `03-fairness-analysis`.

---
## Quick Commands
| Action | Command |
|--------|---------|
| Activate env | `.venv\Scripts\activate` |
| Start MLflow | `mlflow server --host 127.0.0.1 --port 5000` |
| Train model | `python train.py` |
| Run API | `python app.py` |
| Test API (browser) | http://localhost:9696/docs |
| Test API (pytest) | `pytest -q test_api.py` |
