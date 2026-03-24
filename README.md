# 🚀 VenturePulse — Startup Traction Prediction Engine

> **IE University · MBDS · MLOps Final Project · Group 4 · March 2026**

VenturePulse is a production-grade, end-to-end MLOps system that predicts whether
a startup will achieve **high traction within 18 months** of pitch stage. It is
designed to help venture capital analysts cut through hundreds of weekly deal-flow
submissions and focus their attention on the startups most likely to succeed.

---

## 🌐 Live Demo

**[→ Try the VenturePulse Predictor](https://davidcarrilloag.github.io/mlops-venturepulse)**

Enter a startup's pitch-stage data and get an instant ML-powered traction probability.
The demo calls the live production API deployed on Render.com.

**Live API:** `https://mlops-venturepulse.onrender.com`
- `GET /health` — model status
- `POST /predict` — traction prediction
- `GET /docs` — Swagger UI

> ⚠️ The free Render instance may take ~30 seconds to wake up after inactivity.

---

## 👥 Team

| Name | Role |
| :--- | :--- |
| Bojana Belincevic | Data Lead — dataset generation, quality checks, splits |
| David Carrillo | Modeling Lead — training, evaluation, MLflow |
| Sebastião Clemente | Feature Engineering — transformations, domain features, pipeline |
| Bassem El Halawani | Modeling Lead — hyperparameter tuning, fairness evaluation |
| Theo Henry | Deployment — FastAPI, Docker, integration |
| Ocke Moulijn | Coordination — project management, documentation, final submission |

---

## 🎯 The Business Problem

Venture capital is a high-volume, high-stakes industry. A typical VC firm receives
**1,000+ pitch decks per year** but invests in fewer than 1% of them. Despite this,
**75% of VC-backed startups still fail to return capital** — meaning even the
selected deals are often wrong. The core challenge is that current evaluation is
mostly qualitative, network-dependent, and impossible to scale.

On the founder side, **82% of deals flow through warm introductions**, systematically
excluding qualified entrepreneurs without elite network access. A typical analyst
spends ~15 hours per week manually screening deal flow — VenturePulse reduces this
to ~3 hours.

VenturePulse addresses this by bringing a **data-driven, repeatable scoring system**
to the top of the funnel. Given only the information available at pitch stage, the
model assigns each startup a probability of achieving high traction — defined as
meeting all three of the following criteria at t=18 months:

- 📈 **Team growth ≥ 25%**
- 💰 **Revenue ≥ $500K**
- 📊 **MRR ≥ $25K**

This turns a subjective, unscalable process into a ranked weekly shortlist that
analysts can act on immediately.

---

## 📊 The Dataset

| Property | Value |
| :--- | :--- |
| Records | 30,000 synthetic startups |
| Positive rate | ~25% (`high_traction = 1`) |
| Class imbalance | 3:1 negative-to-positive |
| Features | 23 total (12 original + 11 engineered) |
| Split | 70% train / 15% val / 15% test (stratified) |
| Random seed | 42 |
| Generator | `startup_generator.py` (Faker, NumPy, Pandas) |

All features are available at pitch stage to prevent data leakage. Categorical
variables (sector, location, funding stage, founder background, team diversity)
are one-hot encoded with `drop_first=False` to preserve full interpretability.

Distributions are calibrated against NVCA 2024 and PitchBook 2024 benchmarks:
30% Pre-seed, 50% Seed, 20% Series A; no sector exceeds 25% of records.

---

## 🤖 The Model

We evaluated three candidate models in a structured comparison:

| Model | Precision@100 | AUC-ROC | F1 |
| :--- | :---: | :---: | :---: |
| Logistic Regression (baseline) | 28% | 0.651 | 0.312 |
| Decision Tree | 31% | 0.668 | 0.341 |
| **Random Forest** ✅ | **38%** | **0.705** | **0.378** |

**Random Forest** was selected as the production model due to its superior
Precision@100 — the metric that directly maps to business value (how many of
the top 100 recommended startups are genuinely high-traction).

### Final Model Configuration

```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=50,
    min_samples_leaf=20,
    max_features="sqrt",
    class_weight="balanced",   # handles 3:1 class imbalance
    random_state=42,
)
```

### Why Precision@100?

A VC analyst reviews roughly **100 deals per week**. Precision@100 measures
how many of those 100 model-selected startups are truly high-traction — exactly
matching the real operational constraint.

| Strategy | Precision@100 | vs. Random |
| :--- | :---: | :---: |
| Random selection | 25% | baseline |
| Heuristic rule (Series A + funding > $2M) | 30% | +20% |
| **VenturePulse RF** | **38%** | **+52%** |

This means analysts spend their time on **13 more genuine opportunities per week**
compared to random selection, and beat the rule-based heuristic by 8 percentage points.

---

## ⚖️ Fairness Analysis

We audited model performance across all sectors and geographies against a
**30% precision floor** — the minimum acceptable performance per segment,
chosen to match the heuristic baseline. No segment should vary more than 15%
from the mean.

### By Sector

| Sector | Precision@100 | Status |
| :--- | :---: | :---: |
| AI/ML | 36% | ✅ Pass |
| Biotech | 38% | ✅ Pass |
| Climate | 37% | ✅ Pass |
| Consumer | 33% | ✅ Pass |
| **EdTech** | **28%** | ⚠️ Flagged |
| Fintech | 40% | ✅ Pass |
| Hardware | 35% | ✅ Pass |
| Healthcare | 36% | ✅ Pass |
| SaaS | 39% | ✅ Pass |

### By Location

| Location | Precision@100 | Status |
| :--- | :---: | :---: |
| Austin | 35% | ✅ Pass |
| Berlin | 34% | ✅ Pass |
| Boston | 37% | ✅ Pass |
| London | 38% | ✅ Pass |
| NYC | 36% | ✅ Pass |
| Remote | 33% | ✅ Pass |
| Silicon Valley | 39% | ✅ Pass |
| **Singapore** | **28%** | ⚠️ Flagged |
| Tel Aviv | 37% | ✅ Pass |
| Toronto | 35% | ✅ Pass |

### Deployment Decision

Both flagged segments fall only **2 percentage points** below the floor, with
variance across all segments well within acceptable range (CoV < 15%). The model
was **approved for deployment** with the following mitigation:

- All predictions for **EdTech** and **Singapore** startups return
  `"flagged_for_review": true` in the API response
- These cases are automatically routed to a senior analyst for human review
- Segment performance is monitored continuously via the monitoring pipeline

---

## 🏗️ Project Structure

Each folder mirrors the MLOps course module structure, with a dedicated README
explaining the decisions made at each stage.

```
mlops-venturepulse/
│
├── 01-initial-notebook/
│   └── EDA, feature distributions, LR baseline, class imbalance analysis
│
├── 02-decision-tree-comparison/
│   └── Side-by-side comparison of LR, Decision Tree, and Random Forest
│       Random Forest selected as winner (Precision@100 = 38%)
│
├── 03-fairness-analysis/
│   └── Precision@100 audit by sector and location
│       EdTech and Singapore flagged, deployment decision documented
│
├── 04-experiment-tracking/
│   └── MLflow experiment tracking — logs params, metrics, and model artifacts
│       Experiment: venturepulse-startup-prediction
│
├── 05-deployment/
│   └── Production FastAPI service
│       train.py → MLflow → run_id.txt → app.py → /predict endpoint
│       pytest: 2 tests passing
│
├── 06-monitoring/
│   └── simulate.py sends real data to /predict → predictions.csv
│       monitor.py generates Evidently HTML drift report + fairness check
│
├── 07-cicd/
│   └── Docker image with model baked in
│       GitHub Actions: Train → Lint → Build → Test → Push to GHCR
│       Render.com pulls image from GHCR for live deployment
│
├── data/raw/
│   └── venturepulse_dataset.csv (30,000 synthetic startups)
│
├── .github/workflows/
│   ├── ci-cd.yml    Main pipeline orchestrator
│   └── train.yml    Reusable training workflow
│
├── startup_generator.py    Synthetic dataset generator
├── requirements.txt
├── .flake8                 Linter config (max-line-length=88)
└── README.md               This file
```

---

## 🔌 API Reference

The prediction service runs on port `9696`.

### `GET /health`

```json
{
  "status": "ok",
  "run_id": "abc123...",
  "model_loaded": true
}
```

### `POST /predict`

**Request body:**

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

**Response:**

```json
{
  "prediction": 1,
  "probability": 0.7823,
  "confidence": "Very High",
  "flagged_for_review": false,
  "model_version": "abc123..."
}
```

**Confidence levels:** `Very Low` (<15%) · `Low` (15–30%) · `Medium` (30–50%) ·
`High` (50–70%) · `Very High` (≥70%)

**`flagged_for_review: true`** is returned automatically for EdTech and Singapore
startups, routing them to human review.

---

## 🚀 Running Locally

```powershell
# 1. Clone and activate environment
git clone https://github.com/davidcarrilloag/mlops-venturepulse.git
cd mlops-venturepulse
python -m venv venturepulse_env
venturepulse_env\Scripts\activate
pip install -r 05-deployment/requirements.txt

# 2. Generate the dataset
python startup_generator.py
# → data/raw/venturepulse_dataset.csv (30,000 records)

# 3. Start MLflow tracking server (Terminal 1)
mlflow server --host 127.0.0.1 --port 5000

# 4. Train the model (Terminal 2)
cd 05-deployment
python train.py
# Outputs: run_id.txt, logs model to MLflow at http://localhost:5000

# 5. Start the prediction API (Terminal 3)
python app.py
# API live at http://localhost:9696
# Swagger UI at http://localhost:9696/docs

# 6. Run tests (Terminal 4)
pytest -q test_api.py
# Expected: 2 passed
```

---

## 🔄 CI/CD Pipeline

Every push to `main` triggers the full pipeline:

```
Git Push to main
       │
       ▼
  train.yml (reusable)
  └── Trains model, uploads models/ + run_id.txt as GitHub artifact
       │
       ▼
  ci-cd.yml
  ├── 1. Train    — calls train.yml
  ├── 2. Lint     — flake8 07-cicd (0 errors required)
  ├── 3. Build    — Docker image with model baked in, runs test_api.py
  └── 4. Push     — ghcr.io/davidcarrilloag/mlops-venturepulse:latest
       │
       ▼
  Render.com
  └── Pulls image from GHCR → live web service
```

---

## 📊 Monitoring

```powershell
cd 06-monitoring

# Send 100 real startups to the API and log results
python simulate.py
# → data/predictions.csv

# Generate drift + fairness report
python monitor.py
# → monitoring_report.html
```

The monitoring report includes:

- **Data drift detection** comparing reference vs current prediction windows
- **Prediction distribution** shift over time
- **Segment-level fairness** check against the 30% precision floor

A drift score above 0.2 (PSI) triggers an alert recommending model retraining.

---

## 🛠️ Tech Stack

| Layer | Technology |
| :--- | :--- |
| ML | scikit-learn, Random Forest |
| Experiment tracking | MLflow |
| API | FastAPI, Uvicorn, Pydantic |
| Monitoring | Evidently |
| Testing | pytest, requests |
| Containerisation | Docker |
| CI/CD | GitHub Actions |
| Registry | GitHub Container Registry (GHCR) |
| Deployment | Render.com |
| Language | Python 3.12 |

---

*VenturePulse — Group 4, IE University MBDS, MLOps Final Project, March 2026*
