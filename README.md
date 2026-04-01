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
- 💰 **Revenue ≥ $500k**
- 📊 **MRR ≥ $25k**

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

---

## 🤖 The Model

We evaluated three candidate models in a structured comparison:

| Model | Precision@100 | AUC-ROC | F1 |
| :--- | :---: | :---: | :---: |
| Logistic Regression (baseline) | 28% | 0.651 | 0.312 |
| Decision Tree | 31% | 0.668 | 0.341 |
| **Random Forest** ✅ | **38%** | **0.705** | **0.378** |

**Random Forest** was selected as the production model due to its superior
Precision@100 — the metric that directly maps to business value.

### Final Model Configuration

```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=50,
    min_samples_leaf=20,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
)
```

### Why Precision@100?

A VC analyst reviews roughly **100 deals per week**. Precision@100 measures
how many of those 100 model-selected startups are truly high-traction.

| Strategy | Precision@100 | vs. Random |
| :--- | :---: | :---: |
| Random selection | 25% | baseline |
| Heuristic rule (Series A + funding > $2M) | 30% | +20% |
| **VenturePulse RF** | **38%** | **+52%** |

---

## ⚖️ Fairness Analysis

We audited model performance across all sectors and geographies against a
**30% precision floor**. No segment should vary more than 15% from the mean.

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

**EdTech** and **Singapore** predictions are automatically flagged for human review
in the API response.

---

## 🏗️ Project Structure

```
mlops-venturepulse/
│
├── 01-initial-notebook/       EDA, feature distributions, LR baseline
├── 02-decision-tree-comparison/ LR vs DT vs RF — RF wins at P@100=38%
├── 03-fairness-analysis/      Precision@100 audit — EdTech & Singapore flagged
├── 04-experiment-tracking/    MLflow experiment tracking + model registry
├── 05-deployment/             FastAPI serving + MLflow loading (pytest: 2 passed)
├── 06-monitoring/             Evidently drift report via simulate.py + monitor.py
├── 07-cicd/                   Docker + GitHub Actions → GHCR → Render.com
│
├── docs/
│   └── index.html             Live demo (GitHub Pages)
│
├── data/raw/
│   └── venturepulse_dataset.csv (30,000 synthetic startups)
│
├── .github/workflows/
│   ├── ci-cd.yml              Train → Build → Test → Push to GHCR
│   └── train.yml              Reusable training workflow
│
└── startup_generator.py       Synthetic dataset generator
```

---

## 🔌 API Reference

Base URL: `https://mlops-venturepulse.onrender.com`

### `GET /health`

```json
{ "status": "ok", "run_id": "abc123..." }
```

### `POST /predict`

**Request:**
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

Confidence levels: `Very Low` (<15%) · `Low` (15–30%) · `Medium` (30–50%) · `High` (50–70%) · `Very High` (≥70%)

---

## 🚀 Running Locally

```powershell
git clone https://github.com/davidcarrilloag/mlops-venturepulse.git
cd mlops-venturepulse
python -m venv venturepulse_env
venturepulse_env\Scripts\activate
pip install -r 05-deployment/requirements.txt

python startup_generator.py
mlflow server --host 127.0.0.1 --port 5000   # Terminal 1
cd 05-deployment && python train.py            # Terminal 2
python app.py                                  # Terminal 3 → localhost:9696
pytest -q test_api.py                          # Terminal 4 → 2 passed
```

---

## 🔄 CI/CD Pipeline

Every push to `main` triggers:

```
git push → train.yml (train + upload artifact)
         → ci-cd.yml (build Docker → test → push to GHCR)
         → Render.com (pulls latest image → live)
```

---

## 📊 Monitoring

```powershell
cd 06-monitoring
python simulate.py   # → predictions.csv
python monitor.py    # → monitoring_report.html (Evidently drift report)
```

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
| Demo | GitHub Pages |
| Language | Python 3.12 |

---

*VenturePulse — Group 4, IE University MBDS, MLOps Final Project, March 2026*
