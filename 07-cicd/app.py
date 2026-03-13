"""FastAPI service for VenturePulse startup traction prediction.

Mirrors professor's 04-deployment/app.py pattern:
- Reads run_id.txt at startup
- Loads model artifact from MLflow using that run_id
- Exposes /health and /predict
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Optional

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
RUN_ID: Optional[str] = None
model = None

# ──────────────────────────────────────────────
# Categorical vocabularies
# Must match training notebooks one-hot encoding exactly
# ──────────────────────────────────────────────
SECTORS = [
    "AI/ML", "Biotech", "Climate", "Consumer", "EdTech",
    "Fintech", "Hardware", "Healthcare", "SaaS",
]
LOCATIONS = [
    "Austin", "Berlin", "Boston", "London", "NYC",
    "Remote", "Silicon Valley", "Singapore", "Tel Aviv", "Toronto",
]
FUNDING_STAGES = ["Pre-seed", "Seed", "Series A"]
FOUNDER_BACKGROUNDS = ["Technical", "Business", "Mixed", "Academic"]
TEAM_DIVERSITIES = ["Low", "Medium", "High"]

# Segments flagged for human review (from 03-fairness-analysis)
FLAGGED_SECTORS = {"EdTech"}
FLAGGED_LOCATIONS = {"Singapore"}


# ──────────────────────────────────────────────
# Feature engineering
# ──────────────────────────────────────────────
def prepare_features(data: dict) -> pd.DataFrame:
    """Convert startup input dict into the feature DataFrame
    expected by the model. Replicates notebook one-hot encoding
    exactly (drop_first=False, same column order).
    """
    row: dict = {}

    numerical_cols = [
        "initial_funding_amount", "team_size", "months_since_founded",
        "capital_efficiency", "market_timing", "pmf_score",
        "cac_efficiency", "burn_rate", "network_strength",
        "founder_commitment", "technical_moat", "revenue_growth",
        "has_customers", "prev_experience", "investor_quality",
        "tier1_location", "hot_sector",
    ]
    for col in numerical_cols:
        row[col] = data.get(col, 0)

    for s in SECTORS:
        row[f"sector_{s}"] = 1 if data.get("sector") == s else 0

    for loc in LOCATIONS:
        row[f"location_{loc}"] = 1 if data.get("location") == loc else 0

    for fs in FUNDING_STAGES:
        col_name = f"funding_stage_{fs.replace('-', '_').replace(' ', '_')}"
        row[col_name] = 1 if data.get("funding_stage") == fs else 0

    for fb in FOUNDER_BACKGROUNDS:
        row[f"founder_background_{fb}"] = 1 if data.get("founder_background") == fb else 0

    for td in TEAM_DIVERSITIES:
        row[f"team_diversity_{td}"] = 1 if data.get("team_diversity") == td else 0

    df = pd.DataFrame([row])

    if model is not None:
        try:
            expected_cols = model._model_impl.sklearn_model.feature_names_in_
            df = df.reindex(columns=expected_cols, fill_value=0)
        except AttributeError:
            pass

    return df


def probability_to_confidence(prob: float) -> str:
    if prob >= 0.70:
        return "Very High"
    if prob >= 0.50:
        return "High"
    if prob >= 0.30:
        return "Medium"
    if prob >= 0.15:
        return "Low"
    return "Very Low"


# ──────────────────────────────────────────────
# Lifespan — mirrors professor pattern exactly
# ──────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global RUN_ID, model

    with open("run_id.txt", "r") as f:
        RUN_ID = f.read().strip()

    model = mlflow.pyfunc.load_model("models/model")
    print(f"[startup] Loaded model from models/model")
    yield


# ──────────────────────────────────────────────
# App
# ──────────────────────────────────────────────
app = FastAPI(
    title="VenturePulse API",
    description="Predicts startup high-traction probability at t=18 months.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────
# Pydantic schemas
# ──────────────────────────────────────────────
class StartupRequest(BaseModel):
    sector: str = Field(..., example="Fintech")
    location: str = Field(..., example="NYC")
    funding_stage: str = Field(..., example="Seed")
    founder_background: str = Field(..., example="Technical")
    team_diversity: str = Field(..., example="High")
    initial_funding_amount: float = Field(..., ge=0, example=500000)
    team_size: int = Field(..., ge=1, example=8)
    months_since_founded: int = Field(..., ge=0, example=12)
    capital_efficiency: float = Field(..., example=0.65)
    market_timing: float = Field(..., example=0.72)
    pmf_score: float = Field(..., example=0.58)
    cac_efficiency: float = Field(..., example=0.45)
    burn_rate: float = Field(..., ge=0, example=50000)
    network_strength: float = Field(..., example=0.60)
    founder_commitment: float = Field(..., example=0.85)
    technical_moat: float = Field(..., example=0.55)
    revenue_growth: float = Field(..., example=0.30)
    has_customers: int = Field(..., ge=0, le=1, example=1)
    prev_experience: float = Field(..., example=0.70)
    investor_quality: float = Field(..., example=0.65)
    tier1_location: int = Field(..., ge=0, le=1, example=1)
    hot_sector: int = Field(..., ge=0, le=1, example=1)

    class Config:
        json_schema_extra = {
            "example": {
                "sector": "Fintech", "location": "NYC",
                "funding_stage": "Seed", "founder_background": "Technical",
                "team_diversity": "High", "initial_funding_amount": 500000,
                "team_size": 8, "months_since_founded": 12,
                "capital_efficiency": 0.65, "market_timing": 0.72,
                "pmf_score": 0.58, "cac_efficiency": 0.45,
                "burn_rate": 50000, "network_strength": 0.60,
                "founder_commitment": 0.85, "technical_moat": 0.55,
                "revenue_growth": 0.30, "has_customers": 1,
                "prev_experience": 0.70, "investor_quality": 0.65,
                "tier1_location": 1, "hot_sector": 1,
            }
        }


class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    confidence: str
    flagged_for_review: bool
    model_version: str


# ──────────────────────────────────────────────
# Endpoints — mirrors professor pattern
# ──────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message": "Welcome to the VenturePulse startup traction prediction API",
        "model": "Random Forest",
        "primary_metric": "Precision@100",
        "precision_at_100": "38%",
        "run_id": RUN_ID or "unknown",
    }


@app.get("/health")
def health():
    return {
        "status": "ok" if model is not None else "degraded",
        "run_id": RUN_ID or "unknown",
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(startup: StartupRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Check /health.")

    try:
        features_df = prepare_features(startup.dict())
        proba_output = model.predict(features_df)
        # pyfunc returns a DataFrame or array — extract positive class probability
        if hasattr(proba_output, "values"):
            proba_output = proba_output.values
        proba_output = np.array(proba_output).flatten()
        # If model returns probabilities for both classes, take index 1
        prob = float(proba_output[1]) if len(proba_output) == 2 else float(proba_output[0])
        pred = 1 if prob >= 0.5 else 0
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Prediction failed: {str(e)}")

    flagged = (
        startup.sector in FLAGGED_SECTORS
        or startup.location in FLAGGED_LOCATIONS
    )

    return PredictionResponse(
        prediction=pred,
        probability=round(prob, 4),
        confidence=probability_to_confidence(prob),
        flagged_for_review=flagged,
        model_version=RUN_ID or "unknown",
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=9696, reload=True)
