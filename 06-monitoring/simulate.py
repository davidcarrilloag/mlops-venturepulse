"""
Simulate real-time requests to the running VenturePulse API and log predictions.

Mirrors professor's 05-monitoring/simulate.py pattern.

Usage:
    python simulate.py

This will:
- Load real startups from the dataset
- Send random samples to the /predict endpoint
- Collect predictions + ground truth (high_traction)
- Append them to data/predictions.csv
"""

import math
import time
import requests
import pandas as pd
from pathlib import Path


def safe_float(val, default=0.0) -> float:
    """Return default if val is NaN or inf."""
    try:
        v = float(val)
        return default if (math.isnan(v) or math.isinf(v)) else v
    except (TypeError, ValueError):
        return default


def safe_int(val, default=0) -> int:
    """Return default if val is NaN."""
    try:
        v = float(val)
        return default if (math.isnan(v) or math.isinf(v)) else int(v)
    except (TypeError, ValueError):
        return default


def safe_str(val, default="Unknown") -> str:
    """Return default if val is NaN."""
    if pd.isna(val):
        return default
    return str(val)

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
API_URL  = "http://localhost:9696/predict"
LOG_PATH = Path("data/predictions.csv")
DATA_PATH = Path("../data/raw/venturepulse_dataset.csv")
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

CATEGORICAL_COLS = ["sector", "location", "funding_stage",
                    "founder_background", "team_diversity"]


def load_data(n_rows: int = 100) -> pd.DataFrame:
    """Load a sample of VenturePulse startups for simulation."""
    print(f"📥 Loading data from {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    df = df.sample(n=n_rows, random_state=99).reset_index(drop=True)
    print(f"✓ Loaded {len(df)} rows for simulation")
    return df


def simulate_requests(df: pd.DataFrame, sleep_s: float = 0.05):
    """Send each startup to the prediction API and log the results."""
    rows = []

    for i, row in df.iterrows():
        payload = {
            "sector":                  safe_str(row.get("sector"), "SaaS"),
            "location":                safe_str(row.get("location"), "NYC"),
            "funding_stage":           safe_str(row.get("funding_stage"), "Seed"),
            "founder_background":      safe_str(row.get("founder_background"), "Technical"),
            "team_diversity":          safe_str(row.get("team_diversity"), "Medium"),
            "initial_funding_amount":  safe_float(row.get("initial_funding_amount"), 500000.0),
            "team_size":               safe_int(row.get("team_size"), 8),
            "months_since_founded":    safe_int(row.get("months_since_founded"), 12),
            "capital_efficiency":      safe_float(row.get("capital_efficiency"), 0.5),
            "market_timing":           safe_float(row.get("market_timing"), 0.5),
            "pmf_score":               safe_float(row.get("pmf_score"), 0.5),
            "cac_efficiency":          safe_float(row.get("cac_efficiency"), 0.5),
            "burn_rate":               safe_float(row.get("burn_rate"), 50000.0),
            "network_strength":        safe_float(row.get("network_strength"), 0.5),
            "founder_commitment":      safe_float(row.get("founder_commitment"), 0.5),
            "technical_moat":          safe_float(row.get("technical_moat"), 0.5),
            "revenue_growth":          safe_float(row.get("revenue_growth"), 0.2),
            "has_customers":           safe_int(row.get("has_customers"), 0),
            "prev_experience":         safe_float(row.get("prev_experience"), 0.5),
            "investor_quality":        safe_float(row.get("investor_quality"), 0.5),
            "tier1_location":          safe_int(row.get("tier1_location"), 0),
            "hot_sector":              safe_int(row.get("hot_sector"), 0),
        }

        try:
            resp = requests.post(API_URL, json=payload, timeout=5)
            resp.raise_for_status()
            result = resp.json()

            rows.append({
                "ts":               pd.Timestamp.utcnow().isoformat(),
                "sector":           payload["sector"],
                "location":         payload["location"],
                "funding_stage":    payload["funding_stage"],
                "team_size":        payload["team_size"],
                "initial_funding":  payload["initial_funding_amount"],
                "prediction":       result["prediction"],
                "probability":      result["probability"],
                "flagged":          result["flagged_for_review"],
                "high_traction":    int(row.get("high_traction", -1)),  # ground truth
            })

        except Exception as e:
            print(f"⚠️  Request failed: {e}")

        if (i + 1) % 20 == 0:
            print(f"   Progress: {i + 1}/{len(df)}")
        time.sleep(sleep_s)

    return pd.DataFrame(rows)


def main():
    print("\n🚀 Starting VenturePulse simulation...\n")
    df = load_data(n_rows=100)
    out = simulate_requests(df)

    if out.empty:
        print("❌ No predictions recorded. Make sure app.py is running.")
        return

    if LOG_PATH.exists():
        prev = pd.read_csv(LOG_PATH)
        out = pd.concat([prev, out], ignore_index=True)

    out.to_csv(LOG_PATH, index=False)
    print(f"✅ Wrote {len(out)} total rows to {LOG_PATH}")


if __name__ == "__main__":
    main()
