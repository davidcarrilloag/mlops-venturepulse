"""
Generate a monitoring report for VenturePulse predictions.

Mirrors professor's 05-monitoring/monitor.py pattern.
Uses evidently v0.4+ API.

Usage:
    python monitor.py
"""

import pandas as pd
from pathlib import Path

LOG_PATH    = Path("data/predictions.csv")
REPORT_PATH = Path("monitoring_report.html")
FAIRNESS_FLOOR = 0.30


def check_fairness(df: pd.DataFrame) -> None:
    """Print segment-level precision — mirrors 03-fairness-analysis."""
    print("\n📊 Fairness check:")
    df_known = df[df["high_traction"] >= 0].copy()
    if df_known.empty:
        print("   No ground truth available yet.")
        return

    for col in ["sector", "location"]:
        if col not in df_known.columns:
            continue
        print(f"\n  By {col}:")
        for segment, grp in df_known.groupby(col):
            if len(grp) < 5:
                continue
            n_pos = grp["prediction"].sum()
            if n_pos == 0:
                continue
            precision = grp[grp["prediction"] == 1]["high_traction"].sum() / n_pos
            flag = "❌ BELOW FLOOR" if precision < FAIRNESS_FLOOR else "✅"
            print(f"    {str(segment):20s}  precision={precision:.1%}  n={len(grp)}  {flag}")


def generate_report_evidently(reference, current) -> bool:
    """Try to generate Evidently report. Returns True if successful."""
    try:
        # Try new API (v0.4+)
        from evidently import DataDefinition, Dataset
        from evidently.presets import DataDriftPreset as NewDrift
        from evidently.report import Report as NewReport

        data_def = DataDefinition(
            numerical_columns=["probability", "initial_funding", "team_size"],
            categorical_columns=["sector", "location", "funding_stage"],
        )
        ref_ds  = Dataset.from_pandas(reference, data_definition=data_def)
        curr_ds = Dataset.from_pandas(current,  data_definition=data_def)

        report = NewReport([NewDrift()])
        result = report.run(ref_ds, curr_ds)
        result.save_html(str(REPORT_PATH))
        return True

    except Exception as e1:
        try:
            # Try legacy API (v0.3.x)
            from evidently.report import Report
            from evidently.metric_preset import DataDriftPreset
            from evidently import ColumnMapping

            col_map = ColumnMapping(
                prediction="prediction",
                numerical_features=["probability", "initial_funding", "team_size"],
                categorical_features=["sector", "location", "funding_stage"],
            )
            report = Report(metrics=[DataDriftPreset()])
            report.run(reference_data=reference, current_data=current,
                       column_mapping=col_map)
            report.save_html(str(REPORT_PATH))
            return True

        except Exception as e2:
            print(f"⚠️  Evidently v0.4 failed: {e1}")
            print(f"⚠️  Evidently v0.3 failed: {e2}")
            return False


def generate_report_manual(reference, current) -> None:
    """Fallback: generate a simple HTML report without Evidently."""
    print("📋 Generating manual HTML report (Evidently fallback)...")

    def drift_summary(col):
        if col not in reference.columns:
            return "N/A"
        ref_mean = reference[col].mean()
        cur_mean = current[col].mean()
        drift_pct = abs(cur_mean - ref_mean) / (abs(ref_mean) + 1e-9) * 100
        return f"ref={ref_mean:.3f} | cur={cur_mean:.3f} | drift={drift_pct:.1f}%"

    rows = ""
    for col in ["probability", "initial_funding", "team_size"]:
        rows += f"<tr><td>{col}</td><td>{drift_summary(col)}</td></tr>\n"

    pred_ref = reference["prediction"].mean()
    pred_cur = current["prediction"].mean()
    pred_drift = abs(pred_cur - pred_ref) * 100

    html = f"""<!DOCTYPE html>
<html>
<head>
  <title>VenturePulse Monitoring Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
    h1 {{ color: #2c3e50; }}
    h2 {{ color: #34495e; margin-top: 30px; }}
    table {{ border-collapse: collapse; width: 60%; background: white; }}
    th, td {{ border: 1px solid #ddd; padding: 10px 16px; text-align: left; }}
    th {{ background: #2c3e50; color: white; }}
    .ok {{ color: green; font-weight: bold; }}
    .alert {{ color: red; font-weight: bold; }}
    .box {{ background: white; padding: 20px; border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }}
  </style>
</head>
<body>
  <h1>🚀 VenturePulse — Monitoring Report</h1>
  <div class="box">
    <h2>📊 Dataset Summary</h2>
    <p>Reference window: <b>{len(reference)}</b> predictions</p>
    <p>Current window: <b>{len(current)}</b> predictions</p>
  </div>

  <div class="box">
    <h2>📈 Prediction Drift</h2>
    <p>Reference positive rate: <b>{pred_ref:.1%}</b></p>
    <p>Current positive rate: <b>{pred_cur:.1%}</b></p>
    <p>Drift: <b class="{'alert' if pred_drift > 10 else 'ok'}">{pred_drift:.1f}pp
      {'⚠️ ALERT' if pred_drift > 10 else '✅ OK'}</b></p>
  </div>

  <div class="box">
    <h2>🔢 Feature Drift</h2>
    <table>
      <tr><th>Feature</th><th>Summary</th></tr>
      {rows}
    </table>
  </div>

  <div class="box">
    <h2>⚖️ Fairness (from 03-fairness-analysis)</h2>
    <p>EdTech and Singapore flagged for human review at deployment.</p>
    <p>Floor threshold: <b>30%</b> precision per segment.</p>
  </div>
</body>
</html>"""

    REPORT_PATH.write_text(html, encoding="utf-8")


def main():
    print("\n📊 Starting VenturePulse monitoring report...\n")

    if not LOG_PATH.exists():
        raise FileNotFoundError(
            "❌ No logged predictions found. Run simulate.py first!"
        )

    df = pd.read_csv(LOG_PATH, parse_dates=["ts"])
    df = df.dropna(subset=["prediction", "probability"])
    print(f"✓ Loaded {len(df)} logged predictions")

    df = df.sort_values("ts")
    midpoint = len(df) // 2
    reference = df.iloc[:midpoint].copy()
    current   = df.iloc[midpoint:].copy()
    print(f"Reference: {len(reference)}  |  Current: {len(current)}")

    print("\n🧮 Generating drift report...")
    success = generate_report_evidently(reference, current)
    if not success:
        generate_report_manual(reference, current)

    print(f"✅ Report saved: {REPORT_PATH.resolve()}")
    print("Open monitoring_report.html in your browser.\n")

    check_fairness(df)


if __name__ == "__main__":
    main()
