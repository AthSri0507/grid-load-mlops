#!/usr/bin/env python3
"""
Batch inference with a saved model.

Usage:
  python predict_energy.py --csv "/path/to/new_data.csv" --model "energy_models/models/RandomForest.joblib" --timecol timestamp --target demand

Writes predictions to predictions.csv next to the input CSV.
"""
import argparse
from pathlib import Path
import pandas as pd
from joblib import load

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--timecol", default=None)
    ap.add_argument("--target",  default=None)
    args = ap.parse_args()

    model_path = Path(args.model)
    csv_path   = Path(args.csv)
    out_csv    = csv_path.parent / "predictions.csv"

    fe_path = model_path.parent.parent / "transformers" / "time_feature_engineer.joblib"
    fe = load(fe_path)

    df = pd.read_csv(csv_path)
    time_col   = args.timecol or fe.time_col
    target_col = args.target  or fe.target_col

    df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    df = df.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)

    dff = fe.transform(df)
    need = [c for c in dff.columns if c.startswith("lag_") or c.startswith("roll_mean_")]
    dff = dff.dropna(subset=need).reset_index(drop=True)

    if target_col in dff.columns:
        dff = dff.drop(columns=[target_col])

    model = load(model_path)
    preds = model.predict(dff)

    out = dff[[time_col]].copy()
    out["prediction"] = preds
    out.to_csv(out_csv, index=False)
    print(f"Wrote predictions to: {out_csv.resolve()}")

if __name__ == "__main__":
    main()
