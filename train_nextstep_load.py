#!/usr/bin/env python3
"""
One-step-ahead load forecasting (next timestamp) for Grid Supply (kW).

Models:
  - XGBoost (tabular with engineered features)
  - LSTM (sequence)
  - GRU  (sequence)

Features:
  - Calendar (hour/dayofweek/month/weekend/night)
  - Target lags and rolling stats (auto tuned to data frequency)
  - All other numeric columns (voltage/current/weather/etc.) as exogenous inputs

Outputs (next to CSV in energy_load_next/):
  - metrics_scaled.csv   (MAE, RMSE, SMAPE on scaled)
  - metrics_real.csv     (MAE, RMSE, SMAPE in original units)
  - preds_[xgb|lstm|gru].csv (timestamp, y_true, y_pred in real units)
  - [optional] plot_[xgb|lstm|gru].png with --plots
"""

import argparse
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from xgboost import XGBRegressor

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt


# -------------------- helpers --------------------
def rmse_compat(y_true, y_pred):
    try:
        return mean_squared_error(y_true, y_pred, squared=False)
    except TypeError:
        return mean_squared_error(y_true, y_pred) ** 0.5

def smape(y_true, y_pred, eps=1e-9):
    denom = (np.abs(y_true) + np.abs(y_pred)) + eps
    return 100.0 * np.mean(2.0 * np.abs(y_pred - y_true) / denom)

def eval_all(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": rmse_compat(y_true, y_pred),
        "SMAPE_%": smape(y_true, y_pred),
    }

def infer_granularity(ts: pd.Series) -> str:
    t = pd.to_datetime(ts).sort_values()
    diffs = t.diff().dropna().values.astype("timedelta64[m]").astype(int)
    if len(diffs) == 0:
        return "H"
    m = int(np.median(diffs))
    if m <= 15:  return "15min"
    if m <= 30:  return "30min"
    if m <= 60:  return "H"
    if m <= 1440: return "D"
    return "D"

def lags_and_windows(freq: str):
    # choose lags/rolling windows based on median freq
    if freq == "15min":
        # per 15 min: 4/h, 96/day, 672/week
        return [1, 2, 4, 8, 24, 96, 192, 672], [8, 24, 96, 672]
    if freq == "30min":
        # per 30 min: 2/h, 48/day, 336/week
        return [1, 2, 4, 12, 24, 48, 96, 336], [4, 24, 48, 336]
    if freq == "H":
        return [1, 2, 6, 12, 24, 48, 168], [3, 24, 168]
    # daily or coarser
    return [1, 7, 14, 28], [3, 7, 28]

def add_calendar(df, time_col):
    t = pd.to_datetime(df[time_col])
    df["hour"] = t.dt.hour
    df["dayofweek"] = t.dt.dayofweek
    df["month"] = t.dt.month
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["is_night"] = ((df["hour"] <= 6) | (df["hour"] >= 22)).astype(int)
    return df

def add_target_lag_roll(df, target_col, lags, rolls):
    for L in lags:
        df[f"lag_{L}"] = df[target_col].shift(L)
    for W in rolls:
        df[f"roll_mean_{W}"] = df[target_col].shift(1).rolling(W).mean()
        df[f"roll_std_{W}"]  = df[target_col].shift(1).rolling(W).std()
    return df

def make_sequences(X_2d, y_1d, n_steps):
    Xs, ys = [], []
    for i in range(n_steps, len(X_2d)):
        Xs.append(X_2d[i-n_steps:i, :])
        ys.append(y_1d[i])
    return np.array(Xs), np.array(ys)

def build_lstm(input_shape):
    m = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(64),
        Dense(1)
    ])
    m.compile(optimizer="adam", loss="mse")
    return m

def build_gru(input_shape):
    m = Sequential([
        GRU(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        GRU(64),
        Dense(1)
    ])
    m.compile(optimizer="adam", loss="mse")
    return m

def plot_series(ts, y, yhat, title, path):
    plt.figure(figsize=(11,4))
    plt.plot(ts, y, label="Actual")
    plt.plot(ts, yhat, label="Predicted")
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Grid Supply (kW)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


# -------------------- main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--timecol", default="Timestamp")
    ap.add_argument("--target",  default="Grid Supply (kW)")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--seq_len", type=int, default=96, help="lookback for LSTM/GRU")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--plots", action="store_true")
    args = ap.parse_args()

    path = Path(args.csv)
    outdir = path.parent / "energy_load_next"
    outdir.mkdir(exist_ok=True)

    # Load
    df = pd.read_csv(path)
    # Fix temperature column name if mis-encoded
    if "Temperature (Â°C)" in df.columns and "Temperature (°C)" not in df.columns:
        df = df.rename(columns={"Temperature (Â°C)": "Temperature (°C)"})

    time_col = args.timecol
    target_col = args.target

    # Drop any non-feature "Predicted Load" leakage column if present
    if "Predicted Load (kW)" in df.columns:
        df = df.drop(columns=["Predicted Load (kW)"])

    # Parse time & sort
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col, target_col]).sort_values(time_col).reset_index(drop=True)

    # One-step ahead target: y(t+1)
    df["target_next"] = df[target_col].shift(-1)

    # Calendar + target lags/rolls
    df = add_calendar(df, time_col)
    freq = infer_granularity(df[time_col])
    lags, rolls = lags_and_windows(freq)
    df = add_target_lag_roll(df, target_col, lags, rolls)

    # Remove rows with NaNs from lags/rolls and from shifted target
    df = df.dropna(subset=[c for c in df.columns if c.startswith("lag_") or c.startswith("roll_")] + ["target_next"]).reset_index(drop=True)

    # Build feature list: all numeric columns excluding raw target_next and original target at t+1
    exclude = {time_col, "target_next"}
    feature_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    # We keep current-time target_col (t) as a feature via lags/rolls; the raw target at t is allowed as exogenous input at time t.
    # If you want to forbid using current target, comment the next line:
    # if target_col in feature_cols: feature_cols.remove(target_col)

    # Train/test split (time-based)
    n = len(df)
    split = int(n * (1 - args.test_size))
    train_df = df.iloc[:split].copy()
    test_df  = df.iloc[split:].copy()

    # Prepare X (tabular) and y
    X_train_raw = train_df[feature_cols].values
    X_test_raw  = test_df[feature_cols].values
    y_train_real = train_df["target_next"].values.reshape(-1, 1)
    y_test_real  = test_df["target_next"].values.reshape(-1, 1)

    # Scalers
    X_scaler = StandardScaler()
    y_scaler = MinMaxScaler()

    X_train = X_scaler.fit_transform(X_train_raw)
    X_test  = X_scaler.transform(X_test_raw)
    y_train = y_scaler.fit_transform(y_train_real).ravel()
    y_test  = y_scaler.transform(y_test_real).ravel()

    # ================= XGBoost (tabular) =================
    xgb = XGBRegressor(
        n_estimators=800,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        objective="reg:squarederror",
        n_jobs=-1,
    )
    xgb.fit(X_train, y_train)
    pred_xgb_scaled = xgb.predict(X_test)
    pred_xgb_real = y_scaler.inverse_transform(pred_xgb_scaled.reshape(-1,1)).ravel()

    # ================= Sequences for LSTM/GRU =================
    # Build continuous sequences across train+test for feature continuity
    X_all = np.vstack([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])
    n_steps = args.seq_len

    X_seq_all, y_seq_all = make_sequences(X_all, y_all, n_steps)
    # determine which sequence targets are in the test region
    test_start_idx = X_train.shape[0]
    # the target index corresponding to each sequence is its position in y_all
    tgt_idx = np.arange(n_steps, n_steps + len(y_seq_all))
    mask_tr = tgt_idx < test_start_idx
    mask_te = tgt_idx >= test_start_idx

    X_seq_tr = X_seq_all[mask_tr]
    y_seq_tr = y_seq_all[mask_tr]
    X_seq_te = X_seq_all[mask_te]
    y_seq_te = y_seq_all[mask_te]

    # LSTM
    early = EarlyStopping(patience=8, restore_best_weights=True)
    lstm = build_lstm((n_steps, X_seq_tr.shape[-1]))
    lstm.fit(X_seq_tr, y_seq_tr, epochs=args.epochs, batch_size=args.batch_size,
             validation_split=0.1, verbose=0, callbacks=[early])
    pred_lstm_scaled = lstm.predict(X_seq_te, verbose=0).ravel()
    pred_lstm_real = y_scaler.inverse_transform(pred_lstm_scaled.reshape(-1,1)).ravel()

    # GRU
    gru = build_gru((n_steps, X_seq_tr.shape[-1]))
    gru.fit(X_seq_tr, y_seq_tr, epochs=args.epochs, batch_size=args.batch_size,
            validation_split=0.1, verbose=0, callbacks=[early])
    pred_gru_scaled = gru.predict(X_seq_te, verbose=0).ravel()
    pred_gru_real = y_scaler.inverse_transform(pred_gru_scaled.reshape(-1,1)).ravel()

    # Align timestamps for outputs
    test_time = test_df[args.timecol].to_numpy()
    # For sequences, first valid prediction aligns to test_time[n_steps:]
    seq_time = test_time[n_steps : n_steps + len(pred_lstm_real)]

    # Build comparable y_true arrays (real units)
    y_true_xgb = y_test_real.ravel()
    y_true_seq = y_scaler.inverse_transform(y_seq_te.reshape(-1,1)).ravel()

    # Metrics
    metrics_scaled = pd.DataFrame({
        "XGBoost": eval_all(y_test, pred_xgb_scaled),
        "LSTM":    eval_all(y_seq_te, pred_lstm_scaled),
        "GRU":     eval_all(y_seq_te, pred_gru_scaled),
    }).T
    metrics_real = pd.DataFrame({
        "XGBoost": eval_all(y_true_xgb, pred_xgb_real),
        "LSTM":    eval_all(y_true_seq, pred_lstm_real),
        "GRU":     eval_all(y_true_seq, pred_gru_real),
    }).T

    metrics_scaled.to_csv(outdir / "metrics_scaled.csv")
    metrics_real.to_csv(outdir / "metrics_real.csv")

    # Predictions CSV (real units)
    pd.DataFrame({
        "timestamp": test_time[:len(y_true_xgb)],
        "y_true": y_true_xgb[:len(y_true_xgb)],
        "y_pred": pred_xgb_real[:len(y_true_xgb)]
    }).to_csv(outdir / "preds_xgb.csv", index=False)

    pd.DataFrame({
        "timestamp": seq_time,
        "y_true": y_true_seq[:len(seq_time)],
        "y_pred": pred_lstm_real[:len(seq_time)]
    }).to_csv(outdir / "preds_lstm.csv", index=False)

    pd.DataFrame({
        "timestamp": seq_time,
        "y_true": y_true_seq[:len(seq_time)],
        "y_pred": pred_gru_real[:len(seq_time)]
    }).to_csv(outdir / "preds_gru.csv", index=False)

    # Optional plots
    if args.plots:
        plot_series(test_time[:len(y_true_xgb)], y_true_xgb, pred_xgb_real[:len(y_true_xgb)],
                    "XGBoost: Actual vs Pred (next step)", outdir / "plot_xgb.png")
        plot_series(seq_time, y_true_seq[:len(seq_time)], pred_lstm_real[:len(seq_time)],
                    "LSTM: Actual vs Pred (next step)", outdir / "plot_lstm.png")
        plot_series(seq_time, y_true_seq[:len(seq_time)], pred_gru_real[:len(seq_time)],
                    "GRU: Actual vs Pred (next step)", outdir / "plot_gru.png")

    print("\n=== Metrics (scaled) ===")
    print(metrics_scaled.round(6))
    print("\n=== Metrics (real units) ===")
    print(metrics_real.round(6))
    print(f"\nArtifacts saved to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
