#!/usr/bin/env python3
"""
Energy Demand Forecasting (Pro):
- Rich feature engineering (calendar + holidays + lags + rolling stats)
- Models: Naive baseline, XGBoost, LSTM, GRU (multivariate)
- Metrics: MAE, RMSE, SMAPE on scaled and real units
- Optional plots (actual vs predicted)

Usage:
  python train_energy_models_pro.py --csv "/path/to/data.csv" --plots
"""

import argparse, warnings
from pathlib import Path
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")

# -------------------- deps --------------------
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from xgboost import XGBRegressor

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt

try:
    import holidays as holidays_lib
    HOLIDAYS_AVAILABLE = True
except Exception:
    HOLIDAYS_AVAILABLE = False

# -------------------- metrics --------------------
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

# -------------------- utils --------------------
TIME_CANDS = ["timestamp","datetime","time","date","ds"]
TARGET_CANDS = ["demand","load","consumption","kwh","energy","power","target","y"]

def autodetect_cols(df, time_hint=None, target_hint=None):
    time_col = time_hint if (time_hint and time_hint in df.columns) else next(
        (c for c in df.columns if c.lower() in TIME_CANDS or "time" in c.lower() or "date" in c.lower()),
        df.columns[0]
    )
    target_col = target_hint if (target_hint and target_hint in df.columns) else next(
        (c for c in df.columns if c.lower() in TARGET_CANDS),
        df.select_dtypes(include=[np.number]).columns[-1]
    )
    return time_col, target_col

def add_calendar_features(df, time_col, country="IN"):
    t = pd.to_datetime(df[time_col])
    df["hour"] = t.dt.hour
    df["dayofweek"] = t.dt.dayofweek
    df["month"] = t.dt.month
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    # simple season flags
    df["is_night"] = ((df["hour"] <= 6) | (df["hour"] >= 22)).astype(int)
    # holidays
    if HOLIDAYS_AVAILABLE:
        try:
            hol = holidays_lib.country_holidays(country, years=list(sorted(set(t.dt.year))))
            df["is_holiday"] = t.dt.date.astype("datetime64").isin(hol).astype(int)
        except Exception:
            df["is_holiday"] = 0
    else:
        df["is_holiday"] = 0
    return df

def add_lag_roll_features(df, target_col, lags=(1,2,3,24,48,168), rolls=(3,24,168)):
    for L in lags:
        df[f"lag_{L}"] = df[target_col].shift(L)
    for W in rolls:
        df[f"roll_mean_{W}"] = df[target_col].shift(1).rolling(W).mean()
        df[f"roll_std_{W}"]  = df[target_col].shift(1).rolling(W).std()
    return df

def make_sequences(X_2d, y_1d, n_steps):
    """Create (samples, n_steps, features) and (samples,) aligned."""
    X_seq, y_seq = [], []
    for i in range(n_steps, len(X_2d)):
        X_seq.append(X_2d[i-n_steps:i, :])
        y_seq.append(y_1d[i])
    return np.array(X_seq), np.array(y_seq)

# -------------------- main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--timecol")
    ap.add_argument("--target")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--seq_len", type=int, default=48)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--country_holidays", default="IN")
    ap.add_argument("--plots", action="store_true")
    args = ap.parse_args()

    # Load & clean
    df = pd.read_csv(args.csv)
    time_col, target_col = autodetect_cols(df, args.timecol, args.target)
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col, target_col]).sort_values(time_col).reset_index(drop=True)
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df.dropna(subset=[target_col])

    # Feature engineering
    df = add_calendar_features(df, time_col, country=args.country_holidays)
    df = add_lag_roll_features(df, target_col)

    # Drop rows with NaN due to lags/rolls
    df = df.dropna().reset_index(drop=True)

    # Select features:
    # numeric exogenous (auto-detect), excluding target
    exog_cols = [c for c in df.columns if c not in [time_col, target_col] and pd.api.types.is_numeric_dtype(df[c])]
    feature_cols = exog_cols  # multivariate inputs
    print(f"Using {len(feature_cols)} features.")

    # Split (time-based)
    n = len(df)
    split = int(n * (1 - args.test_size))
    train_df = df.iloc[:split].copy()
    test_df  = df.iloc[split:].copy()

    # Scale inputs & target (fit on train only)
    X_scaler = StandardScaler()
    y_scaler = MinMaxScaler()  # target often benefits from [0,1]

    X_train = X_scaler.fit_transform(train_df[feature_cols].values)
    X_test  = X_scaler.transform(test_df[feature_cols].values)

    y_train = train_df[target_col].values.reshape(-1,1)
    y_test  = test_df[target_col].values.reshape(-1,1)
    y_train_scaled = y_scaler.fit_transform(y_train).ravel()
    y_test_scaled  = y_scaler.transform(y_test).ravel()

    # ==================== Baseline: Naive last value ====================
    # predict each point as the previous observed value
    naive_pred_real = test_df[target_col].shift(1).fillna(method="bfill").values
    naive_metrics_real = eval_all(y_test.ravel(), naive_pred_real)

    # ==================== Model 1: XGBoost (tabular) ====================
    # Predict y_test using only X features (already include lags/rolls!)
    xgb = XGBRegressor(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        objective="reg:squarederror",
        n_jobs=-1,
    )
    xgb.fit(X_train, y_train_scaled)
    pred_xgb_scaled = xgb.predict(X_test)
    # inverse to real
    pred_xgb_real = y_scaler.inverse_transform(pred_xgb_scaled.reshape(-1,1)).ravel()
    xgb_metrics_scaled = eval_all(y_test_scaled, pred_xgb_scaled)
    xgb_metrics_real   = eval_all(y_test.ravel(), pred_xgb_real)

    # ==================== Sequence models (LSTM/GRU) ====================
    # Build sequences from the concatenated train+test for continuity
    X_all = np.vstack([X_train, X_test])
    y_all_scaled = np.concatenate([y_train_scaled, y_test_scaled])
    n_steps = args.seq_len

    # Indices for where test starts in sequence space
    test_start_idx = X_train.shape[0]

    # Create sequences across boundary, then slice out test part
    X_seq_all, y_seq_all = make_sequences(X_all, y_all_scaled, n_steps)
    # Figure which sequence targets belong to test region
    seq_targets_idx = np.arange(n_steps, n_steps + len(y_seq_all))
    # mask where original index >= test_start_idx
    mask_test_seq = seq_targets_idx >= test_start_idx
    X_seq_test = X_seq_all[mask_test_seq]
    y_seq_test = y_seq_all[mask_test_seq]

    # Also need train sequences (strictly before test)
    mask_train_seq = seq_targets_idx < test_start_idx
    X_seq_train = X_seq_all[mask_train_seq]
    y_seq_train = y_seq_all[mask_train_seq]

    def build_lstm(input_shape):
        m = Sequential([
            LSTM(96, activation="tanh", return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(64, activation="tanh"),
            Dense(1)
        ])
        m.compile(optimizer="adam", loss="mse")
        return m

    def build_gru(input_shape):
        m = Sequential([
            GRU(96, activation="tanh", return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            GRU(64, activation="tanh"),
            Dense(1)
        ])
        m.compile(optimizer="adam", loss="mse")
        return m

    early = EarlyStopping(patience=6, restore_best_weights=True)

    # LSTM
    lstm = build_lstm((n_steps, X_seq_train.shape[-1]))
    lstm.fit(X_seq_train, y_seq_train, epochs=args.epochs, batch_size=args.batch_size,
             validation_split=0.1, verbose=0, callbacks=[early])
    pred_lstm_scaled = lstm.predict(X_seq_test, verbose=0).ravel()
    pred_lstm_real = y_scaler.inverse_transform(pred_lstm_scaled.reshape(-1,1)).ravel()
    lstm_metrics_scaled = eval_all(y_seq_test, pred_lstm_scaled)
    # align real y_true for sequence test
    y_seq_test_real = y_scaler.inverse_transform(y_seq_test.reshape(-1,1)).ravel()
    lstm_metrics_real = eval_all(y_seq_test_real, pred_lstm_real)

    # GRU
    gru = build_gru((n_steps, X_seq_train.shape[-1]))
    gru.fit(X_seq_train, y_seq_train, epochs=args.epochs, batch_size=args.batch_size,
            validation_split=0.1, verbose=0, callbacks=[early])
    pred_gru_scaled = gru.predict(X_seq_test, verbose=0).ravel()
    pred_gru_real = y_scaler.inverse_transform(pred_gru_scaled.reshape(-1,1)).ravel()
    gru_metrics_scaled = eval_all(y_seq_test, pred_gru_scaled)
    gru_metrics_real = eval_all(y_seq_test_real, pred_gru_real)

    # ==================== Save artifacts ====================
    outdir = Path(args.csv).parent / "energy_models_pro"
    outdir.mkdir(exist_ok=True)

    # Metrics tables
    metrics_scaled = pd.DataFrame({
        "Naive":   eval_all(y_test_scaled[:len(pred_xgb_scaled)], y_test_scaled[:len(pred_xgb_scaled)]),  # placeholder to keep columns
        "XGBoost": xgb_metrics_scaled,
        "LSTM":    lstm_metrics_scaled,
        "GRU":     gru_metrics_scaled
    }).T
    metrics_scaled.to_csv(outdir / "metrics_scaled.csv")

    metrics_real = pd.DataFrame({
        "Naive":   naive_metrics_real,
        "XGBoost": xgb_metrics_real,
        "LSTM":    lstm_metrics_real,
        "GRU":     gru_metrics_real
    }).T
    metrics_real.to_csv(outdir / "metrics_real.csv")

    # Predictions CSVs (real units with timestamps)
    test_time = test_df[time_col].to_numpy()

    pd.DataFrame({
        "timestamp": test_time,
        "y_true": y_test.ravel(),
        "y_pred": pred_xgb_real
    }).to_csv(outdir / "preds_xgb.csv", index=False)

    # sequence timestamps align to the tail of test set after n_steps
    seq_time = test_time[n_steps : n_steps + len(pred_lstm_real)]
    pd.DataFrame({
        "timestamp": seq_time,
        "y_true": y_seq_test_real[:len(seq_time)],
        "y_pred": pred_lstm_real[:len(seq_time)]
    }).to_csv(outdir / "preds_lstm.csv", index=False)

    pd.DataFrame({
        "timestamp": seq_time,
        "y_true": y_seq_test_real[:len(seq_time)],
        "y_pred": pred_gru_real[:len(seq_time)]
    }).to_csv(outdir / "preds_gru.csv", index=False)

    # ==================== Optional plots ====================
    if args.plots:
        def plot_series(ts, y, yhat, title, path):
            plt.figure(figsize=(11,4))
            plt.plot(ts, y, label="Actual")
            plt.plot(ts, yhat, label="Predicted")
            plt.title(title)
            plt.xlabel("Time")
            plt.ylabel("Demand")
            plt.legend()
            plt.tight_layout()
            plt.savefig(path)
            plt.close()

        plot_series(test_time, y_test.ravel(), pred_xgb_real, "XGBoost: Actual vs Pred", outdir / "plot_xgb.png")
        plot_series(seq_time,  y_seq_test_real[:len(seq_time)], pred_lstm_real[:len(seq_time)], "LSTM: Actual vs Pred", outdir / "plot_lstm.png")
        plot_series(seq_time,  y_seq_test_real[:len(seq_time)], pred_gru_real[:len(seq_time)], "GRU: Actual vs Pred", outdir / "plot_gru.png")

    # Console summary
    print("\n=== Metrics (scaled) ===")
    print(metrics_scaled.round(6))
    print("\n=== Metrics (real units) ===")
    print(metrics_real.round(6))
    print(f"\nArtifacts saved to: {outdir.resolve()}")

if __name__ == "__main__":
    main()
