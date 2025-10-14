import io
import os
import tempfile
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gradio as gr

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Try XGBoost; fallback to GradientBoosting if wheel is unavailable
try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except Exception:
    from sklearn.ensemble import GradientBoostingRegressor
    _HAS_XGB = False

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# --------------------- helpers ---------------------
def generate_synthetic(n_days=380, n_stores=6, n_items=15):
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    rows = []
    for s in range(1, n_stores + 1):
        for i in range(1, n_items + 1):
            base = 20 + 6 * np.sin(np.arange(n_days)/365 * 2*np.pi)   # yearly seasonality
            dow = pd.Series(dates).dt.dayofweek.values
            wknd = np.where(dow < 5, 1.0, 1.2)                        # weekends higher
            store_bias = 1 + (s - 1) * 0.05
            item_bias  = 1 + (i - 1) * 0.03
            promo = (np.random.rand(n_days) < 0.07).astype(float)     # random promos
            promo_boost = 1 + promo * 0.35
            noise = np.random.normal(0, 1.5, n_days)
            sales = np.clip(base * wknd * store_bias * item_bias * promo_boost + noise, 0, None)
            rows.append(pd.DataFrame({
                "date": dates, "store": str(s), "item": str(i), "sales": sales.round(2)
            }))
    return pd.concat(rows, ignore_index=True)

def add_time_features(df_):
    df_ = df_.copy()
    df_["dow"] = df_["date"].dt.dayofweek
    df_["week"] = df_["date"].dt.isocalendar().week.astype(int)
    df_["month"] = df_["date"].dt.month
    df_["year"] = df_["date"].dt.year
    df_["is_weekend"] = (df_["dow"] >= 5).astype(int)
    return df_

def make_supervised(df_, lags=(1,7,14,28), rolls=(7,14,28)):
    parts = []
    for (s,i), g in df_.groupby(["store","item"], as_index=False):
        g = g.sort_values("date").copy()
        for L in lags:  g[f"lag_{L}"] = g["sales"].shift(L)
        for R in rolls: g[f"rmean_{R}"] = g["sales"].shift(1).rolling(R).mean()
        parts.append(g)
    out = pd.concat(parts, ignore_index=True)
    need = max(max(lags), max(rolls))
    out = out[out["date"] >= (out["date"].min() + pd.Timedelta(days=need))].copy()
    out = out.dropna().reset_index(drop=True)
    return out

def build_pipeline(features_cat, features_num):
    # OneHotEncoder compatibility across sklearn versions
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
    pre = ColumnTransformer(
        transformers=[("cat", ohe, features_cat), ("num", "passthrough", features_num)]
    )
    if _HAS_XGB:
        model = XGBRegressor(
            n_estimators=600, learning_rate=0.05, max_depth=6,
            subsample=0.9, colsample_bytree=0.9, random_state=RANDOM_STATE,
            objective="reg:squarederror", tree_method="hist"
        )
    else:
        from sklearn.ensemble import GradientBoostingRegressor
        model = GradientBoostingRegressor(random_state=RANDOM_STATE)
    return Pipeline([("prep", pre), ("model", model)])

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

# ---- Baseline (used when history is very short) ----
def naive_backtest_and_forecast(series_df, store_id, item_id, horizon=14, k_mean=3):
    """
    Fallback when series has < 2 rows or train/test too small.
    - Backtest: last k_mean rolling mean vs. actual (if enough points)
    - Forecast: recent mean of last k_mean observations
    """
    s = series_df.sort_values("date").copy()
    s["pred"] = np.nan
    if len(s) > 1:
        s["rollmean"] = s["sales"].rolling(min(k_mean, len(s))).mean()
        s["pred"] = s["rollmean"].shift(1)
        tail = s.tail(min(120, len(s)))
    else:
        tail = s.copy()
        tail["pred"] = np.nan

    # simple forecast
    if len(s) == 0:
        future = pd.DataFrame(columns=["date","pred","store","item"])
    else:
        last_date = s["date"].max()
        recent_mean = float(s["sales"].tail(min(k_mean, len(s))).mean())
        preds = [{"date": d, "pred": recent_mean, "store": store_id, "item": item_id}
                 for d in pd.date_range(last_date + pd.Timedelta(days=1), periods=int(horizon), freq="D")]
        future = pd.DataFrame(preds)

    # plots
    plt.figure(figsize=(9,3))
    if not tail.empty:
        plt.plot(tail["date"], tail["sales"], label="actual")
        if tail["pred"].notna().any():
            plt.plot(tail["date"], tail["pred"], "--", label="pred (baseline)")
    plt.title("Backtest (baseline)")
    plt.legend(); plt.tight_layout()
    backtest_fig = plt.gcf()

    plt.figure(figsize=(8,3))
    if not future.empty:
        plt.plot(future["date"], future["pred"])
    plt.title(f"Forecast (baseline) next {int(horizon)} days")
    plt.tight_layout()
    forecast_fig = plt.gcf()

    if tail["pred"].notna().any():
        m_rmse = rmse(tail.loc[tail["pred"].notna(), "sales"], tail.loc[tail["pred"].notna(), "pred"])
        m_mae  = float(mean_absolute_error(tail.loc[tail["pred"].notna(), "sales"], tail.loc[tail["pred"].notna(), "pred"]))
        metrics = f"Baseline — RMSE: {m_rmse:.3f} | MAE: {m_mae:.3f}"
    else:
        metrics = "Baseline — not enough history to compute backtest metrics."

    return metrics, backtest_fig, forecast_fig, future

# ---- Auto-interpretation text ----
def interpret_output(kind, store_id, item_id, horizon, df_all, fc_df,
                     cv_rmse=None, cv_mae=None, test_rmse=None, test_mae=None):
    """
    Produce a short, recruiter-friendly narrative.
    """
    series = df_all[(df_all["store"]==store_id) & (df_all["item"]==item_id)].sort_values("date")
    date_min = series["date"].min().date() if not series.empty else "N/A"
    date_max = series["date"].max().date() if not series.empty else "N/A"

    # Growth next 7 days vs last 7 days
    last7 = float(series["sales"].tail(7).mean()) if len(series) >= 1 else np.nan
    next7 = float(fc_df["pred"].head(min(7, len(fc_df))).mean()) if not fc_df.empty else np.nan
    if not np.isnan(last7) and not np.isnan(next7) and last7 != 0:
        growth_pct = 100.0 * (next7 - last7) / last7
        growth_str = f"{growth_pct:+.1f}%"
    else:
        growth_str = "n/a"

    # Simple trend on forecast using slope
    trend = "n/a"
    if len(fc_df) >= 3:
        x = np.arange(len(fc_df))
        y = fc_df["pred"].values
        slope = np.polyfit(x, y, 1)[0]
        if slope > 0.02: trend = "upward"
        elif slope < -0.02: trend = "downward"
        else: trend = "roughly flat"

    metrics_line = ""
    if kind == "model" and cv_rmse is not None:
        metrics_line = f"**CV RMSE:** {cv_rmse:.3f} | **CV MAE:** {cv_mae:.3f} | **Test RMSE:** {test_rmse:.3f} | **Test MAE:** {test_mae:.3f}"
    elif kind == "baseline":
        metrics_line = "_Baseline used due to short history; metrics are approximate._"

    md = f"""
### 📌 Auto-interpretation
- **Series:** store **{store_id}**, item **{item_id}**
- **Data window:** {date_min} → {date_max}
- **Horizon:** next **{int(horizon)}** days
- **Engine:** {"XGBoost/GB Pipeline" if kind=="model" else "Naïve baseline (recent mean)"}  
- {metrics_line}

**What this means:**
- The model expects an average of **{next7:.2f}** units/day over the next 7 days vs **{last7:.2f}** in the last 7 days (**{growth_str}** change).
- Forecast trajectory looks **{trend}** over the horizon.
- Use this to plan inventory and staffing; peaks likely follow recent seasonality (weekends/holidays).

> **Note:** No confidence intervals shown. For production you’d add residual bootstrapping or quantile models to express uncertainty, and schedule daily retraining with data quality checks.
"""
    return md

# --------------------- Gradio functions ---------------------
def load_data(file, use_synth):
    try:
        # Read or generate dataframe
        if use_synth or file is None:
            df = generate_synthetic()
            source_msg = "Using synthetic dataset (auto-generated)."
        else:
            df = pd.read_csv(file.name, parse_dates=["date"])
            source_msg = "Loaded uploaded CSV."

        # Validation & normalization
        required = {"date","store","item","sales"}
        if not required.issubset(df.columns):
            return (gr.Update(), gr.Update(), gr.Update(), gr.Update(),
                    f"❌ CSV must include columns: {required}. Found: {list(df.columns)}", None)
        df["date"]  = pd.to_datetime(df["date"])
        df["store"] = df["store"].astype(str)
        df["item"]  = df["item"].astype(str)
        if df["sales"].isna().any():  # safety
            df["sales"] = df["sales"].fillna(0)
        df = df.sort_values(["store","item","date"]).reset_index(drop=True)

        preview = df.head(10)
        stores = sorted(df["store"].unique().tolist())
        items  = sorted(df["item"].unique().tolist())
        info = (f"✅ {source_msg} Rows: {len(df):,} | Stores: {len(stores)} | Items: {len(items)} | "
                f"Date range: {df['date'].min().date()} → {df['date'].max().date()}")

        df_json = df.to_json(orient="records", date_format="iso")
        return preview, gr.Dropdown(choices=stores, value=stores[0]), gr.Dropdown(choices=items, value=items[0]), info, df_json
    except Exception as e:
        return (gr.Update(), gr.Update(), gr.Update(), gr.Update(), f"❌ Load error: {e}", None)

def train_and_forecast(df_json, store_id, item_id, horizon):
    try:
        if df_json is None:
            return "Please load data first.", None, None, None, None, None

        df = pd.read_json(io.StringIO(df_json), orient="records")
        df["date"] = pd.to_datetime(df["date"])
        # IMPORTANT: force to strings after JSON load to avoid type mismatch
        df["store"] = df["store"].astype(str)
        df["item"]  = df["item"].astype(str)

        store_id, item_id = str(store_id), str(item_id)
        horizon = int(horizon)

        # Slice selected series for baseline checks
        series = df[(df["store"]==store_id) & (df["item"]==item_id)].copy().sort_values("date")
        n = len(series)

        # If the series has < 2 rows, use baseline immediately
        if n < 2:
            base_metrics, base_backtest, base_forecast, fc_tbl = naive_backtest_and_forecast(series, store_id, item_id, horizon=horizon)
            tmpdir = tempfile.mkdtemp()
            out_path = os.path.join(tmpdir, f"forecast_store_{store_id}_item_{item_id}.csv")
            fc_tbl.to_csv(out_path, index=False)
            note = interpret_output("baseline", store_id, item_id, horizon, df, fc_tbl)
            return f"ℹ️ {base_metrics}", base_backtest, base_forecast, fc_tbl, out_path, note

        # ----- Feature engineering for model path -----
        df_fe = add_time_features(df)
        df_fe = make_supervised(df_fe)

        FEATURES_NUM = [c for c in df_fe.columns if c.startswith(("lag_","rmean_")) or c in ["dow","week","month","year","is_weekend"]]
        FEATURES_CAT = ["store","item"]
        TARGET = "sales"

        # Time split (last H days test). If train/test tiny, fallback to baseline.
        last_date = df_fe["date"].max()
        test_start = last_date - pd.Timedelta(days=horizon - 1)
        train_df = df_fe[df_fe["date"] < test_start].copy()
        test_df  = df_fe[df_fe["date"] >= test_start].copy()

        if train_df.empty or test_df.empty or len(train_df) < 20:
            base_metrics, base_backtest, base_forecast, fc_tbl = naive_backtest_and_forecast(series, store_id, item_id, horizon=horizon)
            tmpdir = tempfile.mkdtemp()
            out_path = os.path.join(tmpdir, f"forecast_store_{store_id}_item_{item_id}.csv")
            fc_tbl.to_csv(out_path, index=False)
            note = interpret_output("baseline", store_id, item_id, horizon, df, fc_tbl)
            return f"ℹ️ Using baseline due to short history. {base_metrics}", base_backtest, base_forecast, fc_tbl, out_path, note

        # ----- Model path -----
        tscv = TimeSeriesSplit(n_splits=5)
        pipe = build_pipeline(FEATURES_CAT, FEATURES_NUM)
        X = train_df[FEATURES_CAT + FEATURES_NUM]; y = train_df[TARGET]
        cv_scores = []
        for (tr_idx, va_idx) in tscv.split(X):
            X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
            y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
            pipe.fit(X_tr, y_tr)
            pred = pipe.predict(X_va)
            cv_scores.append((rmse(y_va, pred), float(mean_absolute_error(y_va, pred))))
        cv_rmse = float(np.mean([r for r,_ in cv_scores]))
        cv_mae  = float(np.mean([m for _,m in cv_scores]))

        pipe.fit(train_df[FEATURES_CAT + FEATURES_NUM], train_df[TARGET])
        pred_test = pipe.predict(test_df[FEATURES_CAT + FEATURES_NUM])
        test_rmse = rmse(test_df[TARGET], pred_test)
        test_mae  = float(mean_absolute_error(test_df[TARGET], pred_test))

        # Backtest plot for selected series
        def backtest_plot():
            sub = df_fe[(df_fe["store"]==store_id) & (df_fe["item"]==item_id)].copy().sort_values("date")
            sub["pred"] = np.nan
            mask = sub["date"] >= test_start
            if mask.any():
                sub.loc[mask, "pred"] = pipe.predict(sub.loc[mask, FEATURES_CAT + FEATURES_NUM])
            tail = sub.tail(120)
            plt.figure(figsize=(9,3))
            plt.plot(tail["date"], tail["sales"], label="actual")
            if tail["pred"].notna().any():
                plt.plot(tail["date"], tail["pred"], "--", label="pred")
            plt.title(f"Backtest: store={store_id}, item={item_id}")
            plt.legend(); plt.tight_layout()
            return plt.gcf()

        # Adaptive recursive forecast
        def recursive_forecast(h=14, base_lags=(1,7,14,28), base_rolls=(7,14,28)):
            hist = df[(df["store"]==store_id) & (df["item"]==item_id)].sort_values("date").copy()
            nn = len(hist)
            max_allowed = max(1, nn - 1)
            lags  = [L for L in base_lags  if L <= max_allowed] or [1]
            rolls = [R for R in base_rolls if R <= max_allowed] or [min(7, max_allowed)]

            work = hist[["date","store","item","sales"]].copy()
            preds = []
            for _ in range(int(h)):
                d = work["date"].max() + pd.Timedelta(days=1)
                row = pd.DataFrame({"date":[d],"store":[store_id],"item":[item_id],"sales":[np.nan]})
                tmp = pd.concat([work, row], ignore_index=True)
                tmp = add_time_features(tmp)
                parts = []
                for (s,i), g in tmp.groupby(["store","item"], as_index=False):
                    g = g.sort_values("date").copy()
                    for L in lags:  g[f"lag_{L}"] = g["sales"].shift(L)
                    for R in rolls: g[f"rmean_{R}"] = g["sales"].shift(1).rolling(R).mean()
                    parts.append(g)
                tmp_sup = pd.concat(parts, ignore_index=True)
                new_row = tmp_sup.sort_values("date").tail(1)

                FEATURES_NUM2 = [c for c in new_row.columns
                                 if c.startswith(("lag_","rmean_")) or c in ["dow","week","month","year","is_weekend"]]
                FEATURES_CAT2 = ["store","item"]

                yhat = float(pipe.predict(new_row[FEATURES_CAT2 + FEATURES_NUM2])[0])
                preds.append({"date": d, "store": store_id, "item": item_id, "pred": yhat})
                work = pd.concat(
                    [work, pd.DataFrame({"date":[d],"store":[store_id],"item":[item_id],"sales":[yhat]})],
                    ignore_index=True
                )
            return pd.DataFrame(preds)

        fc = recursive_forecast(h=horizon)

        # Forecast plot
        def forecast_plot():
            plt.figure(figsize=(8,3))
            plt.plot(fc["date"], fc["pred"])
            plt.title(f"Forecast (next {int(horizon)} days) — store={store_id}, item={item_id}")
            plt.tight_layout()
            return plt.gcf()

        tmpdir = tempfile.mkdtemp()
        out_path = os.path.join(tmpdir, f"forecast_store_{store_id}_item_{item_id}.csv")
        fc.to_csv(out_path, index=False)

        metrics = (f"Model — CV RMSE: {cv_rmse:.3f} | CV MAE: {cv_mae:.3f} | "
                   f"Test RMSE: {test_rmse:.3f} | Test MAE: {test_mae:.3f} | "
                   f"Model: {'XGBoost' if _HAS_XGB else 'GradientBoosting'}")

        note = interpret_output("model", store_id, item_id, horizon, df, fc,
                                cv_rmse=cv_rmse, cv_mae=cv_mae, test_rmse=test_rmse, test_mae=test_mae)

        return metrics, backtest_plot(), forecast_plot(), fc, out_path, note

    except Exception as e:
        # Last-resort guard: fallback to baseline for ANY unexpected issue
        try:
            df = pd.read_json(io.StringIO(df_json), orient="records")
            df["date"] = pd.to_datetime(df["date"])
            df["store"] = df["store"].astype(str)
            df["item"]  = df["item"].astype(str)
            store_id, item_id = str(store_id), str(item_id)
            series = df[(df["store"]==store_id) & (df["item"]==item_id)].copy().sort_values("date")
            base_metrics, base_backtest, base_forecast, fc_tbl = naive_backtest_and_forecast(series, store_id, item_id, horizon=int(horizon))
            tmpdir = tempfile.mkdtemp()
            out_path = os.path.join(tmpdir, f"forecast_store_{store_id}_item_{item_id}.csv")
            fc_tbl.to_csv(out_path, index=False)
            note = interpret_output("baseline", store_id, item_id, horizon, df, fc_tbl)
            return f"ℹ️ Using baseline due to: {e}", base_backtest, base_forecast, fc_tbl, out_path, note
        except Exception as e2:
            err = f"❌ Error: {e2}"
            return err, None, None, None, None, None

# --------------------- UI ---------------------
with gr.Blocks(title="Retail Sales Forecasting") as demo:
    gr.Markdown("# 📈 Retail Sales Forecasting (Multi-Store, Multi-Item)")
    gr.Markdown("Upload CSV with columns **date, store, item, sales** (daily) or tick **Use synthetic data**.")

    with gr.Row():
        file_in = gr.File(label="Upload CSV")
        use_synth = gr.Checkbox(value=True, label="Use synthetic data (ignore upload)")

    load_btn = gr.Button("Load data", variant="primary")
    info_box = gr.Markdown("")
    df_preview = gr.Dataframe(interactive=False, label="Preview (first 10 rows)")

    with gr.Row():
        store_dd = gr.Dropdown(choices=[], label="Store")
        item_dd  = gr.Dropdown(choices=[], label="Item")
        horizon  = gr.Slider(7, 60, value=14, step=1, label="Forecast horizon (days)")

    df_state = gr.State(value=None)

    load_btn.click(
        fn=load_data,
        inputs=[file_in, use_synth],
        outputs=[df_preview, store_dd, item_dd, info_box, df_state]
    )

    gr.Markdown("---")
    run_btn = gr.Button("Train & Forecast", variant="primary")

    metrics_out = gr.Markdown(label="Metrics")
    backtest_plot_out = gr.Plot(label="Backtest")
    forecast_plot_out = gr.Plot(label="Forecast")
    df_forecast_out = gr.Dataframe(label="Forecast table", interactive=False)
    download_out = gr.File(label="Download forecast CSV")
    narrative_out = gr.Markdown(label="Auto-interpretation")

    run_btn.click(
        fn=train_and_forecast,
        inputs=[df_state, store_dd, item_dd, horizon],
        outputs=[metrics_out, backtest_plot_out, forecast_plot_out, df_forecast_out, download_out, narrative_out]
    )

if __name__ == "__main__":
    demo.launch()
