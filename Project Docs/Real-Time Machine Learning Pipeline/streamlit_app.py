
"""
streamlit_app.py
----------------
Interactive frontend for real-time ticker (or any symbol) data & predictions.
- Displays live quotes from Kafka
- Shows model predictions
- Adds diagnostics: accuracy metrics & visuals (actual vs predicted, residuals, histogram, scatter)
- Includes a 'what-if' scenario slider

Run:
    streamlit run streamlit_app.py --server.runOnSave true

Environment vars expected:
    FINNHUB_API_KEY  # used by producer if launched via this app
"""

import json
import time
from collections import deque

import numpy as np
import pandas as pd
import streamlit as st # type: ignore
from streamlit_autorefresh import st_autorefresh # type: ignore
from kafka import KafkaConsumer # type: ignore
import matplotlib.pyplot as plt
from model_utils import OnlinePricePredictor


st.set_page_config(page_title="Real-time Stock Predictor", layout="wide")
st_autorefresh(interval=10000, limit=None, key="auto_refresh")

st.sidebar.header("Settings")
bootstrap_servers = st.sidebar.text_input("Kafka bootstrap servers", value="localhost:9092")
quotes_topic = st.sidebar.text_input("Quotes topic", value="amc-quotes")
preds_topic = st.sidebar.text_input("Predictions topic", value="amc-predictions")
symbol = st.sidebar.text_input("Symbol label (for display)", value="AMC")

start_from_latest = st.sidebar.checkbox("Start from latest only (ignore history)", value=True)
only_selected_symbol = st.sidebar.checkbox("Only show selected symbol", value=True)

if st.sidebar.button("Clear UI buffers"):
    st.session_state.pop("quotes_buffer", None)
    st.session_state.pop("preds_buffer", None)
    st.success("Cleared UI buffers for this session. The app will refill with fresh messages.")

eval_window = st.sidebar.slider("Evaluation window (# of predictions)", 30, 1000, 200, step=10)

if "quotes_buffer" not in st.session_state:
    st.session_state.quotes_buffer = deque(maxlen=5000)
if "preds_buffer" not in st.session_state:
    st.session_state.preds_buffer = deque(maxlen=5000)
if "model" not in st.session_state:
    st.session_state.model = OnlinePricePredictor()

@st.cache_resource(show_spinner=False)
def get_consumer(topic: str, bootstrap: str, latest: bool):
    """
    Cache Kafka consumer per topic.

    When latest=True:
        - start at 'latest' with a stable group id (only show new data).
    When latest=False:
        - start at 'earliest' with a unique group id (show history immediately).
    """
    auto = "latest" if latest else "earliest"
    group = f"streamlit-{topic}" if latest else f"streamlit-{topic}-{int(time.time())}"
    return KafkaConsumer(
        topic,
        bootstrap_servers=bootstrap,
        auto_offset_reset=auto,
        enable_auto_commit=True,
        group_id=group,
        value_deserializer=lambda x: json.loads(x.decode("utf-8")),
        key_deserializer=lambda x: x.decode("utf-8") if x else None,
        consumer_timeout_ms=2000,
        max_poll_records=500,
    )


def poll_consumer(consumer: KafkaConsumer, buffer: deque):
    """Poll the Kafka consumer and append any messages to the buffer."""
    try:
        recs = consumer.poll(timeout_ms=500, max_records=500)
        count = 0
        for _, msgs in recs.items():
            for m in msgs:
                buffer.append(m.value)
                count += 1
        return count
    except Exception:
        return 0


def buffer_df(buffer: deque) -> pd.DataFrame:
    """Convert a buffer of dicts to a sorted DataFrame (by epoch if present)."""
    if not buffer:
        return pd.DataFrame()
    df = pd.DataFrame(list(buffer))
    if "epoch" in df.columns:
        df = df.sort_values("epoch")
    return df

def align_predictions_with_actuals(quotes: pd.DataFrame, preds: pd.DataFrame, grace_seconds: int = 60) -> pd.DataFrame:
    """
    Align each prediction (made at epoch t) to an actual next price.
    Strategy:
      1) Try strict next tick: first quote strictly after t (forward, no exact match).
      2) If none, allow same-tick match (forward, allow exact matches).
      3) If still none and grace_seconds > 0, allow the first quote within [t, t+grace].
    Returns a DataFrame with predicted/actual pairs and error columns.
    """
    if quotes.empty or preds.empty:
        return pd.DataFrame()

    q = quotes.copy()
    p = preds.copy()
    q["epoch"] = pd.to_numeric(q["epoch"], errors="coerce")
    q["price"] = pd.to_numeric(q["price"], errors="coerce")
    p["epoch"] = pd.to_numeric(p["epoch"], errors="coerce")
    p["predicted_price"] = pd.to_numeric(p["predicted_price"], errors="coerce")
    q = q.dropna(subset=["epoch", "price"])
    p = p.dropna(subset=["epoch", "predicted_price"])
    if q.empty or p.empty:
        return pd.DataFrame()

    q = q.sort_values(["symbol", "epoch"])
    p = p.sort_values(["symbol", "epoch"])

    nxt = pd.merge_asof(
        p.rename(columns={"epoch": "epoch_pred"}),
        q[["symbol", "epoch", "price"]].rename(columns={"epoch": "epoch_actual", "price": "actual_next_price"}),
        by="symbol",
        left_on="epoch_pred",
        right_on="epoch_actual",
        direction="forward",
        allow_exact_matches=False,
    )
    df = nxt.dropna(subset=["actual_next_price", "predicted_price"]).copy()

    if df.empty:
        nxt2 = pd.merge_asof(
            p.rename(columns={"epoch": "epoch_pred"}),
            q[["symbol", "epoch", "price"]].rename(columns={"epoch": "epoch_actual", "price": "actual_next_price"}),
            by="symbol",
            left_on="epoch_pred",
            right_on="epoch_actual",
            direction="forward",
            allow_exact_matches=True,   
        )
        df = nxt2.dropna(subset=["actual_next_price", "predicted_price"]).copy()

    if df.empty and grace_seconds and grace_seconds > 0:
        p_ = p.rename(columns={"epoch": "epoch_pred"}).copy()
        q_ = q.rename(columns={"epoch": "epoch_actual"})[["symbol", "epoch_actual", "price"]].copy()
        q_ = q_.rename(columns={"price": "actual_next_price"})
        tmp = pd.merge_asof(
            p_,
            q_,
            by="symbol",
            left_on="epoch_pred",
            right_on="epoch_actual",
            direction="forward",
            allow_exact_matches=True,
        )
        tmp = tmp[(tmp["epoch_actual"] - tmp["epoch_pred"]).between(0, grace_seconds)]
        df = tmp.dropna(subset=["actual_next_price", "predicted_price"]).copy()

    if df.empty:
        return df 
    baseline = pd.merge_asof(
        p.rename(columns={"epoch": "epoch_pred"}),
        q[["symbol", "epoch", "price"]].rename(columns={"price": "last_observed_price"}),
        by="symbol",
        left_on="epoch_pred",
        right_on="epoch",
        direction="backward",
        allow_exact_matches=True,
    )
    df["error"] = df["predicted_price"] - df["actual_next_price"]
    df["abs_error"] = df["error"].abs()
    df["pct_error"] = df["error"] / df["actual_next_price"].replace(0, np.nan)
    if "last_observed_price" in baseline.columns:
        df["baseline_pred"] = baseline["last_observed_price"].values
        df["baseline_error"] = df["baseline_pred"] - df["actual_next_price"]
        df["baseline_abs_error"] = df["baseline_error"].abs()
        df["baseline_pct_error"] = df["baseline_error"] / df["actual_next_price"].replace(0, np.nan)

    keep = [
        "symbol", "epoch_pred", "epoch_actual",
        "predicted_price", "actual_next_price",
        "error", "abs_error", "pct_error",
        "baseline_pred", "baseline_error", "baseline_abs_error", "baseline_pct_error",
    ]
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["predicted_price", "actual_next_price"])
    cols = [c for c in keep if c in df.columns]
    return df[cols].sort_values("epoch_pred").reset_index(drop=True)


def compute_metrics(ev: pd.DataFrame, window: int):
    """
    Compute rolling and overall metrics vs. baseline.
    Returns dicts for model and baseline with MAE, RMSE, MAPE.
    """
    if ev.empty:
        return None, None

    ev = ev.replace([np.inf, -np.inf], np.nan)
    ev = ev.dropna(subset=["actual_next_price", "predicted_price", "error", "abs_error"])

    if ev.empty:
        return None, None

    evw = ev.tail(window)

    def _safe_mape(ae, y):
        denom = np.where(np.asarray(y) == 0, np.nan, np.asarray(y))
        return np.nanmean(np.asarray(ae) / denom)

    model = {
        "MAE_window": float(evw["abs_error"].mean()),
        "RMSE_window": float(np.sqrt((evw["error"] ** 2).mean())),
        "MAPE_window": float(_safe_mape(evw["abs_error"], evw["actual_next_price"])) * 100.0,
        "MAE_overall": float(ev["abs_error"].mean()),
        "RMSE_overall": float(np.sqrt((ev["error"] ** 2).mean())),
        "MAPE_overall": float(_safe_mape(ev["abs_error"], ev["actual_next_price"])) * 100.0,
        "count_window": int(len(evw)),
        "count_overall": int(len(ev)),
    }
    if "baseline_abs_error" in ev.columns and "baseline_error" in ev.columns:
        base = {
            "MAE_window": float(evw["baseline_abs_error"].mean()),
            "RMSE_window": float(np.sqrt((evw["baseline_error"] ** 2).mean())),
            "MAPE_window": float(_safe_mape(evw["baseline_abs_error"], evw["actual_next_price"])) * 100.0,
            "MAE_overall": float(ev["baseline_abs_error"].mean()),
            "RMSE_overall": float(np.sqrt((ev["baseline_error"] ** 2).mean())),
            "MAPE_overall": float(_safe_mape(ev["baseline_abs_error"], ev["actual_next_price"])) * 100.0,
        }
    else:
        base = None

    return model, base

def finite_series(s: pd.Series) -> pd.Series:
    """Return only finite values from a pandas Series (drop NaN/inf)."""
    return s.replace([np.inf, -np.inf], np.nan).dropna()


col_left, col_right = st.columns([2, 1])

with col_left:
    st.title("Real-time Price & Predictions")
    st.caption("Quotes every ~15s; model updates every ~120s (consumer).")

    quotes_consumer = get_consumer(quotes_topic, bootstrap_servers, latest=start_from_latest)
    preds_consumer = get_consumer(preds_topic, bootstrap_servers, latest=start_from_latest)

    _ = poll_consumer(quotes_consumer, st.session_state.quotes_buffer)
    _ = poll_consumer(preds_consumer, st.session_state.preds_buffer)

    qdf = buffer_df(st.session_state.quotes_buffer)
    pdf = buffer_df(st.session_state.preds_buffer)

    if not qdf.empty:
        st.session_state.model.partial_update(qdf)

    if only_selected_symbol:
        if not qdf.empty and "symbol" in qdf.columns:
            qdf = qdf[qdf["symbol"] == symbol]
        if not pdf.empty and "symbol" in pdf.columns:
            pdf = pdf[pdf["symbol"] == symbol]
    if not qdf.empty:
        px = finite_series(qdf.set_index("epoch")["price"])
        if not px.empty:
            st.subheader("Live Price")
            st.line_chart(px, height=240)
        else:
            st.info("Waiting for valid price data…")
    st.subheader("Accuracy & Diagnostics")

    if not pdf.empty and not qdf.empty:
        ev = align_predictions_with_actuals(qdf, pdf)
        if not ev.empty:
            model_m, base_m = compute_metrics(ev, eval_window)
            if model_m is not None:
                m1, m2, m3 = st.columns(3)
                if base_m is not None:
                    m1.metric("MAE (last N)", f"{model_m['MAE_window']:.4f}",
                              delta=f"{(base_m['MAE_window']-model_m['MAE_window']):+.4f} vs baseline")
                    m2.metric("RMSE (last N)", f"{model_m['RMSE_window']:.4f}",
                              delta=f"{(base_m['RMSE_window']-model_m['RMSE_window']):+.4f} vs baseline")
                    m3.metric("MAPE% (last N)", f"{model_m['MAPE_window']:.2f}%",
                              delta=f"{(base_m['MAPE_window']-model_m['MAPE_window']):+.2f}pp vs baseline")
                else:
                    m1.metric("MAE (last N)", f"{model_m['MAE_window']:.4f}")
                    m2.metric("RMSE (last N)", f"{model_m['RMSE_window']:.4f}")
                    m3.metric("MAPE% (last N)", f"{model_m['MAPE_window']:.2f}%")

                st.caption(f"Evaluation window N = {model_m['count_window']}, total evaluated = {model_m['count_overall']}")

            chart_df = pd.DataFrame({
                "epoch": ev["epoch_pred"],
                "Predicted next price": ev["predicted_price"],
                "Actual next price": ev["actual_next_price"],
            }).set_index("epoch")
            chart_df = chart_df.replace([np.inf, -np.inf], np.nan).dropna()

            if not chart_df.empty:
                st.line_chart(chart_df.tail(max(eval_window, 50)), height=260)
            else:
                st.info("No valid predicted/actual pairs yet—waiting for next tick after predictions.")
            res = finite_series(ev.set_index("epoch_pred")["error"]).tail(max(eval_window, 50))
            if not res.empty:
                st.markdown("**Residuals over time (prediction − actual next)**")
                st.line_chart(res, height=160)
            else:
                st.info("No residuals to plot yet.")

            hist_vals = finite_series(ev["error"].tail(max(eval_window, 200)))
            if not hist_vals.empty:
                st.markdown("**Residual distribution**")
                fig_h, ax_h = plt.subplots(figsize=(5, 3))
                ax_h.hist(hist_vals, bins=30, alpha=0.8, color="#4C78A8")
                ax_h.axvline(0.0, color="red", linestyle="--", linewidth=1)
                ax_h.set_xlabel("Error (predicted − actual)")
                ax_h.set_ylabel("Count")
                st.pyplot(fig_h, clear_figure=True)
            else:
                st.info("No residuals available for histogram yet.")
            sub = ev.tail(max(eval_window, 200)).copy()
            sub = sub.replace([np.inf, -np.inf], np.nan).dropna(subset=["actual_next_price", "predicted_price"])

            if not sub.empty:
                st.markdown("**Predicted vs Actual (scatter)**")
                fig_s, ax_s = plt.subplots(figsize=(5, 4))
                ax_s.scatter(sub["actual_next_price"], sub["predicted_price"], s=12, alpha=0.6)
                mn = float(min(sub["actual_next_price"].min(), sub["predicted_price"].min()))
                mx = float(max(sub["actual_next_price"].max(), sub["predicted_price"].max()))
                ax_s.plot([mn, mx], [mn, mx], "r--", linewidth=1)
                ax_s.set_xlabel("Actual next price")
                ax_s.set_ylabel("Predicted next price")
                st.pyplot(fig_s, clear_figure=True)
            else:
                st.info("Not enough valid predicted/actual points for scatter yet.")
        else:
            st.info("Waiting for enough predictions and quotes to compute evaluation…")
    else:
        st.info("Waiting for predictions and quotes…")
    with st.expander("Raw Quotes", expanded=False):
        st.dataframe(qdf.tail(50), use_container_width=True)
    with st.expander("Raw Predictions", expanded=False):
        st.dataframe(pdf.tail(50), use_container_width=True)

with col_right:
    st.subheader("What-if Scenario")
    latest_price = float(qdf["price"].iloc[-1]) if not qdf.empty else np.nan
    st.write(f"Latest observed price: **{latest_price if not np.isnan(latest_price) else 'N/A'}**")
    pct = st.slider("Adjust last price by %", -2.0, 2.0, 0.0, 0.1)

    if not qdf.empty and not np.isnan(latest_price):
        qdf_adj = qdf.copy()
        qdf_adj.loc[qdf_adj.index[-1], "price"] = latest_price * (1 + pct / 100.0)
        pred, info = st.session_state.model.predict_next(qdf_adj)
        if pred is not None:
            st.metric("Scenario next price", f"${pred:,.2f}")
            st.caption(f"Δ vs last: {pred - qdf_adj['price'].iloc[-1]:+.4f}")
        else:
            st.info("Model not ready yet—collecting more data…")

st.sidebar.markdown("---")
with st.sidebar.expander("How to run", expanded=False):
    st.write(
        """
        1. Start Kafka locally (e.g., via Docker compose).
        2. Create topics: `python setup_kafka.py`.
        3. Export your Finnhub key: `export FINNHUB_API_KEY=...`.
        4. Start producer: `python producer.py --symbol AMC --interval 15`.
        5. Start consumer-model: `python consumer_processor.py`.
        6. Launch this UI: `streamlit run streamlit_app.py`.
        """
    )