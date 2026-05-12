---
title: Real‑Time Machine Learning Pipeline
parent: Machine Learning and Artificial Intelligence
nav_order: 2
---

### Overview
This project streams real-time quotes for a stock ticker from Finnhub every **15 seconds**, 
updates an **online ML model every 30 seconds**, publishes predictions to Kafka, and displays 
metrics in a Streamlit dashboard.

### Quickstart

1. **Prerequisites**
   - Python 3.10+
   - Running Kafka cluster (e.g., `localhost:9092`)
   - Finnhub API key: https://finnhub.io/ (set as env var `FINNHUB_API_KEY`)

2. **Install**
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Create Kafka topics**
   ```bash
   python setup_kafka.py --bootstrap-servers localhost:9092
   ```

4. **Start Producer** (fetches from Finnhub every 15s)
   ```bash
   export FINNHUB_API_KEY=YOUR_KEY
   python producer.py --symbol TSLA --interval 15 --bootstrap-servers localhost:9092
   ```

5. **Start Consumer+Model** (updates model every 30s and publishes predictions)
   ```bash
   python consumer_processor.py
   ```

6. **Launch Streamlit UI**
   ```bash
   streamlit run streamlit_app.py
   ```

### Kafka Setup
```python
"""
setup_kafka.py
----------------
Creates Kafka topics required for the AMC real-time pipeline.

Topics:
- amc-quotes: raw quotes from Finnhub REST API
- amc-predictions: model predictions (next-tick price forecast)

Usage:
    python setup_kafka.py --bootstrap-servers localhost:9092

Notes:
- Robust to topic already existing.
- Logs informative messages.
"""
import argparse
import logging
from kafka.admin import KafkaAdminClient, NewTopic # type: ignore
from kafka.errors import TopicAlreadyExistsError # type: ignore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_TOPICS = [
    {"name": "amc-quotes", "partitions": 1, "replication_factor": -1},
    {"name": "amc-predictions", "partitions": 1, "replication_factor": -1},
]


def create_topics(bootstrap_servers: str, topics=DEFAULT_TOPICS) -> None:
    """Create topics if they don't exist.

    Parameters
    ----------
    bootstrap_servers : str
        Kafka bootstrap servers (e.g., "localhost:9092").
    topics : list[dict]
        Each dict has keys: name, partitions, replication_factor.
    """
    try:
        admin_client = KafkaAdminClient(
            bootstrap_servers=bootstrap_servers,
            client_id="kafka-setup"
        )
    except Exception as e:
        logger.error(f"Failed to create Kafka admin client: {e}")
        return

    new_topics = []
    for t in topics:
        new_topics.append(
            NewTopic(
                name=t["name"],
                num_partitions=int(t.get("partitions", 1)),
                replication_factor=int(t.get("replication_factor", -1)),
            )
        )

    try:
        admin_client.create_topics(new_topics=new_topics, validate_only=False)
        for t in topics:
            logger.info(f"Topic '{t['name']}' created successfully")
    except TopicAlreadyExistsError:
        for t in topics:
            logger.info(f"Topic '{t['name']}' already exists")
    except Exception as e:
        logger.error(f"Error creating topics: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create Kafka topics for AMC pipeline")
    parser.add_argument("--bootstrap-servers", default="localhost:9092",
                        help="Kafka bootstrap servers, default: localhost:9092")
    args = parser.parse_args()
    create_topics(args.bootstrap_servers)
```
### Kafka Producer
```python
"""
producer.py
-----------
Kafka producer that fetches NVDA quotes from Finnhub REST API every 15 seconds
and publishes them to the `nvda-quotes` topic.

Security:
- Reads API key from FINNHUB_API_KEY env var (never hard-code secrets).

Usage:
    python producer.py --symbol NVDA --interval 15 --bootstrap-servers localhost:9092

"""
import argparse
import json
import logging
import os
import time
from datetime import datetime, timezone

import requests
from kafka import KafkaProducer # type: ignore
from kafka.errors import KafkaError, NoBrokersAvailable # type: ignore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FINNHUB_BASE = "https://finnhub.io/api/v1/quote"


class StockProducer:
    """Produces real-time quotes to Kafka from Finnhub REST API."""

    def __init__(self, bootstrap_servers: str = "localhost:9092", topic_name: str = "amc-quotes"):
        self.bootstrap_servers = bootstrap_servers
        self.topic_name = topic_name
        self.producer = None

    def connect(self) -> bool:
        """Connect to Kafka with sensible defaults and small linger for batching."""
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                key_serializer=lambda v: str(v).encode("utf-8"),
                acks="all",
                retries=3,
                batch_size=32768,
                linger_ms=10,
                request_timeout_ms=45000,
                api_version_auto_timeout_ms=30000,
                max_block_ms=60000,
            )
            logger.info("Kafka producer connected successfully")
            return True
        except NoBrokersAvailable:
            logger.error("No Kafka brokers available. Start Kafka and try again.")
            return False
        except Exception as e:
            logger.error(f"Failed to connect producer: {e}")
            return False

    def fetch_quote(self, symbol: str, api_key: str) -> dict | None:
        """Fetch latest quote for the symbol from Finnhub.

        Finnhub 'quote' fields:
            c: Current price
            h: High price of the day
            l: Low price of the day
            o: Open price of the day
            pc: Previous close price
            t: Timestamp (Unix time)
        """
        params = {"symbol": symbol.upper(), "token": api_key}
        try:
            resp = requests.get(FINNHUB_BASE, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if not isinstance(data, dict) or "c" not in data or data.get("t") in (0, None):
                logger.warning(f"Unexpected Finnhub quote payload: {data}")
                return None
            ts_epoch = int(data.get("t", 0))
            ts_iso = datetime.fromtimestamp(ts_epoch, tz=timezone.utc).isoformat()
            payload = {
                "symbol": symbol.upper(),
                "price": float(data.get("c", 0.0)),
                "high": float(data.get("h", 0.0)),
                "low": float(data.get("l", 0.0)),
                "open": float(data.get("o", 0.0)),
                "prev_close": float(data.get("pc", 0.0)),
                "epoch": ts_epoch,
                "timestamp": ts_iso,
                "source": "finnhub_quote_api",
                "fetch_timestamp": datetime.now(timezone.utc).isoformat(),
            }
            return payload
        except requests.exceptions.HTTPError as e:
            status = getattr(e.response, "status_code", "?")
            logger.error(f"HTTP error from Finnhub ({status}): {e}")
            return None
        except requests.exceptions.Timeout:
            logger.error("Timeout fetching Finnhub quote")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error fetching Finnhub quote: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching quote: {e}")
            return None

    def stream_quotes(self, symbol: str = "AMC", api_key: str | None = None, interval_seconds: int = 15):
        """Continuously fetch and stream quotes every `interval_seconds` seconds."""
        if api_key is None:
            api_key = os.getenv("FINNHUB_API_KEY")
        if not api_key:
            raise ValueError("FINNHUB_API_KEY not set. Export it as an environment variable.")

        if self.producer is None and not self.connect():
            logger.warning("Producer not connected. Will fetch data but not send to Kafka.")

        logger.info(f"Starting Finnhub quote stream for {symbol} every {interval_seconds}s")
        backoff = 5
        while True:
            payload = self.fetch_quote(symbol, api_key)
            if payload is not None:
                key = f"{symbol}-{payload['epoch']}"
                if self.producer:
                    try:
                        fut = self.producer.send(self.topic_name, key=key, value=payload)
                        meta = fut.get(timeout=10)
                        logger.info(
                            f"Sent {symbol} @ {payload['price']} (t={payload['epoch']}) to partition {meta.partition}, offset {meta.offset}"
                        )
                    except KafkaError as e:
                        logger.error(f"Kafka error sending data: {e}")
                else:
                    logger.info(f"Fetched (not sent) {symbol} @ {payload['price']}")
                backoff = 5
            else:
                time.sleep(backoff)
                backoff = min(backoff * 2, 60)
                continue

            time.sleep(max(1, int(interval_seconds)))

    def close(self):
        if self.producer:
            try:
                self.producer.flush(timeout=10)
                self.producer.close(timeout=10)
                logger.info("Producer closed")
            except Exception as e:
                logger.error(f"Error closing producer: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stream Finnhub quotes to Kafka")
    parser.add_argument("--symbol", default="AMC", help="Ticker symbol, default: AMC")
    parser.add_argument("--interval", type=int, default=15, help="Fetch interval in seconds (>=15 recommended)")
    parser.add_argument("--bootstrap-servers", default="localhost:9092", help="Kafka bootstrap servers")
    args = parser.parse_args()

    sp = StockProducer(bootstrap_servers=args.bootstrap_servers)
    try:
        sp.stream_quotes(symbol=args.symbol, interval_seconds=args.interval)
    except KeyboardInterrupt:
        logger.info("Interrupted by user. Shutting down...")
    finally:
        sp.close()
```

### Kafka Consumer
```python
"""
consumer_processor.py
---------------------
Consumes AMC quotes from Kafka, maintains an in-memory buffer, updates an
online model every 30 seconds, and publishes predictions to `amc-predictions`.

Also exposes a helper to return a pandas DataFrame of the buffered data.
"""
import json
import logging
import time
from collections import deque
from datetime import datetime, timezone
import pandas as pd
from kafka import KafkaConsumer, KafkaProducer # type: ignore
from kafka.errors import KafkaError # type: ignore
from model_utils import OnlinePricePredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StockConsumer:
    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        topic_name: str = "amc-quotes",
        predictions_topic: str = "amc-predictions",
        buffer_size: int = 2000,
    ):
        self.bootstrap_servers = bootstrap_servers
        self.topic_name = topic_name
        self.predictions_topic = predictions_topic
        self.data_buffer = deque(maxlen=buffer_size)
        self.consumer: KafkaConsumer | None = None
        self.producer: KafkaProducer | None = None
        self.consumer_timeout = 0
        self.model = OnlinePricePredictor()
        self._last_model_update_ts = 0.0
        self.update_interval_seconds = 30

    def connect(self) -> bool:
        try:
            self.consumer = KafkaConsumer(
                self.topic_name,
                bootstrap_servers=self.bootstrap_servers,
                auto_offset_reset="latest",
                enable_auto_commit=True,
                group_id="amc-consumer-group",
                value_deserializer=lambda x: json.loads(x.decode("utf-8")),
                key_deserializer=lambda x: x.decode("utf-8") if x else None,
                max_poll_records=200,
                fetch_max_wait_ms=3000,
                request_timeout_ms=45000,
                session_timeout_ms=20000,
                heartbeat_interval_ms=6000,
                api_version_auto_timeout_ms=30000,
            )
            self.producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                key_serializer=lambda v: str(v).encode("utf-8"),
                acks="all",
                retries=3,
                linger_ms=10,
            )
            logger.info("Kafka consumer & producer connected successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to connect consumer/producer: {e}")
            return False

    def _buffer_to_df(self) -> pd.DataFrame:
        if not self.data_buffer:
            return pd.DataFrame(columns=["symbol", "price", "epoch", "timestamp"])
        df = pd.DataFrame(list(self.data_buffer))
        df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce").astype("Int64")
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        df = df.dropna(subset=["price", "epoch"]).reset_index(drop=True)
        return df

    def _try_model_update(self, now_ts: float):
        if (now_ts - self._last_model_update_ts) < self.update_interval_seconds:
            return
        df = self._buffer_to_df()
        used = self.model.partial_update(df)
        if used > 0:
            self._last_model_update_ts = now_ts
            logger.info(f"Model updated with {used} samples. is_fitted={self.model.is_fitted}")
        else:
            logger.debug("Model update skipped (insufficient data)")

    def _publish_prediction(self, symbol: str, df: pd.DataFrame):
        pred, info = self.model.predict_next(df)
        if pred is None:
            logger.debug(f"Prediction skipped: {info}")
            return
        last_epoch = int(df["epoch"].iloc[-1]) if not df.empty else int(time.time())
        payload = {
            "symbol": symbol,
            "predicted_price": float(pred),
            "epoch": last_epoch,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "meta": info,
        }
        try:
            if self.producer is not None:
                key = f"{symbol}-{payload['epoch']}"
                fut = self.producer.send(self.predictions_topic, key=key, value=payload)
                meta = fut.get(timeout=10)
                logger.info(
                    f"Published prediction to {self.predictions_topic} "
                    f"p={meta.partition} o={meta.offset}: {payload['predicted_price']:.4f}"
                )
        except KafkaError as e:
            logger.error(f"Kafka error publishing prediction: {e}")

    def run_forever(self):
        if (self.consumer is None or self.producer is None) and not self.connect():
            logger.error("Could not connect to Kafka. Exiting.")
            return

        logger.info(f"Consuming messages from topic '{self.topic_name}'...")
        try:
            for message in self.consumer:
                if message is None:
                    continue
                data = message.value
                if not isinstance(data, dict) or "symbol" not in data or "price" not in data or "epoch" not in data:
                    logger.debug(f"Skipping malformed record: {data}")
                    continue
                self.data_buffer.append(data)
                logger.info(f"Consumed {data.get('symbol')} @ {data.get('price')} (epoch={data.get('epoch')})")

                now_ts = time.time()
                self._try_model_update(now_ts)

                df = self._buffer_to_df()
                self._publish_prediction(symbol=data["symbol"], df=df)
        except Exception as e:
            logger.error(f"Unexpected error while consuming: {e}")
        finally:
            if self.consumer:
                try:
                    self.consumer.close()
                except Exception:
                    pass
            if self.producer:
                try:
                    self.producer.flush()
                    self.producer.close()
                except Exception:
                    pass
            logger.info("Consumer closed")

    def get_dataframe(self) -> pd.DataFrame:
        """Return the buffered quotes as a DataFrame."""
        return self._buffer_to_df()


if __name__ == "__main__":
    sc = StockConsumer()
    try:
        sc.run_forever()
    except KeyboardInterrupt:
        logger.info("Interrupted by user. Bye.")
```
### ML Model
```python
"""
model_utils.py
--------------
Online model utilities for next-tick stock price prediction using scikit-learn.

We use an incremental learning approach with SGDRegressor and StandardScaler
so the model can be updated every 30 seconds efficiently.

Target: 1-step-ahead price change (delta) so that prediction = last_price + delta_hat.
Features:
- lag returns (1, 2, 3 lags)
- rolling mean/std of returns (window 5, 10)
- price momentum (price - rolling mean price)
- time-of-day cyclical encoding (sin/cos of seconds since midnight)

Notes:
- The model trains only when enough samples are available (> min_batch).
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


@dataclass
class OnlinePricePredictor:
    random_state: int = 42
    max_iter: int = 1 
    eta0: float = 0.01
    l2: float = 1e-4
    min_batch: int = 20
    feature_lags: int = 3
    price_column: str = "price"

    scaler: StandardScaler = field(default_factory=StandardScaler)
    model: SGDRegressor = field(default_factory=lambda: SGDRegressor(
        loss="squared_error", penalty="l2", alpha=1e-4, learning_rate="constant", eta0=0.01, random_state=42
    ))
    is_fitted: bool = False

    def _feature_engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute features and next-step target.

        Expects df to contain columns: 'timestamp' (ISO) and 'price'.
        Returns a DataFrame with engineered features and target 'y'.
        """
        df = df.copy()
        df = df.sort_values("epoch").reset_index(drop=True)

        df["log_price"] = np.log(df[self.price_column].clip(lower=1e-8))
        df["ret1"] = df["log_price"].diff(1)
        for lag in range(1, self.feature_lags + 1):
            df[f"ret_lag{lag}"] = df["ret1"].shift(lag)

        df["ret_mean5"] = df["ret1"].rolling(5).mean()
        df["ret_std5"] = df["ret1"].rolling(5).std()
        df["ret_mean10"] = df["ret1"].rolling(10).mean()
        df["ret_std10"] = df["ret1"].rolling(10).std()
        df["price_ma10"] = df[self.price_column].rolling(10).mean()
        df["mom10"] = df[self.price_column] - df["price_ma10"]


        sod = (df["epoch"] % 86400).astype(float)
        df["sod_sin"] = np.sin(2 * np.pi * sod / 86400.0)
        df["sod_cos"] = np.cos(2 * np.pi * sod / 86400.0)

        df["y"] = df[self.price_column].diff(-1) * -1

        feature_cols = [
            *(f"ret_lag{lag}" for lag in range(1, self.feature_lags + 1)),
            "ret_mean5", "ret_std5", "ret_mean10", "ret_std10", "mom10",
            "sod_sin", "sod_cos",
        ]
        model_df = df.dropna(subset=feature_cols + ["y"]).copy()
        model_df = model_df.reset_index(drop=True)
        return model_df

    def partial_update(self, df: pd.DataFrame) -> int:
        """Update the model incrementally using available history.

        Returns number of samples used for this update.
        """
        model_df = self._feature_engineer(df)
        if len(model_df) < self.min_batch:
            return 0
        X = model_df.drop(columns=["y", "price", "log_price", "price_ma10"]).select_dtypes(include=[np.number]).values
        y = model_df["y"].values
        self.scaler.partial_fit(X)
        Xs = self.scaler.transform(X)

        if not self.is_fitted:
            self.model.partial_fit(Xs, y)
            self.is_fitted = True
        else:
            self.model.partial_fit(Xs, y)
        return len(model_df)

    def predict_next(self, df: pd.DataFrame) -> tuple[float | None, dict]:
        """Predict next price given the latest observations.

        Returns (predicted_price, debug_info)
        """
        if not self.is_fitted or len(df) < self.feature_lags + 11:
            return None, {"reason": "insufficient_data_or_model_not_fitted"}

        fe = self._feature_engineer(df)
        if fe.empty:
            return None, {"reason": "no_features"}

        last_row = fe.iloc[[-1]]
        X = last_row.drop(columns=["y", "price", "log_price", "price_ma10"]).select_dtypes(include=[np.number]).values
        Xs = self.scaler.transform(X)
        delta_hat = float(self.model.predict(Xs)[0])
        last_price = float(df[self.price_column].iloc[-1])
        return last_price + delta_hat, {"last_price": last_price, "delta_hat": delta_hat}
```
### Streamlit
```python
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
```
### Outcomes
- Fully streaming ML pipeline
- Real‑time prediction dashboard
- Strong back‑end and front‑end integration
- [Project Link](https://github.com/jeffmoe/jeffmoe.github.io/tree/main/Project%20Docs/Real-Time%20Machine%20Learning%20Pipeline)

---
