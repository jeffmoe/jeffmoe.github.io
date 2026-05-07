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
