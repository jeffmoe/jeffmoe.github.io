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
