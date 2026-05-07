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
