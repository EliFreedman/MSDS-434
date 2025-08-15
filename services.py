import json
import os

from kafka import KafkaConsumer, KafkaProducer

KAFKA_BROKER = os.environ.get("KAFKA_BROKER", "localhost:9092")
PREDICTION_TOPIC = os.environ.get("PREDICTION_TOPIC", "url_predictions")


def get_producer():
    return KafkaProducer(
        bootstrap_servers=KAFKA_BROKER,
        value_serializer=lambda v: json.dumps(v).encode("utf-8")
    )


def publish_prediction(prediction: dict):
    """
    Publishes a prediction result to the Kafka topic.
    Args:
        prediction (dict): The prediction result to publish.
    """
    producer = get_producer()
    producer.send(PREDICTION_TOPIC, prediction)
    producer.flush()


def get_consumer(topic=PREDICTION_TOPIC, group_id="url-prediction-group"):
    return KafkaConsumer(
        topic,
        bootstrap_servers=KAFKA_BROKER,
        group_id=group_id,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        auto_offset_reset="earliest",
        enable_auto_commit=True
    )


def consume_predictions():
    """
    Consumes prediction results from the Kafka topic.
    Yields:
        dict: The prediction result.
    """
    consumer = get_consumer()
    for message in consumer:
        yield message.value
