import json
import logging
import os
from dataclasses import asdict

from confluent_kafka import Consumer, Producer

import config
from events import PolyEncodersTrainingTriggeredEvent, PolyEncodersTrainingCompletedMessage
from train import train

logger = logging.getLogger(__name__)
ERROR_TOLERANCE = os.getenv('ERROR_TOLERANCE', 'all')


class TrainingTaskListener:
    def __init__(self,
                 kafka_server: str,
                 training_task_topic: str,
                 training_completed_topic: str):
        self.training_completed_topic = training_completed_topic
        self.training_task_topic = training_task_topic
        self.consumer = Consumer({
            'bootstrap.servers': kafka_server,
            'group.id': "poly_encoders_model_training",
            'enable.auto.commit': False,
            'auto.offset.reset': 'earliest',
            'metadata.max.age.ms': 10000
        })
        self.producer = Producer({
            'bootstrap.servers': kafka_server
        })

        self.consumer.subscribe([training_task_topic])

    def delivery_report(self, err, msg):
        """ Called once for each message produced to indicate delivery result.
            Triggered by poll() or flush(). """
        if err is not None:
            logger.error('Message delivery failed: {}'.format(err))
            raise err
        else:
            logger.debug('Message delivered to {} [{}]'.format(msg.topic(), msg.partition()))

    def listen(self):
        logger.info(f"Listening on topic: {self.training_task_topic}")
        while True:
            msgs = self.consumer.consume(1, timeout=1)
            if msgs is None or len(msgs) == 0:
                continue
            for msg in msgs:
                if msg.error():
                    logger.error("Consumer error: {}".format(msg.error()))
                    continue
                else:
                    try:
                        value = msg.value().decode('utf-8')
                        event = PolyEncodersTrainingTriggeredEvent(**json.loads(value))
                        logger.debug('Received message: {}'.format(value))
                        train(**asdict(event))
                        self.consumer.commit(message=msgs[-1])
                        self.producer.poll(0)
                        logger.debug(f"Producing to topic: {self.training_completed_topic}...")
                        completed_event = PolyEncodersTrainingCompletedMessage(
                            model_dir=event.output_dir,
                            max_query_len=event.max_query_len,
                            max_candidate_len=event.max_candidate_len,
                            poly_m=event.poly_m,
                            random_seed=event.random_seed
                        )
                        self.producer.produce(
                            self.training_completed_topic,
                            asdict(completed_event),
                            callback=self.delivery_report
                        )
                    except Exception as e:
                        logger.exception(e)
                        logger.error(msg.value())
                        if ERROR_TOLERANCE == 'none':
                            raise e

        self.producer.flush()
        self.consumer.close()


if __name__ == '__main__':
    listener = TrainingTaskListener(
        kafka_server=config.KAFKA_SERVER,
        training_task_topic=config.TRAINING_TASK_TOPIC,
        training_completed_topic=config.TRAINING_COMPLETED_TOPIC
    )
    listener.listen()
