import json
import logging
import os
from concurrent import futures

import grpc
from confluent_kafka import Consumer

import config
import poly_encoders_pb2_grpc
from application_service import ApplicationService
from events import PolyEncodersTrainingCompletedMessage
from poly_encoders_pb2 import CandidatesEmbeddingResponse, QueriesEmbeddingResponse, Embedding
from poly_encoders_pb2_grpc import ControllerServicer

logger = logging.getLogger(__name__)
application_services = {
    'default': ApplicationService(
        model_dir=config.MODEL_DIR,
        poly_m=config.POLY_M,
        max_query_len=config.MAX_QUERY_LEN,
        max_candidate_len=config.MAX_CANDIDATE_LEN,
        random_seed=config.RANDOM_SEED
    )
}


class Listener:
    def __init__(self, kafka_server, topic):
        self.topic = topic
        self.kafka_server = kafka_server
        self.consumer = Consumer({
            'bootstrap.servers': kafka_server,
            'group.id': "poly_encoders_server",
            'enable.auto.commit': False,
            'auto.offset.reset': 'earliest',
            'metadata.max.age.ms': 10000
        })
        self.consumer.subscribe([topic])

    def listen(self):
        logger.info(f"Listening on topic: {self.topic}")
        while True:
            msgs = self.consumer.consume(500, timeout=1)
            if msgs is None or len(msgs) == 0:
                continue
            msg = msgs[-1]
            if msg.error():
                logger.error("Consumer error: {}".format(msg.error()))
                continue
            else:
                try:
                    value = msg.value().decode('utf-8')
                    event = PolyEncodersTrainingCompletedMessage(**json.loads(value))
                    application_services['latest'] = ApplicationService(
                        model_dir=event.model_dir,
                        poly_m=event.poly_m,
                        max_query_len=event.max_query_len,
                        max_candidate_len=event.max_candidate_len,
                        random_seed=event.random_seed
                    )
                    logger.debug('Received message: {}'.format(value))
                except Exception as e:
                    logger.exception(e)
                    logger.error(msg.value())
                    raise e

                self.consumer.commit(message=msgs[-1])


def get_latest_application_service():
    return application_services.get('latest') or application_services['default']


class Controller(ControllerServicer):
    def EmbedCandidates(self, request, context):
        texts = request.formatted_texts
        embeddings = []
        for vector in get_latest_application_service().embed_candidates(texts):
            embeddings.append(Embedding(vector=list(vector)))

        return CandidatesEmbeddingResponse(embeddings=embeddings)

    def EmbedQueries(self, request, context):
        texts = request.formatted_texts
        embeddings = []
        for vectors in get_latest_application_service().embed_queries(texts):
            query_embeddings = []
            for vector in vectors:
                query_embeddings.append(Embedding(vector=list(vector)))
            embeddings.append(QueriesEmbeddingResponse.QueryEmbedding(embeddings=query_embeddings))

        return QueriesEmbeddingResponse(embeddings=embeddings)


def serve():
    port = 50051
    print('gRPC server listening at port {}...'.format(port))
    server = grpc.server(futures.ThreadPoolExecutor(),
                         options=[
                             ('grpc.max_send_message_length', GPRC_MAX_MESSAGE_LENGTH),
                             ('grpc.max_receive_message_length', GPRC_MAX_MESSAGE_LENGTH)
                         ])
    poly_encoders_pb2_grpc.add_ControllerServicer_to_server(Controller(), server)
    server.add_insecure_port('[::]:{}'.format(port))
    listener = Listener(
        kafka_server=config.KAFKA_SERVER,
        topic=config.TRAINING_COMPLETED_TOPIC
    )
    server.start()
    listener.listen()
    server.wait_for_termination()


if __name__ == '__main__':
    GPRC_MAX_MESSAGE_LENGTH = int(os.getenv('GPRC_MAX_MESSAGE_LENGTH', 50 * 1024 * 1024))
    serve()
