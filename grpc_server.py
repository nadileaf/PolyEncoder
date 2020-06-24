import os
from concurrent import futures

import grpc

import poly_encoders_pb2_grpc
from application_service import ApplicationService
from poly_encoders_pb2 import CandidatesEmbeddingResponse, QueriesEmbeddingResponse, Embedding
from poly_encoders_pb2_grpc import ControllerServicer

application_service = ApplicationService()


class Controller(ControllerServicer):
    def EmbedCandidates(self, request, context):
        texts = request.formatted_texts
        embeddings = []
        for vector in application_service.embed_candidates(texts):
            embeddings.append(Embedding(vector=list(vector)))

        return CandidatesEmbeddingResponse(embeddings=embeddings)

    def EmbedQueries(self, request, context):
        texts = request.formatted_texts
        embeddings = []
        for vectors in application_service.embed_queries(texts):
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
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    GPRC_MAX_MESSAGE_LENGTH = int(os.getenv('GPRC_MAX_MESSAGE_LENGTH', 50 * 1024 * 1024))
    serve()
