import logging
import os

MODEL_DIR = os.getenv('MODEL_DIR', 'result')
POLY_M = int(os.getenv('POLY_M', 16))
MAX_QUERY_LEN = int(os.getenv('MAX_QUERY_LEN', 256))
MAX_CANDIDATE_LEN = int(os.getenv('MAX_CANDIDATE_LEN', 256))
RANDOM_SEED = int(os.getenv('RANDOM_SEED', 12345))
BATCH_SIZE = int(os.getenv('BATCH_SIZE', 5))

KAFKA_SERVER = os.getenv('KAFKA_SERVER', 'localhost:9092')
TRAINING_TASK_TOPIC = os.getenv('TRAINING_TASK_TOPIC', 'training_task')
TRAINING_COMPLETED_TOPIC = os.getenv('TRAINING_COMPLETED_TOPIC', 'training_completed')

os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % int(os.getenv('CUDA_VISIBLE_DEVICES', 0))
LOG_LEVEL = os.getenv('LOG_LEVEL', 'DEBUG')
logging.basicConfig(level=LOG_LEVEL, format='[%(asctime)s] [%(levelname)s] %(name)s - %(message)s')
