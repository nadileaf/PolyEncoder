import os

MODEL_DIR = os.getenv('MODEL_DIR', 'result')
POLY_M = int(os.getenv('POLY_M', 16))
MAX_QUERY_LEN = int(os.getenv('MAX_QUERY_LEN', 256))
MAX_CANDIDATE_LEN = int(os.getenv('MAX_CANDIDATE_LEN', 256))
RANDOM_SEED = int(os.getenv('RANDOM_SEED', 12345))

os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % int(os.getenv('CUDA_VISIBLE_DEVICES', 0))
