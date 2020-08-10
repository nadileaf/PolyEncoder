from dataclasses import dataclass


@dataclass
class PolyEncodersTrainingTriggeredEvent:
    bert_model_dir: str
    max_query_len: int
    max_candidate_len: int
    max_history: int
    train_dir: str
    output_dir: str
    max_negative: int
    train_batch_size: int
    eval_batch_size: int
    train_shuffle: bool
    num_train_epochs: int
    poly_m: int
    weight_decay: float
    learning_rate: float
    adam_epsilon: float
    warmup_steps: int
    max_grad_norm: int
    random_seed: int


@dataclass
class PolyEncodersTrainingCompletedMessage:
    model_dir: str
    max_query_len: int
    max_candidate_len: int
    poly_m: int
    random_seed: int
