# encoding:utf-8
import argparse
import os
import random
import time
from contextlib import contextmanager
from datetime import timedelta

import numpy as np
import s3fs
import torch
import torch.distributed as dist
from torch.nn import CrossEntropyLoss
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import BertModel, BertConfig, BertTokenizerFast
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

from model import BertPolyDssmModel, BertDssmModel
from model import SelectionDataset, SelectionSequentialTransform, SelectionJoinTransform


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # if args.n_gpu > 0:
    #   torch.cuda.manual_seed_all(args.seed)


def eval_running_model(dataloader, model, device_id, tr_loss, nb_tr_steps, epoch, global_step):
    loss_fct = CrossEntropyLoss()
    model.eval()
    eval_loss, eval_hit_times = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    for step, batch in enumerate(dataloader, start=1):
        batch = tuple(t.cuda(device_id, non_blocking=True) for t in batch)
        context_token_ids_list_batch, context_segment_ids_list_batch, context_input_masks_list_batch, \
        response_token_ids_list_batch, response_segment_ids_list_batch, response_input_masks_list_batch, labels_batch = batch

        with torch.no_grad():
            logits = model(context_token_ids_list_batch, context_segment_ids_list_batch, context_input_masks_list_batch,
                           response_token_ids_list_batch, response_segment_ids_list_batch,
                           response_input_masks_list_batch)
            loss = loss_fct(logits * 5, torch.argmax(labels_batch, 1))  # 5 is a coef

        eval_hit_times += (logits.argmax(-1) == torch.argmax(labels_batch, 1)).sum().item()
        eval_loss += loss.item()

        nb_eval_examples += labels_batch.size(0)
        nb_eval_steps += 1
    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_hit_times / nb_eval_examples
    result = {
        'train_loss': tr_loss / nb_tr_steps,
        'eval_loss': eval_loss,
        'eval_accuracy': eval_accuracy,

        'epoch': epoch,
        'global_step': global_step,
    }
    return result


class State:
    """
    Container for objects that we want to checkpoint. Represents the
    current "state" of the worker. This object is mutable.
    """

    def __init__(self, model, optimizer):
        self.epoch = -1
        self.best_eval_loss = float('inf')
        self.model = model
        self.optimizer = optimizer
        self.state_dict = None

    def capture_snapshot(self):
        """
        Essentially a ``serialize()`` function, returns the state as an
        object compatible with ``torch.save()``. The following should work
        ::
        snapshot = state_0.capture_snapshot()
        state_1.apply_snapshot(snapshot)
        assert state_0 == state_1
        """
        return {
            "epoch": self.epoch,
            "best_eval_loss": self.best_eval_loss,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def apply_snapshot(self, obj):
        """
        The complimentary function of ``capture_snapshot()``. Applies the
        snapshot object that was returned by ``capture_snapshot()``.
        This function mutates this state object.
        """

        self.epoch = obj["epoch"]
        self.best_eval_loss = obj["best_eval_loss"]
        self.state_dict = obj["state_dict"]
        self.model.load_state_dict(obj["state_dict"])
        self.optimizer.load_state_dict(obj["optimizer"])

    def save(self, f):
        torch.save(self.capture_snapshot(), f)

    def load(self, f, device_id):
        # Map model to be loaded to specified single gpu.
        snapshot = torch.load(f, map_location=f"cuda:{device_id}")
        self.apply_snapshot(snapshot)


@contextmanager
def tmp_process_group(backend):
    cpu_pg = dist.new_group(backend=backend)
    try:
        yield cpu_pg
    finally:
        dist.destroy_process_group(cpu_pg)


def load_checkpoint(
        checkpoint_file: str,
        device_id: int,
        model: DistributedDataParallel,
        optimizer,  # ` SGD
) -> State:
    """
    Loads a local checkpoint (if any). Otherwise, checks to see if any of
    the neighbors have a non-zero state. If so, restore the state
    from the rank that has the most up-to-date checkpoint.
    .. note:: when your job has access to a globally visible persistent storage
              (e.g. nfs mount, S3) you can simply have all workers load
              from the most recent checkpoint from such storage. Since this
              example is expected to run on vanilla hosts (with no shared
              storage) the checkpoints are written to local disk, hence
              we have the extra logic to broadcast the checkpoint from a
              surviving node.
    """

    state = State(model, optimizer)
    s3 = s3fs.S3FileSystem()
    if s3.exists(checkpoint_file):
        with s3.open(checkpoint_file, 'rb') as f:
            print(f"=> loading checkpoint file: {checkpoint_file}")
            state.load(f, device_id)
            print(f"=> loaded checkpoint file: {checkpoint_file}")
        print(f"=> done restoring from previous checkpoint")
    print(f"=> fresh state")
    return state


def cp_to_s3(local_filename, filename):
    s3 = s3fs.S3FileSystem()
    with open(local_filename, 'rb') as fin:
        with s3.open(filename, 'wb') as fout:
            fout.write(fin.read())


def save_checkpoint(state: State, is_best: bool, filename: str):
    checkpoint_dir = os.path.dirname(filename)
    s3 = s3fs.S3FileSystem()

    # save to tmp, then commit by moving the file in case the job
    # gets interrupted while writing the checkpoint
    tmp_filename = filename + ".tmp"
    model_filename = os.path.join(checkpoint_dir, "pytorch_model.bin")
    tmp_model_filename = model_filename + ".tmp"

    with s3.open(tmp_filename, 'wb') as f:
        torch.save(state.capture_snapshot(), f)
    s3.mv(tmp_filename, filename)
    print(f"=> saved checkpoint for epoch {state.epoch} at {filename}")
    if is_best:
        with s3.open(tmp_model_filename, 'wb') as f:
            torch.save(state.capture_snapshot()['state_dict'], f)
        print(f"=> best model found at epoch {state.epoch} saving to {model_filename}")
        s3.mv(tmp_model_filename, model_filename)


def init_model(bert_model_dir,
               architecture,
               train_dataloader,
               poly_m,
               weight_decay,
               lr,
               eps,
               warmup_steps,
               total_steps,
               print_freq):
    # MODEL_CLASSES = {
    #     'bert': (BertConfig, BertTokenizer, BertModel),
    #     'distilbert': (DistilBertConfig, DistilBertTokenizer, DistilBertModel)
    # }
    # ConfigClass, TokenizerClass, BertModelClass = MODEL_CLASSES[args.model_type]

    bert_config = BertConfig.from_json_file(os.path.join(bert_model_dir, 'config.json'))
    previous_model_file = os.path.join(bert_model_dir, "pytorch_model.bin")
    print('Loading parameters from', previous_model_file)
    model_state_dict = torch.load(previous_model_file, map_location="cpu")
    bert = BertModel.from_pretrained(bert_model_dir, state_dict=model_state_dict)
    del model_state_dict

    if architecture == 'poly':
        model = BertPolyDssmModel(bert_config, bert=bert, poly_m=poly_m)
    elif architecture == 'bi':
        model = BertDssmModel(bert_config, bert=bert)
    else:
        raise Exception('Unknown architecture.')

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=eps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    eval_freq = min(len(train_dataloader) // 2, 1000)
    print('Print freq:', print_freq, "Eval freq:", eval_freq)
    return model, optimizer, scheduler, eval_freq


def init_dataloader(train_dir,
                    train_batch_size,
                    eval_batch_size,
                    bert_model_dir,
                    max_history,
                    max_contexts_length,
                    max_response_length,
                    train_epochs):
    tokenizer = BertTokenizerFast.from_pretrained(os.path.join(bert_model_dir, "vocab.txt"), do_lower_case=True)
    context_transform = SelectionJoinTransform(tokenizer=tokenizer, max_len=max_contexts_length,
                                               max_history=max_history)
    response_transform = SelectionSequentialTransform(tokenizer=tokenizer, max_len=max_response_length,
                                                      max_history=None, pair_last=False)
    train_dataset = SelectionDataset(os.path.join(train_dir, 'train.txt'),
                                     context_transform, response_transform, sample_cnt=None)
    val_dataset = SelectionDataset(os.path.join(train_dir, 'valid.txt'),
                                   context_transform, response_transform, sample_cnt=5000)
    test_dataset = SelectionDataset(os.path.join(train_dir, 'test.txt'),
                                    context_transform, response_transform, sample_cnt=5000)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=train_batch_size,
                                  collate_fn=train_dataset.batchify_join_str,
                                  sampler=DistributedSampler(train_dataset, shuffle=True))
    val_dataloader = DataLoader(val_dataset,
                                batch_size=eval_batch_size,
                                collate_fn=val_dataset.batchify_join_str,
                                sampler=DistributedSampler(val_dataset, shuffle=False))
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=eval_batch_size,
                                 collate_fn=test_dataset.batchify_join_str,
                                 sampler=DistributedSampler(test_dataset, shuffle=False))
    total_steps = len(train_dataloader) // train_batch_size * (max(5, train_epochs))
    return train_dataloader, val_dataloader, test_dataloader, total_steps


def train(args):
    print('=' * 80)
    print('Train dir:', args.train_dir)
    print('S3 Output dir:', args.s3_output_dir)
    print('=' * 80)

    # log_wf = open(os.path.join(args.output_dir, 'log.txt'), 'a', encoding='utf-8')

    checkpoint_file = os.path.join(args.s3_output_dir, 'state_checkpoint.bin')
    device_id = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(device_id)
    print(f"=> set cuda device = {device_id}")
    os.environ["NCCL_BLOCKING_WAIT"] = "1"
    dist.init_process_group(
        backend=args.dist_backend, init_method="env://", timeout=timedelta(seconds=10)
    )

    train_dataloader, val_dataloader, test_dataloader, total_steps = init_dataloader(
        train_dir=args.train_dir,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        bert_model_dir=args.bert_model,
        max_history=args.max_history,
        max_contexts_length=args.max_contexts_length,
        max_response_length=args.max_response_length,
        train_epochs=args.num_train_epochs
    )

    model, optimizer, scheduler, eval_freq = init_model(
        bert_model_dir=args.bert_model,
        architecture=args.architecture,
        train_dataloader=train_dataloader,
        poly_m=args.poly_m,
        weight_decay=args.weight_decay,
        lr=args.learning_rate,
        eps=args.adam_epsilon,
        warmup_steps=args.warmup_steps,
        total_steps=total_steps,
        print_freq=args.print_freq
    )
    model.cuda(device_id)
    model = DistributedDataParallel(model,
                                    device_ids=[device_id],
                                    find_unused_parameters=True)

    state = load_checkpoint(
        checkpoint_file,
        device_id=device_id,
        model=model,
        optimizer=optimizer
    )

    epoch_start = state.epoch + 1
    global_step = 0
    best_eval_loss = float('inf')
    cp_to_s3(os.path.join(args.bert_model, 'vocab.txt'), os.path.join(args.s3_output_dir, 'vocab.txt'))
    cp_to_s3(os.path.join(args.bert_model, 'config.json'), os.path.join(args.s3_output_dir, 'config.json'))
    for epoch in range(epoch_start, int(args.num_train_epochs) + 1):
        state.epoch = epoch
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        with tqdm(total=len(train_dataloader)) as bar:
            for step, batch in enumerate(train_dataloader, start=1):
                model.train()
                optimizer.zero_grad()
                batch = tuple(t.cuda(device_id, non_blocking=True) for t in batch)
                context_token_ids_list_batch, context_segment_ids_list_batch, context_input_masks_list_batch, \
                response_token_ids_list_batch, response_segment_ids_list_batch, response_input_masks_list_batch, labels_batch = batch
                loss = model(context_token_ids_list_batch, context_segment_ids_list_batch,
                             context_input_masks_list_batch,
                             response_token_ids_list_batch, response_segment_ids_list_batch,
                             response_input_masks_list_batch,
                             labels_batch)
                tr_loss += loss.item()
                nb_tr_examples += context_token_ids_list_batch.size(0)
                nb_tr_steps += 1

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                if global_step < args.warmup_steps:
                    scheduler.step()
                model.zero_grad()
                optimizer.zero_grad()
                global_step += 1

                if step % args.print_freq == 0:
                    bar.update(min(args.print_freq, step))
                    time.sleep(0.02)
                    print(global_step, tr_loss / nb_tr_steps)

                if global_step % eval_freq == 0:
                    if global_step == 4000:
                        eval_freq *= 2
                        args.print_freq *= 2
                    if global_step == 16000:
                        eval_freq *= 2
                        args.print_freq *= 2

                    scheduler.step()
                    val_result = eval_running_model(
                        dataloader=val_dataloader,
                        model=model,
                        device_id=device_id,
                        tr_loss=tr_loss,
                        nb_tr_steps=nb_tr_steps,
                        epoch=epoch,
                        global_step=global_step
                    )
                    print('Global Step %d VAL res:\n' % global_step, val_result)
                    state.best_eval_loss = min(val_result['eval_loss'], state.best_eval_loss)
                    if device_id == 0:
                        save_checkpoint(state, val_result['eval_loss'] < best_eval_loss, checkpoint_file)
                pass

        # add a eval step after each epoch
        scheduler.step()
        val_result = eval_running_model(
            dataloader=val_dataloader,
            model=model,
            device_id=device_id,
            tr_loss=tr_loss,
            nb_tr_steps=nb_tr_steps,
            epoch=epoch,
            global_step=global_step
        )
        print('Global Step %d VAL res:\n' % global_step, val_result)
        state.best_eval_loss = min(val_result['eval_loss'], state.best_eval_loss)
        if device_id == 0:
            save_checkpoint(state, val_result['eval_loss'] < best_eval_loss, checkpoint_file)
        print(global_step, tr_loss / nb_tr_steps)


def main():
    parser = argparse.ArgumentParser()
    ## Required parameters
    # parser.add_argument("--bert_model", default='ckpt/pretrained/distilbert-base-uncased', type=str)
    # parser.add_argument("--model_type", default='distilbert', type=str)
    parser.add_argument("--bert_model", default='ckpt/pretrained/bert-small-uncased', type=str)
    parser.add_argument("--model_type", default='bert', type=str)
    parser.add_argument("--dist_backend", default='nccl', choices=["nccl", "gloo"], type=str,
                        help="distributed backend")
    parser.add_argument("--s3_output_dir", required=True, type=str)
    parser.add_argument("--train_dir", default='data/ubuntu_data', type=str)

    parser.add_argument("--use_pretrain", action="store_true")
    parser.add_argument("--architecture", required=True, type=str, help='[poly, bi]')

    parser.add_argument("--max_contexts_length", default=128, type=int)
    parser.add_argument("--max_response_length", default=64, type=int)
    parser.add_argument("--train_batch_size", default=32, type=int, help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=2, type=int, help="Total batch size for eval.")
    parser.add_argument("--print_freq", default=100, type=int, help="Total batch size for eval.")

    parser.add_argument("--poly_m", default=16, type=int, help="Total batch size for eval.")
    parser.add_argument("--max_history", default=4, type=int, help="Total batch size for eval.")

    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--warmup_steps", default=2000, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")

    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--seed', type=int, default=12345, help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
             "See details at https://nvidia.github.io/apex/amp.html",
    )
    args = parser.parse_args()
    print(args)
    set_seed(args)
    train(args)


if __name__ == '__main__':
    main()
