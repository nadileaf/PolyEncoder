# encoding:utf-8
import argparse
import os
import random
import shutil
import time

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertModel, BertConfig, BertTokenizerFast
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

from model import BertPolyDssmModel
from model import SelectionDataset, SelectionSequentialTransform, SelectionJoinTransform


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # if args.n_gpu > 0:
    #   torch.cuda.manual_seed_all(args.seed)


def eval_running_model(dataloader, model, device, tr_loss, nb_tr_steps, epoch, global_step):
    loss_fct = CrossEntropyLoss()
    model.eval()
    eval_loss, eval_hit_times = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    for step, batch in enumerate(dataloader, start=1):
        batch = tuple(t.to(device) for t in batch)
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


def train(bert_model_dir: str,
          train_dir: str,
          output_dir: str,
          max_query_len: int = 512,
          max_candidate_len: int = 512,
          max_history: int = 4,
          max_negative: int = 4,
          train_batch_size: int = 32,
          eval_batch_size: int = 2,
          train_shuffle: bool = False,
          num_train_epochs: int = 3,
          poly_m: int = 16,
          weight_decay: float = 0.01,
          learning_rate: float = 5e-5,
          adam_epsilon: float = 1e-8,
          warmup_steps: int = 200,
          max_grad_norm: int = 1,
          random_seed: int = 1234):
    set_seed(random_seed)
    MODEL_CLASSES = {'bert': (BertConfig, BertTokenizerFast, BertModel)}
    ConfigClass, TokenizerClass, BertModelClass = MODEL_CLASSES['bert']

    ## init dataset and bert model
    tokenizer = TokenizerClass.from_pretrained(os.path.join(bert_model_dir, "vocab.txt"), do_lower_case=True)
    context_transform = SelectionJoinTransform(tokenizer=tokenizer, max_len=max_query_len,
                                               max_history=max_history)
    response_transform = SelectionSequentialTransform(tokenizer=tokenizer, max_len=max_candidate_len,
                                                      max_history=None, pair_last=False)

    print('=' * 80)
    print('Train dir:', train_dir)
    print('Output dir:', output_dir)
    print('=' * 80)
    train_dataset = SelectionDataset(os.path.join(train_dir, 'train.txt'),
                                     context_transform, response_transform, sample_cnt=None,
                                     max_negative=max_negative)
    val_dataset = SelectionDataset(os.path.join(train_dir, 'test.txt'),
                                   context_transform, response_transform, sample_cnt=5000,
                                   max_negative=max_negative)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=train_batch_size, collate_fn=train_dataset.batchify_join_str,
                                  shuffle=train_shuffle)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=eval_batch_size, collate_fn=val_dataset.batchify_join_str,
                                shuffle=False)
    t_total = len(train_dataloader) // train_batch_size * (max(5, num_train_epochs))

    epoch_start = 1
    global_step = 0
    best_eval_loss = float('inf')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    shutil.copyfile(os.path.join(bert_model_dir, 'vocab.txt'), os.path.join(output_dir, 'vocab.txt'))
    shutil.copyfile(os.path.join(bert_model_dir, 'config.json'), os.path.join(output_dir, 'config.json'))
    log_wf = open(os.path.join(output_dir, 'log.txt'), 'a', encoding='utf-8')

    state_save_path = os.path.join(output_dir, 'pytorch_model.bin')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ########################################
    ## build BERT encoder
    ########################################
    bert_config = ConfigClass.from_json_file(os.path.join(bert_model_dir, 'config.json'))
    previous_model_file = os.path.join(bert_model_dir, "pytorch_model.bin")
    print('Loading parameters from', previous_model_file)
    log_wf.write('Loading parameters from %s' % previous_model_file + '\n')
    model_state_dict = torch.load(previous_model_file, map_location="cpu")
    bert = BertModelClass.from_pretrained(bert_model_dir, state_dict=model_state_dict)
    del model_state_dict

    model = BertPolyDssmModel(bert_config, bert=bert, poly_m=poly_m)
    model.to(device)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
    )
    print_freq = 100
    eval_freq = min(len(train_dataloader) // 2, 1000)
    print('Print freq:', print_freq, "Eval freq:", eval_freq)

    for epoch in range(epoch_start, int(num_train_epochs) + 1):
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        with tqdm(total=len(train_dataloader)) as bar:
            for step, batch in enumerate(train_dataloader, start=1):
                model.train()
                optimizer.zero_grad()
                batch = tuple(t.to(device) for t in batch)
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                optimizer.step()
                if global_step < warmup_steps:
                    scheduler.step()
                model.zero_grad()
                optimizer.zero_grad()
                global_step += 1

                if step % print_freq == 0:
                    bar.update(min(print_freq, step))
                    time.sleep(0.02)
                    print(global_step, tr_loss / nb_tr_steps)
                    log_wf.write('%d\t%f\n' % (global_step, tr_loss / nb_tr_steps))

                if global_step % eval_freq == 0:
                    if global_step == 4000:
                        eval_freq *= 2
                        print_freq *= 2
                    if global_step == 16000:
                        eval_freq *= 2
                        print_freq *= 2

                    scheduler.step()
                    val_result = eval_running_model(
                        dataloader=val_dataloader,
                        model=model,
                        device=device,
                        tr_loss=tr_loss,
                        nb_tr_steps=nb_tr_steps,
                        epoch=epoch,
                        global_step=global_step
                    )
                    print('Global Step %d VAL res:\n' % global_step, val_result)
                    log_wf.write('Global Step %d VAL res:\n' % global_step)
                    log_wf.write(str(val_result) + '\n')

                    if val_result['eval_loss'] < best_eval_loss:
                        best_eval_loss = val_result['eval_loss']
                        val_result['best_eval_loss'] = best_eval_loss
                        # save model
                        print('[Saving at]', state_save_path)
                        log_wf.write('[Saving at] %s\n' % state_save_path)
                        torch.save(model.state_dict(), state_save_path)
                log_wf.flush()
                pass

        # add a eval step after each epoch
        scheduler.step()
        val_result = eval_running_model(
            dataloader=val_dataloader,
            model=model,
            device=device,
            tr_loss=tr_loss,
            nb_tr_steps=nb_tr_steps,
            epoch=epoch,
            global_step=global_step
        )
        print('Epoch %d, Global Step %d VAL res:\n' % (epoch, global_step), val_result)
        log_wf.write('Global Step %d VAL res:\n' % global_step)
        log_wf.write(str(val_result) + '\n')

        if val_result['eval_loss'] < best_eval_loss:
            best_eval_loss = val_result['eval_loss']
            val_result['best_eval_loss'] = best_eval_loss
            # save model
            print('[Saving at]', state_save_path)
            log_wf.write('[Saving at] %s\n' % state_save_path)
            torch.save(model.state_dict(), state_save_path)
        print(global_step, tr_loss / nb_tr_steps)
        log_wf.write('%d\t%f\n' % (global_step, tr_loss / nb_tr_steps))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ## Required parameters
    # parser.add_argument("--bert_model", default='ckpt/pretrained/distilbert-base-uncased', type=str)
    # parser.add_argument("--model_type", default='distilbert', type=str)
    parser.add_argument("--bert_model_dir", default='ckpt/pretrained/bert-small-uncased', type=str)
    # parser.add_argument("--model_type", default='bert', type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--train_dir", default='data/ubuntu_data', type=str)

    # parser.add_argument("--use_pretrain", action="store_true")
    parser.add_argument("--train_shuffle", action="store_true")
    # parser.add_argument("--architecture", required=True, type=str, help='[poly, bi]')

    parser.add_argument("--max_contexts_length", default=128, type=int)
    parser.add_argument("--max_response_length", default=64, type=int)
    parser.add_argument("--train_batch_size", default=32, type=int, help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=2, type=int, help="Total batch size for eval.")
    # parser.add_argument("--print_freq", default=100, type=int, help="Total batch size for eval.")

    parser.add_argument("--poly_m", default=16, type=int, help="Total batch size for eval.")
    parser.add_argument("--max_history", default=4, type=int, help="Total batch size for eval.")
    parser.add_argument("--max_negative", default=5, type=int, help="Max negative size for response batch.")

    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--warmup_steps", default=2000, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")

    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--random_seed', type=int, default=12345, help="random seed for initialization")
    # parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
    #                     help="Number of updates steps to accumulate before performing a backward/update pass.")
    # parser.add_argument(
    #     "--fp16",
    #     action="store_true",
    #     help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    # )
    # parser.add_argument(
    #     "--fp16_opt_level",
    #     type=str,
    #     default="O1",
    #     help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
    #          "See details at https://nvidia.github.io/apex/amp.html",
    # )
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    print(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % args.gpu
    train(
        bert_model_dir=args.bert_model_dir,
        max_query_len=args.max_contexts_length,
        max_candidate_len=args.max_response_length,
        max_history=args.max_history,
        train_dir=args.train_dir,
        output_dir=args.output_dir,
        max_negative=args.max_negative,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        train_shuffle=args.train_shuffle,
        num_train_epochs=args.num_train_epochs,
        poly_m=args.poly_m,
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,
        adam_epsilon=args.adam_epsilon,
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.max_grad_norm,
        random_seed=args.random_seed
    )
