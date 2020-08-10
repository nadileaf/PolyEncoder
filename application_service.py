import os
import random

import numpy as np
import s3fs
import torch
from torch.utils.data import DataLoader
from transformers import BertModel, BertTokenizer, BertConfig

import config
from model import BertPolyDssmModel, SelectionDataset, SelectionJoinTransform, SelectionSequentialTransform

os.environ['LRU_CACHE_CAPACITY'] = '1'


class Embedding:
    def __init__(self, model, query_transform, candidate_transform):
        self.candidate_transform = candidate_transform
        self.query_transform = query_transform
        self.model = model
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def transform_candidate_text(self, candidate_text: str):
        return self.candidate_transform([candidate_text])

    def transform_query_text(self, query_text: str):
        return self.query_transform([query_text])

    def batch_candidate_to_text(self, texts):
        token_ids_list_batch, segment_ids_list_batch, input_masks_list_batch = [], [], []
        for text in texts:
            input_ids_list, segment_ids_list, input_masks_list, _ = self.transform_candidate_text(text)
            token_ids_list_batch.append(input_ids_list)
            segment_ids_list_batch.append(segment_ids_list)
            input_masks_list_batch.append(input_masks_list)
        token_ids_list_batch = torch.tensor(token_ids_list_batch, dtype=torch.long)
        segment_ids_list_batch = torch.tensor(segment_ids_list_batch, dtype=torch.long)
        input_masks_list_batch = torch.tensor(input_masks_list_batch, dtype=torch.long)
        return token_ids_list_batch, segment_ids_list_batch, input_masks_list_batch

    def batch_query_to_text(self, texts):
        token_ids_list_batch, segment_ids_list_batch, input_masks_list_batch = [], [], []
        for text in texts:
            input_ids_list, segment_ids_list, input_masks_list = self.transform_query_text(text)
            token_ids_list_batch.append(input_ids_list)
            segment_ids_list_batch.append(segment_ids_list)
            input_masks_list_batch.append(input_masks_list)
        token_ids_list_batch = torch.tensor(token_ids_list_batch, dtype=torch.long)
        segment_ids_list_batch = torch.tensor(segment_ids_list_batch, dtype=torch.long)
        input_masks_list_batch = torch.tensor(input_masks_list_batch, dtype=torch.long)
        return token_ids_list_batch, segment_ids_list_batch, input_masks_list_batch

    def embed_candidates(self, candidate_texts):
        with torch.no_grad():
            token_ids_list_batch, segment_ids_list_batch, input_masks_list_batch = tuple(
                t.to(self.device) for t in self.batch_candidate_to_text(candidate_texts)
            )
            embeddings = self.model.embed_response(token_ids_list_batch, segment_ids_list_batch, input_masks_list_batch)
            result = embeddings.cpu().detach().numpy()
            del embeddings
            return result[:, 0, :]

    def embed_queries(self, query_texts):
        with torch.no_grad():
            token_ids_list_batch, segment_ids_list_batch, input_masks_list_batch = tuple(
                t.to(self.device) for t in self.batch_query_to_text(query_texts)
            )
            embeddings = self.model.embed_context(token_ids_list_batch, segment_ids_list_batch, input_masks_list_batch)
            result = embeddings.cpu().detach().numpy()
            del embeddings
            return result


def slicing_list(l, n):
    bs = len(l) // n
    for i in range(bs):
        yield l[i * n:n * (i + 1)]
    if bs * n < len(l):
        yield l[bs * n:n * (bs + 1)]


class ApplicationService:
    def __init__(self,
                 model_dir: str,
                 poly_m: int,
                 max_query_len: int,
                 max_candidate_len: int,
                 random_seed: int):
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vocab_path = os.path.join(model_dir, "vocab.txt")
        config_path = os.path.join(model_dir, 'config.json')
        model_path = os.path.join(model_dir, "pytorch_model.bin")
        if model_dir.startswith("s3://"):
            vocab_path = vocab_path.lstrip("s3://")
            config_path = config_path.lstrip("s3://")
            model_path = model_path.lstrip("s3://")
            model_dir = model_dir.lstrip("s3://")
            s3 = s3fs.S3FileSystem()
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            print(f"Downloading [{vocab_path}] from s3...")
            with s3.open(vocab_path, 'rb') as fin:
                with open(vocab_path, 'wb') as fout:
                    fout.write(fin.read())
            print(f"Downloading [{config_path}] from s3...")
            with s3.open(config_path, 'rb') as fin:
                with open(config_path, 'wb') as fout:
                    fout.write(fin.read())
            print(f"Downloading [{model_path}] from s3...")
            with s3.open(model_path, 'rb') as fin:
                with open(model_path, 'wb') as fout:
                    fout.write(fin.read())

        tokenizer = BertTokenizer.from_pretrained(vocab_path, do_lower_case=True)
        bert_config = BertConfig.from_json_file(config_path)
        print('Loading parameters from', model_path)
        model_state_dict = torch.load(model_path, map_location="cpu")
        bert = BertModel.from_pretrained(model_dir, state_dict=model_state_dict)
        del model_state_dict

        self.model = BertPolyDssmModel(bert_config, bert=bert, poly_m=poly_m)
        self.device = device
        self.model.to(device)

        ## init dataset and bert model
        self.context_transform = SelectionJoinTransform(tokenizer=tokenizer,
                                                        max_len=max_query_len,
                                                        max_history=4)
        self.response_transform = SelectionSequentialTransform(tokenizer=tokenizer,
                                                               max_len=max_candidate_len,
                                                               max_history=None,
                                                               pair_last=False)

        self.embedding_service = Embedding(model=self.model,
                                           query_transform=self.context_transform,
                                           candidate_transform=self.response_transform)

    def embed_candidates(self, candidate_texts):
        result = []
        for ts in slicing_list(candidate_texts, config.BATCH_SIZE):
            result.extend(self.embedding_service.embed_candidates(ts))
        return result

    def embed_queries(self, query_texts):
        result = []
        for ts in slicing_list(query_texts, config.BATCH_SIZE):
            result.extend(self.embedding_service.embed_queries(ts))
        return result

    def predict_from_file(self,
                          input_file_path,
                          output_file_path):
        raw_dataset = []
        with open(input_file_path) as f:
            for i in f:
                raw_dataset.append(i)
        dataset = SelectionDataset(input_file_path,
                                   self.context_transform,
                                   self.response_transform,
                                   sample_cnt=None)
        dataloader = DataLoader(dataset,
                                batch_size=10,
                                collate_fn=dataset.batchify_join_str,
                                shuffle=False)
        result = []
        self.model.eval()
        for step, batch in enumerate(dataloader, start=1):
            batch = tuple(t.to(self.device) for t in batch)
            context_token_ids_list_batch, context_segment_ids_list_batch, context_input_masks_list_batch, \
            response_token_ids_list_batch, response_segment_ids_list_batch, response_input_masks_list_batch, labels_batch = batch

            with torch.no_grad():
                logits = self.model(context_token_ids_list_batch, context_segment_ids_list_batch,
                                    context_input_masks_list_batch,
                                    response_token_ids_list_batch, response_segment_ids_list_batch,
                                    response_input_masks_list_batch)

            for i in logits:
                result.extend(i.cpu().detach().numpy().tolist())
        with open(output_file_path, 'w') as f:
            for i in zip(result, raw_dataset):
                f.write(str(i[0]) + '\t' + i[1])
