import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import BertModel, BertTokenizer, BertConfig

import config
from model import BertPolyDssmModel, BertDssmModel, SelectionDataset, SelectionJoinTransform, \
    SelectionSequentialTransform

seed = config.RANDOM_SEED

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


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
        self.model.eval()
        token_ids_list_batch, segment_ids_list_batch, input_masks_list_batch = tuple(
            t.to(self.device) for t in self.batch_candidate_to_text(candidate_texts)
        )
        embeddings = self.model.embed_response(token_ids_list_batch, segment_ids_list_batch, input_masks_list_batch)
        embeddings = embeddings.cpu().detach().numpy()
        return embeddings[: 0, :]

    def embed_queries(self, query_texts):
        self.model.eval()
        token_ids_list_batch, segment_ids_list_batch, input_masks_list_batch = tuple(
            t.to(self.device) for t in self.batch_query_to_text(query_texts)
        )
        embeddings = self.model.embed_context(token_ids_list_batch, segment_ids_list_batch, input_masks_list_batch)
        embeddings = embeddings.cpu().detach().numpy()
        return embeddings


class ApplicationService:
    def __init__(self):
        bert_model_dir = config.MODEL_DIR
        architecture = 'poly'
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ConfigClass, TokenizerClass, BertModelClass = BertConfig, BertTokenizer, BertModel

        bert_config = ConfigClass.from_json_file(os.path.join(bert_model_dir, 'config.json'))
        previous_model_file = os.path.join(bert_model_dir, "pytorch_model.bin")
        print('Loading parameters from', previous_model_file)
        model_state_dict = torch.load(previous_model_file, map_location="cpu")
        bert = BertModelClass.from_pretrained(bert_model_dir, state_dict=model_state_dict)
        del model_state_dict

        if architecture == 'poly':
            self.model = BertPolyDssmModel(bert_config, bert=bert, poly_m=config.POLY_M)
        elif architecture == 'bi':
            self.model = BertDssmModel(bert_config, bert=bert)
        else:
            raise Exception('Unknown architecture.')
        self.device = device
        self.model.to(device)

        MODEL_CLASSES = {
            'bert': (BertConfig, BertTokenizer, BertModel),
        }
        ConfigClass, TokenizerClass, BertModelClass = MODEL_CLASSES['bert']

        ## init dataset and bert model
        tokenizer = TokenizerClass.from_pretrained(os.path.join(bert_model_dir, "vocab.txt"), do_lower_case=True)
        self.context_transform = SelectionJoinTransform(tokenizer=tokenizer,
                                                        max_len=config.MAX_QUERY_LEN,
                                                        max_history=4)
        self.response_transform = SelectionSequentialTransform(tokenizer=tokenizer,
                                                               max_len=config.MAX_CANDIDATE_LEN,
                                                               max_history=None,
                                                               pair_last=False)

        self.embedding_service = Embedding(model=self.model,
                                           query_transform=self.context_transform,
                                           candidate_transform=self.response_transform)

    def embed_candidates(self, candidate_texts):
        self.embedding_service.embed_candidates(candidate_texts)

    def embed_queries(self, query_texts):
        self.embedding_service.embed_queries(query_texts)

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


def predict_test():
    bert_model_dir = 'result'
    test_file_path = '../data/poly_encoder_data_pair/test.txt'
    output_file_path = 'test_output/output.txt'

    os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    application_service = ApplicationService(bert_model_dir=bert_model_dir,
                                             architecture='poly',
                                             device=device,
                                             poly_m=16)
    application_service.predict_from_file(test_file_path, output_file_path)