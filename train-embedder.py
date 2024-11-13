import argparse
import sys
import os
import torch
from gensim.models import Word2Vec, KeyedVectors
import gensim.downloader as api_gen
import numpy as np
from tqdm import tqdm
from collections import deque
from torch import nn
import random
from transformers import AutoTokenizer, AutoModel
from torch import Tensor
import torch.nn.functional as F
from peft import get_peft_model, LoraConfig
from itertools import islice
from src.utils import evaluator

def batch_iterable(iterable, batch_size):
    it = iter(iterable)
    while True:
        batch = list(islice(it, batch_size))
        if not batch:
            break
        yield batch

def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

class Retriever(torch.nn.Module):
    def __init__(self, input_size):
        super(Retriever, self).__init__()
        self.input_layer = nn.Linear(input_size, 1024)
        self.hidden_layers = nn.ModuleList([nn.Linear(1024, 1024) for _ in range(4)])
        self.output_layer = nn.Linear(1024, input_size)
        self.activation = nn.GELU()
        
    def forward(self, x):
        x = self.activation(self.input_layer(x))
        for layer in self.hidden_layers:
            residual = x
            x = self.activation(layer(x)) + residual
        x = self.output_layer(x)
        return x
    
class TransformerDecoderNetwork(nn.Module):
    def __init__(self, input_size, num_layers=4, nhead=8, dim_feedforward=2048, dropout=0.1):
        super(TransformerDecoderNetwork, self).__init__()
        self.embedding = nn.Linear(input_size, dim_feedforward)
        decoder_layer = nn.TransformerDecoderLayer(d_model=dim_feedforward, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(dim_feedforward, input_size)
        
    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer_decoder(x, x)
        x = self.fc_out(x)
        return x
    
class AwesomeEMBEDDER(nn.Module):
    ##########make a model with nn.embedding with two layers on linear on top
    def __init__(self, embedding_size):
        super(AwesomeEMBEDDER, self).__init__()
        self.embedding = nn.Embedding(len(EVAL.anime_mapping)+1, embedding_size)
        self.fc1 = nn.Linear(embedding_size, 1024)
        self.fc2 = nn.Linear(1024, embedding_size)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.embedding(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x
    
def add_positional_info(sentence):
    return [f"{word}_pos{idx}" for idx, word in enumerate(sentence)]


def get_data(EVALUATOR, set):
    if set == 'train':
        history_test_masked, history_test_heldout, test_k = EVALUATOR.get_train_set()
    elif set == 'test':
        history_test_masked, history_test_heldout, test_k = EVALUATOR.start_eval_test_set()

    # user_dict = dict()
    # for i in range(len(history_test_masked)):
    #     user_id = EVALUATOR.test_ids[i]
    #     user_dict[user_id] = {
    #         'history': dict(),
    #         'predict': dict()
    #     }
    #     for j in history_test_masked[i].nonzero()[0]:
    #         anime_id = j.item()
    #         canonical_id = EVALUATOR.canonical_id_to_anime_id[anime_id]
    #         anime_name = EVALUATOR.anime_mapping[canonical_id].name
    #         user_dict[user_id]['history'][anime_id] = [anime_name, canonical_id, history_test_masked[i][anime_id].item()]
    #     for j in history_test_heldout[i].nonzero()[0]:
    #         anime_id = j.item()
    #         canonical_id = EVALUATOR.canonical_id_to_anime_id[anime_id]
    #         anime_name = EVALUATOR.anime_mapping[canonical_id].name
    #         user_dict[user_id]['predict'][anime_id] = [anime_name, canonical_id, history_test_heldout[i][anime_id].item()]

    anime_dict = dict()
    for i in EVALUATOR.anime_mapping.keys():
        anime_dict[i] = {
            'Anime ID': i,
            'Anime Name': EVALUATOR.anime_mapping[i].name,
            # 'Genre': [genre.name for genre in EVALUATOR.anime_mapping[i].genres],
            # 'Type': EVALUATOR.anime_mapping[i].type.name,
            # 'Episodes': EVALUATOR.anime_mapping[i].episodes,
            # 'Rating': EVALUATOR.anime_mapping[i].rating,
        }

    user_history = []
    user_predict = []
    for i in range(len(history_test_masked)):
        user_history.append(history_test_masked[i].nonzero()[0])
    for i in range(len(history_test_heldout)):
        user_predict.append(history_test_heldout[i].nonzero()[0])

    return user_history, user_predict, test_k, anime_dict

def get_word2vec(user_dict, anime_dict):
    user_history_sentences = []
    user_predict_sentences = []
    anime_sentences = []
    for user_key, user_value in user_dict.items():
        history = user_value['history']
        history_sentence = ' '.join([f'{k}: {v}' for k, v in history.items()])
        history_sentence = history_sentence.split()
        history_sentence = add_positional_info(history_sentence)
        user_history_sentences.append(history_sentence)
        predict = user_value['predict']
        predict_sentence = ' '.join([f'{k}: {v}' for k, v in predict.items()])
        predict_sentence = predict_sentence.split()
        predict_sentence = add_positional_info(predict_sentence)
        user_predict_sentences.append(predict_sentence)



    for anime_key, anime_value in anime_dict.items():
        anime_sentence = ' '.join([f'{k}: {v}' for k, v in anime_value.items()])
        anime_sentence = anime_sentence.split()
        anime_sentence = add_positional_info(anime_sentence)
        anime_sentences.append(anime_sentence)

    all_sentences = user_history_sentences + user_predict_sentences + anime_sentences

    model = Word2Vec(all_sentences, vector_size=args.embedding_size, window=1, min_count=1, sg=0)
    # model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
    # model = api_gen.load('word2vec-google-news-300')

    def embed_sentence(sentence):
        valid_words = [word for word in sentence if word in model.wv.key_to_index]
        if not valid_words:
            return np.zeros(model.vector_size) 
        return np.mean(model.wv[valid_words], axis=0)

    # def embed_sentence(sentence):
    #     valid_words = [word for word in sentence if word in model.key_to_index] 
    #     if not valid_words:
    #         return np.zeros(model.vector_size) 
    #     return np.mean(model[valid_words], axis=0)

    user_history_embed = torch.tensor([embed_sentence(sentence) for sentence in user_history_sentences], dtype=torch.float32).cuda()
    user_predict_embed = torch.tensor([embed_sentence(sentence) for sentence in user_predict_sentences], dtype=torch.float32).cuda()
    anime_embed = torch.tensor([embed_sentence(sentence) for sentence in anime_sentences], dtype=torch.float32).cuda()

    # normalize all embeddings to mean 0 and std 1 per vector
    user_history_embed = (user_history_embed - user_history_embed.mean(dim=0)) / user_history_embed.std(dim=0)
    user_predict_embed = (user_predict_embed - user_predict_embed.mean(dim=0)) / user_predict_embed.std(dim=0)
    anime_embed = (anime_embed - anime_embed.mean(dim=0)) / anime_embed.std(dim=0)

    return user_history_embed, user_predict_embed, anime_embed

def get_embeddings():
    model = AutoModel.from_pretrained(args.embedder_name).cuda()
    tokenizer = AutoTokenizer.from_pretrained(args.embedder_name)

    user_history_sentences = []
    user_predict_sentences = []
    anime_sentences = []
    for user_key, user_value in user_dict.items():
        history = user_value['history']
        history_sentence = ' '.join([f'Anime ID {k}: Rating {v}' for k, v in history.items()])
        user_history_sentences.append(history_sentence)
        predict = user_value['predict']
        predict_sentence = ' '.join([f'Anime ID {k}: Rating {v}' for k, v in predict.items()])
        user_predict_sentences.append(predict_sentence)

    for anime_key, anime_value in anime_dict.items():
        anime_sentence = ' '.join([f'{k}: {v}' for k, v in anime_value.items()])
        anime_sentences.append(anime_sentence)

    #loop through batches of args.embed_batch_size on user_history_sentences, user_predict_sentences, anime_sentences
    user_history_embed = []
    user_predict_embed = []
    anime_embed = []
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(user_history_sentences), args.embed_batch_size)):
            user_history_batch = user_history_sentences[i:i+args.embed_batch_size]
            batch_dict = tokenizer(user_history_batch, max_length=512, padding=True, truncation=True, return_tensors='pt')
            batch_dict = {k: v.to('cuda') for k, v in batch_dict.items()}
            outputs = model(**batch_dict)
            embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            user_history_embed.append(embeddings)

    user_history_embed = torch.cat(user_history_embed, dim=0).detach().cpu()
    user_history_embed = F.normalize(user_history_embed, p=2, dim=1)

    with torch.no_grad():
        for i in tqdm(range(0, len(user_predict_sentences), args.embed_batch_size)):
            user_predict_batch = user_predict_sentences[i:i+args.embed_batch_size]
            batch_dict = tokenizer(user_predict_batch, max_length=512, padding=True, truncation=True, return_tensors='pt')
            batch_dict = {k: v.to('cuda') for k, v in batch_dict.items()}
            outputs = model(**batch_dict)
            embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            user_predict_embed.append(embeddings)

    user_predict_embed = torch.cat(user_predict_embed, dim=0).detach().cpu()
    user_predict_embed = F.normalize(user_predict_embed, p=2, dim=1)

    with torch.no_grad():
        for i in tqdm(range(0, len(anime_sentences), args.embed_batch_size)):
            anime_batch = anime_sentences[i:i+args.embed_batch_size]
            batch_dict = tokenizer(anime_batch, max_length=512, padding=True, truncation=True, return_tensors='pt')
            batch_dict = {k: v.to('cuda') for k, v in batch_dict.items()}
            outputs = model(**batch_dict)
            embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            anime_embed.append(embeddings)

    anime_embed = torch.cat(anime_embed, dim=0).detach().cpu()
    anime_embed = F.normalize(anime_embed, p=2, dim=1)

    return user_history_embed, user_predict_embed, anime_embed

def get_sentences():
    user_history_sentences = []
    user_predict_sentences = []
    anime_sentences = []
    for user_key, user_value in user_dict.items():
        history = user_value['history']
        history_sentence = ' '.join([f'{k} | {v[0]} ID {v[1]} Rating {v[2]}, ' for k, v in history.items()])
        user_history_sentences.append(history_sentence)
        predict = user_value['predict']
        predict_sentence = ' '.join([f'{k} | {v[0]} ID {v[1]} Rating {v[2]}, ' for k, v in predict.items()])
        user_predict_sentences.append(predict_sentence)

    for anime_key, anime_value in anime_dict.items():
        anime_sentence = ' '.join([f'{k}: {v}' for k, v in anime_value.items()])
        anime_sentences.append(anime_sentence)

    return user_history_sentences, user_predict_sentences, anime_sentences

def train_embedder(retriever_user, retriever_anime, user_history_embed, anime_embed, user_dict, anime_dict):   
    optimizer = torch.optim.Adam(list(retriever_user.parameters()), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    retriever_user.train()
    retriever_anime.train()
    loss_list = deque(maxlen=100)
    all_anime_ids = set(range(len(anime_embed)))

    for epoch in tqdm(range(args.epochs)):
        user_keys = list(user_dict.keys())
        for i, user_key in enumerate(user_keys):
            user_embed = user_history_embed[i].reshape(1, -1)
            # negative_animes = [anime_embed[anime_id] for anime_id in user_dict[user_key]['history'].keys()]
            history_anime_ids = set(user_dict[user_key]['history'].keys())
            predict_anime_ids = set(user_dict[user_key]['predict'].keys())
            available_anime_ids = list(all_anime_ids - history_anime_ids - predict_anime_ids)
            random_negative_anime_ids = random.sample(available_anime_ids, args.negative_num)
            negative_animes = [anime_embed[anime_id] for anime_id in random_negative_anime_ids]
            negative_animes = torch.stack(negative_animes)
            positive_animes = [anime_embed[anime_id] for anime_id in user_dict[user_key]['predict'].keys()]
            ###get one random positive anime
            # positive_animes = random.sample(positive_animes, 1)
            positive_animes = torch.stack(positive_animes)
            positive_ratings = [pairs[1][2] for pairs in user_dict[user_key]['predict'].items()]
            positive_ratings = torch.tensor(positive_ratings).cuda()
            # positive_ratings = torch.where(positive_ratings < 5, positive_ratings, torch.zeros_like(positive_ratings))
            # positive_ratings = torch.exp(positive_ratings)
            negative_ratings = [anime_dict[EVAL.canonical_id_to_anime_id[anime_id]]['Rating'] for anime_id in random_negative_anime_ids]
            # negative_ratings = [pairs[1] for pairs in user_dict[user_key]['history'].items()]
            negative_ratings = torch.tensor(negative_ratings).cuda()
            negative_ratings_reciprocal = 1 / negative_ratings
            # negative_ratings_reciprocal = torch.where(negative_ratings_reciprocal > 0.15, negative_ratings_reciprocal, torch.zeros_like(negative_ratings))
            # negative_ratings_reciprocal = torch.exp(negative_ratings_reciprocal)

            user_input = user_embed.cuda()
            anime_input = torch.cat([negative_animes, positive_animes]).cuda()
            user_output = retriever_user(user_input)
            anime_output = retriever_user(anime_input)
            user_output = user_output.reshape(1, -1)
            negative_output = anime_output[:len(negative_animes)]
            positive_output = anime_output[len(negative_animes):]

            negative_cos_sim = torch.cosine_similarity(user_output, negative_output, dim=1)
            positive_cos_sim = torch.cosine_similarity(user_output, positive_output, dim=1)

            negative_cos_sim_exp = torch.exp(negative_cos_sim)
            positive_cos_sim_exp = torch.exp(positive_cos_sim)
            
            # pos_term = (positive_cos_sim_exp * positive_ratings).sum()
            # neg_term = (negative_cos_sim_exp * negative_ratings_reciprocal).sum()

            pos_term = positive_cos_sim_exp.sum()
            neg_term = negative_cos_sim_exp.sum()

            loss = -torch.log(pos_term / (pos_term + neg_term)) / args.grad_accum
            # loss = -torch.log(positive_cos_sim_exp.sum())
            # logits = torch.cat([positive_cos_sim, negative_cos_sim], dim=0).float().reshape(1, -1)
            # labels = torch.zeros(len(logits), dtype=torch.long).cuda() 
            # labels[0] = 1
            # loss = criterion(logits, labels) / args.grad_accum
            loss_list.append(loss.item())
            loss.backward()
            if i % args.grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad()

            if i % 100 == 0:
                print(f'Epoch: {epoch}, User: {i}, Avg Loss: {np.mean(loss_list)}', flush=True)
 
        if i % args.grad_accum != 0:
            optimizer.step()
            optimizer.zero_grad()

    return retriever_user, retriever_anime

def train_model(user_dict, user_history_sentences, user_predict_sentences, anime_sentences):
    model = AutoModel.from_pretrained(args.embedder_name)
    tokenizer = AutoTokenizer.from_pretrained(args.embedder_name)

    lora_config = LoraConfig(
        r=4,  # Rank of the low-rank matrices
        lora_alpha=32,  # Scaling factor
        lora_dropout=0.1,  # Dropout rate
        target_modules=["query", "value"]  # Target modules to apply LoRA
    )

    model = get_peft_model(model, lora_config)
    model = model.to(torch.bfloat16).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_list = deque(maxlen=100)
    model.train()
    all_anime_ids = set(range(len(anime_sentences)))
    for epoch in range(args.epochs):
        user_keys = list(user_dict.keys())
        for i, user_key in enumerate(user_keys):
            user_sentence = user_history_sentences[i]
            # negative_animes = [anime_sentences[anime_id] for anime_id in user_dict[user_key]['history'].keys()]
            
            history_anime_ids = set(user_dict[user_key]['history'].keys())
            predict_anime_ids = set(user_dict[user_key]['predict'].keys())
            available_anime_ids = list(all_anime_ids - history_anime_ids - predict_anime_ids)
            random_negative_anime_ids = random.sample(available_anime_ids, 10)
            negative_animes = [anime_sentences[anime_id] for anime_id in random_negative_anime_ids]
            positive_animes = [anime_sentences[anime_id] for anime_id in user_dict[user_key]['predict'].keys()]
            ### randomly select positive
            # positive_animes = random.sample(positive_animes, 1)
            input_batch = [user_sentence] + negative_animes + positive_animes
            batch_dict = tokenizer(input_batch, max_length=512, padding=True, truncation=True, return_tensors='pt')
            # batch_dict = {k: v.to('cuda') for k, v in batch_dict.items()}
            with torch.no_grad():
                first_key = list(batch_dict.keys())[0]
                batch_dict[first_key] = batch_dict[first_key].to('cuda').detach()

            # Move the rest of the entries to GPU with gradient tracking
            for k, v in batch_dict.items():
                if k != first_key:
                    batch_dict[k] = v.to('cuda')
            outputs = model(**batch_dict)
            embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            
            user_output = embeddings[0].reshape(1, -1)
            negative_output = embeddings[1:1+len(negative_animes)]
            positive_output = embeddings[1+len(negative_animes):]

            negative_cos_sim = torch.cosine_similarity(user_output, negative_output, dim=1)
            positive_cos_sim = torch.cosine_similarity(user_output, positive_output, dim=1)

            negative_cos_sim_exp = torch.exp(negative_cos_sim)
            positive_cos_sim_exp = torch.exp(positive_cos_sim)

            pos_term = positive_cos_sim_exp.sum()
            neg_term = negative_cos_sim_exp.sum()

            loss = -torch.log(pos_term / ( + neg_term)) / args.grad_accum
            loss_list.append(loss.item())
            loss.backward()
            if i % args.grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad()

            if i % 5 == 0:
                print(f'Epoch: {epoch}, User: {i}, Avg Loss: {np.mean(loss_list)}', flush=True)
 
        if i % args.grad_accum != 0:
            optimizer.step()
            optimizer.zero_grad()

    return model, tokenizer

def train_awesome_embedder(fresh_embedder, anime_name_embeddings, user_history, user_predict):
    fresh_embedder.train()
    text_embedder.train()
    optimizer = torch.optim.Adam(fresh_embedder.parameters(), lr=args.learning_rate)
    loss_list = deque(maxlen=100)
    all_anime_ids = set(range(len(EVAL.anime_mapping)))

    for epoch in range(args.epochs):
        for batch_idx, (user_history_batch, user_predict_batch) in enumerate(zip(batch_iterable(user_history, args.embed_batch_size), batch_iterable(user_predict, args.embed_batch_size))):
            query_input_batch = []
            positive_animes_batch = []
            negative_animes_batch = []
            # query_name_batch = []
            positive_name_batch = []
            negative_name_batch = []
            for i, (query_list, predict_list) in enumerate(zip(user_history_batch, user_predict_batch)):
                query_input = torch.tensor(query_list).cuda()
                positive_animes = torch.tensor(predict_list).cuda()
                negative_animes = all_anime_ids - set(query_list) - set(predict_list)
                negative_animes = random.sample(list(negative_animes), args.negative_num)
                negative_animes = torch.tensor(negative_animes).cuda()
                query_input_batch.append(query_input)
                positive_animes_batch.append(positive_animes)
                negative_animes_batch.append(negative_animes)

                # name_list = []
                # for anime_id in query_input:
                #     name_list.append(anime_dict[EVAL.canonical_id_to_anime_id[anime_id.item()]]['Anime Name'])
                # query_name_batch.append(name_list)

                name_list = []
                for anime_id in positive_animes:
                    name_list.append(anime_name_embeddings[EVAL.canonical_id_to_anime_id[anime_id.item()]])
                positive_name_batch.append(name_list)

                name_list = []
                for anime_id in negative_animes:
                    name_list.append(anime_name_embeddings[EVAL.canonical_id_to_anime_id[anime_id.item()]])
                negative_name_batch.append(name_list)
            positive_name_batch_tensor = torch.stack([torch.stack(name_list) for name_list in positive_name_batch]).squeeze()
            negative_name_batch_tensor = torch.stack([torch.stack(name_list) for name_list in negative_name_batch]).squeeze()

            ### pad query_input_batch to longest and then stack
            max_len = max([len(query) for query in query_input_batch])
            padding_value = len(EVAL.anime_mapping)
            for i, query in enumerate(query_input_batch):
                query_input_batch[i] = F.pad(query, (0, max_len - len(query)), value=padding_value)

            query_input_batch = torch.stack(query_input_batch)
            positive_animes_batch = torch.stack(positive_animes_batch)
            negative_animes_batch = torch.stack(negative_animes_batch)
            query_output_batch = fresh_embedder(query_input_batch)
            positive_output_batch = fresh_embedder(positive_animes_batch)
            negative_output_batch = fresh_embedder(negative_animes_batch)
            mask = (query_input_batch != padding_value).float()
            mask = mask.unsqueeze(-1)
            masked_sum = (query_output_batch * mask).sum(dim=1)
            valid_counts = mask.sum(dim=1).squeeze()
            query_output_batch_mean = masked_sum / valid_counts.reshape(-1, 1)

            query_output_batch_mean = query_output_batch_mean.reshape(query_output_batch.shape[0], -1)

            query_output_batch = query_output_batch.mean(dim=1).reshape(query_output_batch.shape[0], -1)
            query_output_batch = query_output_batch.reshape(-1, 1, args.embedding_size)

            if args.anime_title:
                positive_output_batch = torch.concat([positive_output_batch, positive_name_batch_tensor], dim=-1)
                negative_output_batch = torch.concat([negative_output_batch, negative_name_batch_tensor], dim=-1)

                ##repeat the last dimension of query_output_batch to match the last dimension of positive_output_batch and negative_output_batch
                query_output_batch =  query_output_batch.repeat(1, 1, 7)

            negative_cos_sim = torch.cosine_similarity(query_output_batch, negative_output_batch, dim=2) / args.temperature
            positive_cos_sim = torch.cosine_similarity(query_output_batch, positive_output_batch, dim=2) / args.temperature

            negative_cos_sim_exp = torch.exp(negative_cos_sim)
            positive_cos_sim_exp = torch.exp(positive_cos_sim)

            pos_term = positive_cos_sim_exp.sum(dim=1)
            neg_term = negative_cos_sim_exp.sum(dim=1)

            loss = -torch.log(pos_term / (pos_term + neg_term)).mean()
            loss_list.append(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch_idx % 10 == 0:
                print(f'Epoch: {epoch}, User Batch: {batch_idx}, Avg Loss: {np.mean(loss_list)}', flush=True)

    return fresh_embedder

            

            
            

def evaluate_embedder(retriever_user, retriever_anime, user_history_embed, anime_embed, user_dict, test_k):
    retriever_user.eval()
    retriever_anime.eval()
    anime_embed_output = torch.zeros(anime_embed.shape).cuda()
    user_history_embed_output = torch.zeros(user_history_embed.shape).cuda()
    for i in range(anime_embed.shape[0]):
        anime_embed_output[i] = retriever_user(anime_embed[i].reshape(1, -1).cuda())
    for i in range(user_history_embed.shape[0]):
        user_history_embed_output[i] = retriever_user(user_history_embed[i].reshape(1, -1).cuda())

    anime_embed_output = anime_embed_output.detach().cpu().numpy()
    user_history_embed_output = user_history_embed_output.detach().cpu().numpy()
    user_history_norm = user_history_embed_output / np.linalg.norm(user_history_embed_output, axis=1, keepdims=True)
    anime_embed_norm = anime_embed_output / np.linalg.norm(anime_embed_output, axis=1, keepdims=True)
    scores = np.dot(user_history_norm, anime_embed_norm.T)

    for i, user_key in enumerate(user_dict.keys()):
        for anime_id in user_dict[user_key]['history'].keys():
            scores[i][anime_id] = -np.inf

    top_k = np.argsort(-scores, axis=1)[:, :test_k]

    return top_k

def evaluate_embedder_big(retriever, tokenizer, user_history_sentences, anime_sentences, user_dict, test_k):
    retriever.eval()
    user_embeddings = []
    anime_embeddings = []
    with torch.no_grad():
        for i in range(0, len(user_history_sentences), args.embed_batch_size):
            user_history_batch = user_history_sentences[i:i+args.embed_batch_size]
            batch_dict = tokenizer(user_history_batch, max_length=512, padding=True, truncation=True, return_tensors='pt')
            batch_dict = {k: v.to('cuda') for k, v in batch_dict.items()}
            outputs = retriever(**batch_dict)
            embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            user_embeddings.append(embeddings)

    user_embeddings = torch.cat(user_embeddings, dim=0).detach().cpu()

    with torch.no_grad():
        for i in range(0, len(anime_sentences), args.embed_batch_size):
            anime_batch = anime_sentences[i:i+args.embed_batch_size]
            batch_dict = tokenizer(anime_batch, max_length=512, padding=True, truncation=True, return_tensors='pt')
            batch_dict = {k: v.to('cuda') for k, v in batch_dict.items()}
            outputs = retriever(**batch_dict)
            embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            anime_embeddings.append(embeddings)

    anime_embeddings = torch.cat(anime_embeddings, dim=0).detach().cpu()

    user_history_norm = user_embeddings / torch.norm(user_embeddings, dim=1, keepdim=True)
    anime_embed_norm = anime_embeddings / torch.norm(anime_embeddings, dim=1, keepdim=True)
    scores = torch.matmul(user_history_norm, anime_embed_norm.T)

    for i, user_key in enumerate(user_dict.keys()):
        for anime_id in user_dict[user_key]['history'].keys():
            scores[i][anime_id] = -np.inf

    scores = scores.to(torch.float32).numpy()
    top_k = np.argsort(-scores, axis=1)[:, :test_k]

    return top_k

def evaluate_awesome_embedder(fresh_embedder, anime_name_embeddings, user_history, test_k):
    fresh_embedder.eval()
    user_embeddings = torch.zeros((len(user_history), args.embedding_size)).cuda()
    anime_embeddings = torch.zeros((len(EVAL.anime_mapping), args.embedding_size)).cuda()
    all_anime_ids = set(range(len(EVAL.anime_mapping)))
    for i, user_input in enumerate(user_history):
        user_input = torch.tensor(user_input).cuda()
        user_output = fresh_embedder(user_input)
        user_output = user_output.mean(dim=0)
        user_embeddings[i] = user_output

    anime_name_embeddings_list = []
    with torch.no_grad():
        for anime_id in all_anime_ids:
            name_embedding = anime_name_embeddings[EVAL.canonical_id_to_anime_id[anime_id]]
            anime_name_embeddings_list.append(name_embedding)
            anime_input = torch.tensor(anime_id).cuda()
            anime_output = fresh_embedder(anime_input)
            anime_embeddings[anime_id] = anime_output

    anime_name_embeddings = torch.stack(anime_name_embeddings_list).cpu().numpy().squeeze()


    user_embeddings = user_embeddings.detach().cpu().numpy()
    anime_embeddings = anime_embeddings.detach().cpu().numpy()
    if args.anime_title:
        anime_embeddings = np.concatenate([anime_embeddings, anime_name_embeddings], axis=-1)
        user_embeddings =  np.tile(user_embeddings, (1, 7))

    user_embeddings_norm = user_embeddings / np.linalg.norm(user_embeddings, axis=1, keepdims=True)
    anime_embeddings_norm = anime_embeddings / np.linalg.norm(anime_embeddings, axis=1, keepdims=True)
    scores = np.dot(user_embeddings_norm, anime_embeddings_norm.T)

    for i in range(len(user_history)):
        for anime_id in user_history[i]:
            scores[i][anime_id] = -np.inf

    top_k = np.argsort(-scores, axis=1)[:, :test_k]

    return top_k

def get_text_embeddings(text_embedder, text_tokenizer, anime_dict):
    text_embedder.eval()
    anime_name_embeddings = dict()
    with torch.no_grad():
        for anime_key, anime_value in tqdm(anime_dict.items()):
            anime_name = anime_value['Anime Name']
            anime_name = text_tokenizer(anime_name, max_length=16, padding=True, truncation=True, return_tensors='pt')
            anime_name = {k: v.to('cuda') for k, v in anime_name.items()}
            outputs = text_embedder(**anime_name)
            embeddings = average_pool(outputs.last_hidden_state, anime_name['attention_mask'])
            anime_name_embeddings[anime_key] = embeddings

    return anime_name_embeddings

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-cache_dir", "--cache_dir", type=str, default="/data/user_data/taeyoun3/huggingface/hub", help="Path to the cache directory")
    parser.add_argument("--data_path", type=str, default='../data/copperunion')
    parser.add_argument("--embedder_name", type=str, default="thenlper/gte-base")
    parser.add_argument("--embed_batch_size", type=int, default=128)
    parser.add_argument("--embedding_size", type=int, default=128)   
    parser.add_argument("--learning_rate", type=float, default=3e-3)
    parser.add_argument("--epochs", type=int, default=1) #25
    parser.add_argument("--negative_num", type=int, default=100)
    parser.add_argument("--grad_accum", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=1)
    parser.add_argument("--anime_title", type=int, default=1)
    args = parser.parse_args()
    print(args)

    EVAL = evaluator.Evaluator(path=args.data_path, normalize_unrated=False)
    # user_dict, anime_dict, test_k = get_data(EVAL)
    user_history, user_predict, test_k, anime_dict = get_data(EVAL, 'train')
    fresh_embedder = nn.Embedding(len(EVAL.anime_mapping)+1, args.embedding_size).cuda()
    text_embedder = AutoModel.from_pretrained(args.embedder_name, cache_dir=args.cache_dir).cuda()
    text_tokenizer = AutoTokenizer.from_pretrained(args.embedder_name)

    anime_name_embeddings = get_text_embeddings(text_embedder, text_tokenizer, anime_dict)

    # fresh_embedder = AwesomeEMBEDDER(args.embedding_size).cuda()
    fresh_embedder = train_awesome_embedder(fresh_embedder, anime_name_embeddings, user_history, user_predict)


    # user_history_embed, user_predict_embed, anime_embed = get_word2vec(user_dict, anime_dict)
    # user_history_embed, user_predict_embed, anime_embed = get_embeddings()
    # retriever_user = Retriever(args.embedding_size).cuda()
    # retriever_anime = Retriever(args.embedding_size).cuda()
    # # retriever = TransformerDecoderNetwork(args.embedding_size).cuda()
    # retriever_user, retriever_anime = train_embedder(retriever_user, retriever_anime, user_history_embed, anime_embed, user_dict, anime_dict)
    # user_history_sentences, user_predict_sentences, anime_sentences = get_sentences()
    # retriever, tokenizer = train_model(user_dict, user_history_sentences, user_predict_sentences, anime_sentences)
    user_history, user_predict, test_k, _ = get_data(EVAL, 'test')
    top_k = evaluate_awesome_embedder(fresh_embedder, anime_name_embeddings, user_history, test_k)
    # top_k = evaluate_embedder(retriever_user, retriever_anime, user_history_embed, anime_embed, user_dict, test_k)
    # top_k = evaluate_embedder_big(retriever, tokenizer, user_history_sentences, anime_sentences, user_dict, test_k)
    score = EVAL.end_eval_test_set(top_k)

    print(f'Test Score: {score}')
