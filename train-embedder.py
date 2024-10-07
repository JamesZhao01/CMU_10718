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

from src.utils import evaluator

class Retriever(torch.nn.Module):
    def __init__(self, input_size):
        super(Retriever, self).__init__()
        layers = []
        layers.append(nn.Linear(input_size, 1024))
        layers.append(nn.GELU())
        for _ in range(2): 
            layers.append(nn.Linear(1024, 1024))
            layers.append(nn.GELU())
        layers.append(nn.Linear(1024, input_size))
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.model(x)
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
    
def add_positional_info(sentence):
    return [f"{word}_pos{idx}" for idx, word in enumerate(sentence)]


def get_data(EVALUATOR):
    history_test_masked, history_test_heldout, test_k = EVALUATOR.start_eval_test_set()

    user_dict = dict()
    for i in range(len(history_test_masked)):
        user_dict[i] = {
            'history': dict(),
            'predict': dict()
        }
        for j in history_test_masked[i].nonzero()[0]:
            anime_id = j.item()
            user_dict[i]['history'][anime_id] = history_test_masked[i][anime_id].item()
        for j in history_test_heldout[i].nonzero()[0]:
            anime_id = j.item()
            user_dict[i]['predict'][anime_id] = history_test_heldout[i][anime_id].item()

    anime_dict = dict()
    for i in EVALUATOR.anime_mapping.keys():
        anime_dict[i] = {
            'Anime ID': i,
            'Anime Name': EVALUATOR.anime_mapping[i].name,
            'Genre': [genre.name for genre in EVALUATOR.anime_mapping[i].genres],
            'Type': EVALUATOR.anime_mapping[i].type.name,
            'Episodes': EVALUATOR.anime_mapping[i].episodes,
            'Rating': EVALUATOR.anime_mapping[i].rating,
        }
    
    return user_dict, anime_dict, test_k

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
    # import pdb; pdb.set_trace() 
    all_sentences = user_history_sentences + user_predict_sentences + anime_sentences

    model = Word2Vec(all_sentences, vector_size=args.embedding_size, window=3, min_count=1, sg=0)
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

def train_embedder(retriever, user_history_embed, anime_embed, user_dict):   
    optimizer = torch.optim.Adam(retriever.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    retriever.train()
    loss_list = deque(maxlen=100)
    for epoch in range(args.epochs):
        user_keys = list(user_dict.keys())
        random.shuffle(user_keys)
        for i, user_key in enumerate(user_keys):
            user_embed = user_history_embed[i].reshape(1, -1)
            negative_animes = [anime_embed[anime_id] for anime_id in user_dict[user_key]['history'].keys()]
            negative_animes = torch.stack(negative_animes)
            positive_animes = [anime_embed[anime_id] for anime_id in user_dict[user_key]['predict'].keys()]
            ###get one random positive anime
            positive_animes = random.sample(positive_animes, 1)
            positive_animes = torch.stack(positive_animes).reshape(1, -1)

            input_embed = torch.cat([user_embed, negative_animes, positive_animes])
            output_embed = retriever(input_embed)
            user_output = output_embed[0].reshape(1, -1)
            negative_output = output_embed[1:1+len(negative_animes)]
            positive_output = output_embed[1+len(negative_animes):]

            negative_cos_sim = torch.cosine_similarity(user_output, negative_output, dim=1)
            positive_cos_sim = torch.cosine_similarity(user_output, positive_output, dim=1)

            negative_cos_sim_exp = torch.exp(negative_cos_sim)
            positive_cos_sim_exp = torch.exp(positive_cos_sim)

            loss = -torch.log(positive_cos_sim_exp.sum() / (positive_cos_sim_exp.sum() + negative_cos_sim_exp.sum())) / args.grad_accum
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
                print(f'Epoch: {epoch}, User: {i}, Avg Loss: {np.mean(loss_list)}')
 
        if i % args.grad_accum != 0:
            optimizer.step()
            optimizer.zero_grad()

    return retriever

def evaluate_embedder(retriever, user_history_embed, anime_embed, user_dict, test_k):
    retriever.eval()
    anime_embed_output = torch.zeros(anime_embed.shape).cuda()
    user_history_embed_output = torch.zeros(user_history_embed.shape).cuda()
    for i in range(anime_embed.shape[0]):
        anime_embed_output[i] = retriever(anime_embed[i].reshape(1, -1))
    for i in range(user_history_embed.shape[0]):
        user_history_embed_output[i] = retriever(user_history_embed[i].reshape(1, -1))

    anime_embed_output = anime_embed_output.detach().cpu().numpy()
    user_history_embed_output = user_history_embed_output.detach().cpu().numpy()
    user_history_norm = user_history_embed_output / np.linalg.norm(user_history_embed_output, axis=1, keepdims=True)
    anime_embed_norm = anime_embed_output / np.linalg.norm(anime_embed_output, axis=1, keepdims=True)
    scores = np.dot(user_history_norm, anime_embed_norm.T)
    top_k = np.argsort(-scores, axis=1)[:, :test_k]

    return top_k

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default='/data/taeyoun_kim/jailbreak_work/attack_mlip/big_attack/data/copperunion')
    parser.add_argument("--embedding_size", type=int, default=300)   
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--grad_accum", type=int, default=64)
    args = parser.parse_args()
    print(args)

    EVAL = evaluator.Evaluator(path=args.data_path)
    user_dict, anime_dict, test_k = get_data(EVAL)
    user_history_embed, user_predict_embed, anime_embed = get_word2vec(user_dict, anime_dict)
    retriever = Retriever(args.embedding_size).cuda()
    # retriever = TransformerDecoderNetwork(args.embedding_size).cuda()
    retriever = train_embedder(retriever, user_history_embed, anime_embed, user_dict)
    top_k = evaluate_embedder(retriever, user_history_embed, anime_embed, user_dict, test_k)
    score = EVAL.end_eval_test_set(top_k)

    print(f'Test Score: {score}')
