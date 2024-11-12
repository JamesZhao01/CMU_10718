from abc import ABC
from enum import Enum
from typing import List, Literal, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import tqdm
import os

import functools
from recommender.generic_recommender import GenericRecommender
from utils.data.data_classes import Anime, User
from utils.data.data_module import DataModule
from recommender.tower_recommender_embedders import (
    user_pos_neg_collator,
    item_collator,
    featurize_item_or_items,
    featurize_singular_user,
    build_user_embedder,
    build_item_embedder,
)


class TowerRecommender(GenericRecommender):
    def __init__(
        self,
        datamodule: DataModule,
        n_users: int,
        n_anime: int,
        embedding_dimension: int,
        user_embedder: List[Tuple[str, dict]],
        item_embedder: List[Tuple[str, dict]],
        temperature: float = 1,
        n_pos: int = 10,
        n_neg: int = 10,
        lr=1e-3,
        epochs=10,
        batch_size=32,
        dataset_args={},
        load_from="",
        save_to="",
        use_histories=False,
        device="cuda",
        **kwargs,
    ):
        super().__init__(datamodule)

        self.n_users = n_users
        self.n_anime = n_anime
        self.embedding_dimension = embedding_dimension
        self.user_embedder = user_embedder
        self.item_embedder = item_embedder
        self.temperature = temperature
        self.n_pos = n_pos
        self.n_neg = n_neg

        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.load_from = load_from
        self.save_to = save_to
        self.use_histories = use_histories

        self.model = TowerModel(
            n_users=n_users,
            n_anime=n_anime,
            embedding_dimension=embedding_dimension,
            user_embedder=user_embedder,
            item_embedder=item_embedder,
            temperature=temperature,
        )
        if self.load_from:
            self.model.load_state_dict(torch.load(self.load_from))
        self.dataset = DatasetWrapper(
            datamodule=datamodule,
            user_embedder=user_embedder,
            item_embedder=item_embedder,
            n_pos=n_pos,
            n_neg=n_neg,
        )
        self.dataloader = data.DataLoader(
            dataset=self.dataset,
            batch_size=batch_size,
            collate_fn=functools.partial(
                user_pos_neg_collator,
                user_embedder=user_embedder,
                item_embedder=item_embedder,
            ),
            **dataset_args,
        )

        self.anime_dataset = AnimeEnumerator(datamodule, item_embedder)
        self.anime_dataloader = data.DataLoader(
            dataset=self.anime_dataset,
            batch_size=batch_size,
            collate_fn=functools.partial(item_collator, item_embedder=item_embedder),
        )
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)

    def train(self, *args, **kwargs):
        self.model.train()
        losses = []
        epoch_losses = []
        for epoch in tqdm.tqdm(range(self.epochs), desc="Epochs..."):
            epoch_loss, n = 0, 0
            for batch_idx, (user, positive, negative) in enumerate(self.dataloader):
                batch_size = len(user)
                user = user.to(self.device)
                positive = positive.to(self.device)
                negative = negative.to(self.device)

                self.optimizer.zero_grad()
                loss = self.model.train(user, positive, negative)
                loss.backward()
                self.optimizer.step()
                loss_item = loss.item()
                losses.append(loss_item)
                epoch_loss = (epoch_loss * n + loss_item) / (n + batch_size)
                n += batch_size
            epoch_losses.append(epoch_loss)
            print(f"[{epoch:>3f}] Loss: {epoch_loss}")
        if self.save_to:
            print(f"Saving model to {self.save_to}")
            os.makedirs(os.path.dirname(self.save_to), exist_ok=True)
            with open(self.save_to, "wb") as f:
                torch.save(self.model.state_dict(), f)
        return {
            "losses": losses,
            "epoch_losses": epoch_losses,
        }

    def infer(self, features, k: int) -> np.ndarray[int]:
        """Takes in a ratings float tensor of size (n_test, n_anime) and returns an int tensor of
            size (n_test, k), the canonical anime ids to rank

        Args:
            k: number of items to recommend
            ratings_features: tensor of dimension (n_test, n_anime)

        Returns:
            np.ndarray[int]: tensor of size (n_test, n_anime)
        """
        self.model.eval()
        with torch.no_grad():
            users, histories = features
            users = torch.tensor(users).to(self.device)
            user_embeddings = self.model.embed_user(users)

            # TODO James


class AnimeEnumerator(data.Dataset):
    def __init__(self, datamodule: DataModule, item_embedder: List[Tuple[str, dict]]):
        super().__init__()
        self.datamodule = datamodule
        self.item_embedder = item_embedder

    def __len__(self):
        return self.datamodule.max_anime_count

    def __getitem__(self, idx):
        anime = self.datamodule.canonical_anime_mapping[idx]
        pos_feature = [
            DatasetWrapper.articulate_animes(anime, embedder)
            for embedder in self.item_embedder
        ]
        return DatasetWrapper.articulate_animes(anime, self.item_embedder)


class DatasetWrapper(data.Dataset):
    def __init__(
        self,
        datamodule: DataModule,
        user_embedder: List[Tuple[str, dict]],
        item_embedder: List[Tuple[str, dict]],
        n_pos: int = 10,
        n_neg: int = 10,
    ):
        super().__init__()
        self.datamodule = datamodule
        self.user_embedder = user_embedder
        self.item_embedder = item_embedder
        self.n_pos = n_pos
        self.n_neg = n_neg

    def __len__(self):
        return self.datamodule.max_user_count

    def __getitem__(self, idx):
        user = self.datamodule.canonical_user_mapping[idx]
        positives, history = user.partition_preserved(self.n_pos)
        positives = [
            self.datamodule.canonical_anime_mapping[anime] for anime in positives
        ]
        history = [self.datamodule.canonical_anime_mapping[anime] for anime in history]
        negatives = user.sample_negative(self.n_neg)
        negatives = [
            self.datamodule.canonical_anime_mapping[anime] for anime in negatives
        ]

        user_feature = self.articulate_user(user, history, self.user_embedder)
        pos_feature = [
            DatasetWrapper.articulate_animes(positives, embedder)
            for embedder in self.item_embedder
        ]
        neg_feature = [
            DatasetWrapper.articulate_animes(negatives, embedder)
            for embedder in self.item_embedder
        ]
        return user_feature, pos_feature, neg_feature


class TowerModel(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_anime: int,
        embedding_dimension: int,
        user_embedder: str,
        item_embedder: List[Tuple[str, dict]],
        temperature: float = 1,
    ):
        self.n_users = n_users
        self.n_anime = n_anime
        self.embedding_dimension = embedding_dimension
        self.temperature = temperature

        self.user_embedders = torch.nn.ModuleList(
            [
                build_user_embedder(embedder_type, embedder_metadata)
                for embedder_type, embedder_metadata in user_embedder
            ]
        )
        self.item_embedders = torch.nn.ModuleList(
            [
                build_item_embedder(embedder_type, embedder_metadata)
                for embedder_type, embedder_metadata in item_embedder
            ]
        )

    def embed_user(self, feats: Tuple[torch.Tensor]):
        user_embeddings = [
            embedder(feats[i]) for i, embedder in enumerate(self.user_embedders)
        ]
        user_embedding = torch.concatenate(user_embeddings, dim=-1)  # (n, n_pos, dim)
        return user_embedding

    def embed_feats(self, feats: Tuple[torch.Tensor]):
        feature_embeddings = [
            embedder(feats[i]) for i, embedder in enumerate(self.item_embedders)
        ]
        feature_embedding = torch.concatenate(
            feature_embeddings, dim=-1
        )  # (n, n_pos, dim)
        return feature_embedding

    def forward(self, users: Tuple[torch.Tensor], feats: Tuple[torch.Tensor]):
        """
        users: Tensor: (n,)
        feats: Tuple[Tensors]
        - len(feats) = # of features
        - feats[i] = (n, n_pos + n_neg, dim)
        """
        user_embedding = None if users is None else self.embed_user(users)
        feature_embedding = None if feats is None else self.embed_feats(feats)

        return user_embedding, feature_embedding

    def train(
        self,
        users: torch.Tensor,
        pos_features: torch.Tensor,
        neg_features: torch.Tensor,
    ):
        user_embedding = self.embed_user(users)
        pos_feature_embedding = self.embed_feats(pos_features)
        neg_feature_embedding = self.embed_feats(neg_features)
        negative_cos_sim = (
            torch.cosine_similarity(user_embedding, pos_feature_embedding, dim=2)
            / self.temperature
        )
        positive_cos_sim = (
            torch.cosine_similarity(user_embedding, neg_feature_embedding, dim=2)
            / self.temperature
        )

        negative_cos_sim_exp = torch.exp(negative_cos_sim)
        positive_cos_sim_exp = torch.exp(positive_cos_sim)

        pos_term = positive_cos_sim_exp.sum(dim=1)
        neg_term = negative_cos_sim_exp.sum(dim=1)

        loss = -torch.log(pos_term / (pos_term + neg_term)).mean()
        return loss
