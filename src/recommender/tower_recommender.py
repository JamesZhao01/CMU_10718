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
    collate_item_feature,
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
        self.load_from = load_from
        self.save_to = save_to
        self.use_histories = use_histories
        self.device = device

        self.model = TowerModel(
            n_users=n_users,
            n_anime=n_anime,
            embedding_dimension=embedding_dimension,
            user_embedder=user_embedder,
            item_embedder=item_embedder,
            temperature=temperature,
        ).to(self.device)
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
        if self.load_from:
            print(f"Skipping training because model is loaded...")
        losses = []
        epoch_losses = []
        display_loss = 0
        for epoch in range(self.epochs):
            print(f"Epoch {epoch+1} / {self.epochs}")
            batch_iterator = tqdm.tqdm(
                enumerate(self.dataloader),
                total=(len(self.dataloader.dataset) + self.batch_size - 1)
                // self.batch_size,
            )
            epoch_loss, n = 0, 0
            for batch_idx, (user, positive, negative) in batch_iterator:
                batch_size = len(user)
                user = [u.to(self.device) for u in user]
                positive = [p.to(self.device) for p in positive]
                negative = [n.to(self.device) for n in negative]

                self.optimizer.zero_grad()
                self.model.train()
                loss = self.model.train_step(user, positive, negative)
                loss.backward()
                self.optimizer.step()
                loss_item = loss.item()
                display_loss = 0.8 * display_loss + 0.2 * loss_item
                batch_iterator.set_description(f"Loss: {display_loss:.4f}")
                losses.append(loss_item)
                epoch_loss = (epoch_loss * n + loss_item) / (n + batch_size)
                n += batch_size
            epoch_losses.append(epoch_loss)
            print(f"[{epoch}] Loss: {epoch_loss}")
        if self.save_to:
            print(f"Saving model to {self.save_to}")
            os.makedirs(os.path.dirname(self.save_to), exist_ok=True)
            torch.save(self.model.state_dict(), self.save_to)
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

        def batchify(data, batch_size):
            n = len(data)
            for i in range(0, n, batch_size):
                yield data[i : min(i + batch_size, n)]

        self.model.eval()
        with torch.no_grad():
            users, histories = features
            users, histories = torch.from_numpy(users), torch.from_numpy(histories)
            user_features = [
                featurize_singular_user(user, history, self.user_embedder)
                for user, history in zip(users, histories)
            ]
            user_embeddings = []
            for batch in batchify(user_features, self.batch_size):
                collated_user_features = collated_user_features(
                    batch, self.user_embedder
                )
                collated_user_features = (
                    feature.to(self.device) for feature in collated_user_features
                )
                batch_user_embedding = self.model.embed_user(users)
                user_embeddings.append(batch_user_embedding)
            user_embeddings = torch.vstack(user_embeddings).cpu()

            anime_embeddings = []
            for batch in self.anime_dataloader:
                collated_anime_features = collate_item_feature(
                    batch, self.item_embedder
                )
                collated_anime_features = (
                    feature.to(self.device) for feature in collated_anime_features
                )
                batch_anime_embedding = self.model.embed_feats(batch)
                anime_embeddings.append(batch_anime_embedding)
            anime_embeddings = torch.vstack(anime_embeddings).cpu()

        user_interaction_mask = np.zeros((len(users), self.n_anime)).int()
        for i, (user, history) in enumerate(zip(users, histories)):
            user_interaction_mask[i, history] = 1
        user_embeddings = user_embeddings[:, :, None]
        anime_embeddings = anime_embeddings.T[None, :, :]
        scores = torch.cosine_similarity(user_embeddings, anime_embeddings, dim=1)
        scores[user_interaction_mask] = -torch.inf
        shows = torch.argsort(scores, dim=1, descending=True)
        recommended = shows[:, :k].numpy()

        return recommended


class AnimeEnumerator(data.Dataset):
    def __init__(self, datamodule: DataModule, item_embedder: List[Tuple[str, dict]]):
        super().__init__()
        self.datamodule = datamodule
        self.item_embedder = item_embedder

    def __len__(self):
        return self.datamodule.max_anime_count

    def __getitem__(self, idx):
        anime = self.datamodule.canonical_anime_mapping[idx]

        return [
            featurize_item_or_items(anime, embedder) for embedder in self.item_embedder
        ]


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

        user_feature = [
            featurize_singular_user(user, history, embedder)
            for embedder in self.user_embedder
        ]
        pos_feature = [
            featurize_item_or_items(positives, embedder)
            for embedder in self.item_embedder
        ]
        neg_feature = [
            featurize_item_or_items(negatives, embedder)
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
        super().__init__()
        self.n_users = n_users
        self.n_anime = n_anime
        self.embedding_dimension = embedding_dimension
        self.temperature = temperature

        self.user_embedders = torch.nn.ModuleList(
            [
                build_user_embedder(embedder_type, embedder_metadata, self.n_users)
                for embedder_type, embedder_metadata in user_embedder
            ]
        )
        self.item_embedders = torch.nn.ModuleList(
            [
                build_item_embedder(embedder_type, embedder_metadata, self.n_anime)
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

    def train_step(
        self,
        users: Tuple[torch.Tensor],
        pos_features: Tuple[torch.Tensor],
        neg_features: Tuple[torch.Tensor],
    ):
        user_embedding = self.embed_user(users)
        pos_feature_embedding = self.embed_feats(pos_features)
        neg_feature_embedding = self.embed_feats(neg_features)
        print(
            f"{user_embedding.shape=} / {pos_feature_embedding.shape=} / {neg_feature_embedding.shape=}"
        )
        negative_cos_sim = (
            torch.cosine_similarity(user_embedding, pos_feature_embedding, dim=-1)
            / self.temperature
        )
        positive_cos_sim = (
            torch.cosine_similarity(user_embedding, neg_feature_embedding, dim=-1)
            / self.temperature
        )
        print(
            f"{user_embedding.shape=} / {pos_feature_embedding.shape=} / {neg_feature_embedding.shape=}"
        )
        negative_cos_sim_exp = torch.exp(negative_cos_sim)
        positive_cos_sim_exp = torch.exp(positive_cos_sim)

        pos_term = positive_cos_sim_exp.sum(dim=1)
        neg_term = negative_cos_sim_exp.sum(dim=1)
        print(f"{pos_term.mean().item()=} / {pos_term.mean().item()=}")

        loss = -torch.log(pos_term / (pos_term + neg_term)).mean()
        print(1 / 0)
        return loss
