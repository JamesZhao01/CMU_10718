from abc import ABC
from collections import defaultdict
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
    collate_user_feature,
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
            self.model.load_state_dict(torch.load(self.load_from, weights_only=True))
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
            return
        losses = []
        epoch_losses = []
        display = defaultdict(int)
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
                outputs = self.model.train_step(user, positive, negative)
                loss = outputs["loss"]
                loss.backward()
                self.optimizer.step()
                loss_item = loss.item()
                pos_term_item = outputs["positive_term"]
                neg_term_item = outputs["negative_term"]

                display["loss"] = display["loss"] * 0.8 + loss_item * 0.2
                display["pos_term"] = display["pos_term"] * 0.8 + pos_term_item * 0.2
                display["neg_term"] = display["neg_term"] * 0.8 + neg_term_item * 0.2

                batch_iterator.set_description(
                    f"L:{display['loss']:.3f}, +:{display['pos_term']:.3f}, -:{display['neg_term']:.3f}"
                )
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
            users: List[User]
            histories: List[List[int]]
            user_features = [
                [
                    featurize_singular_user(user, history, embedder)
                    for embedder in self.user_embedder
                ]
                for user, history in zip(users, histories)
            ]

            user_embeddings = []
            for batch in tqdm.tqdm(
                batchify(user_features, self.batch_size), desc="User Embeddings..."
            ):
                collated_user_features = collate_user_feature(batch, self.user_embedder)
                collated_user_features = [
                    feature.to(self.device) for feature in collated_user_features
                ]
                batch_user_embedding = self.model.embed_user(collated_user_features)
                user_embeddings.append(batch_user_embedding)
            user_embeddings = torch.vstack(user_embeddings)

            anime_embeddings = []
            for batch in tqdm.tqdm(self.anime_dataloader, desc="Anime Embeddings..."):
                collated_anime_features = [feature.to(self.device) for feature in batch]
                batch_anime_embedding = self.model.embed_feats(collated_anime_features)
                anime_embeddings.append(batch_anime_embedding)
            anime_embeddings = torch.vstack(anime_embeddings)

            user_interaction_mask = np.zeros((len(users), self.n_anime), dtype=np.int32)
            for i, (user, history) in enumerate(zip(users, histories)):
                user_interaction_mask[i, history] = 1
            # (n, 1, dim) -> (n, dim, 1)
            user_embeddings = user_embeddings.permute(0, 2, 1).detach().cpu().numpy()
            # -> (n, dim)
            user_embeddings = np.squeeze(user_embeddings, axis=2)
            # (a, 1, dim) -> (1, dim, a)
            anime_embeddings = anime_embeddings.permute(1, 2, 0).detach().cpu().numpy()
            # -> (dim, a)
            anime_embeddings = np.squeeze(anime_embeddings, axis=0)
            # (n, dim, 1) x (1, dim, a) -> (n, a)
            print(f"{user_embeddings.shape=} {anime_embeddings.shape=}")
            print(f"Commence God Operation")
            scores = user_embeddings @ anime_embeddings
            scores[user_interaction_mask] = -np.inf
            print(f"Commence Big Sort Energy")
            shows = np.argsort(-scores, axis=1)
            k_recommended = shows[:, :k]
            return scores, k_recommended


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
        user_embedding = self.embed_user(users)  # (b_s, 1, dim)

        pos_feature_embedding = self.embed_feats(pos_features)  # (b_s, n_pos, dim)
        neg_feature_embedding = self.embed_feats(neg_features)  # (b_s, n_neg, dim)

        # if torch.rand(1).item() < 0.1:
        #     print("=" * 20)
        #     print("user", users[0][:5, :5])
        #     print("pos", pos_features[0][:5, :5])
        #     print("neg", neg_features[0][:5, :5])
        negative_cos_sim = (
            torch.linalg.vecdot(user_embedding, pos_feature_embedding, dim=-1)
            / self.temperature
        )
        positive_cos_sim = (
            torch.linalg.vecdot(user_embedding, neg_feature_embedding, dim=-1)
            / self.temperature
        )
        # assert negative_cos_sim.shape == (b_s,)
        # if torch.rand(1).item() < 0.1:
        #     print(
        #         f"{user_embedding.shape=} / {pos_feature_embedding.shape=} / {neg_feature_embedding.shape=}"
        #     )
        #     # print(f"{user_embedding[:5, :5]=}")
        #     # print(f"{pos_feature_embedding[:5, :5]=}")
        #     print(
        #         f"{negative_cos_sim.mean().item()=} / {positive_cos_sim.mean().item()=}"
        #     )

        # BCE
        data = torch.cat([positive_cos_sim, negative_cos_sim], dim=1)
        labels = torch.cat(
            [torch.ones_like(positive_cos_sim), torch.zeros_like(negative_cos_sim)],
            dim=1,
        ).to(user_embedding.device)
        # print(f"{data.shape=} {data.dtype=} {data.device}")
        # print(f"{labels.shape=} {labels.dtype=} {labels.device}")
        # print(f"{torch.isnan(data).any()} {torch.isinf(data).any()}")
        loss = torch.nn.functional.binary_cross_entropy_with_logits(data, labels)
        return {
            "loss": loss,
            "positive_term": positive_cos_sim.mean().item(),
            "negative_term": negative_cos_sim.mean().item(),
        }

        print()

        # Contrastive loss
        # negative_cos_sim_exp = torch.exp(negative_cos_sim)
        # positive_cos_sim_exp = torch.exp(positive_cos_sim)

        # pos_term = positive_cos_sim_exp.sum(dim=1)
        # neg_term = negative_cos_sim_exp.sum(dim=1)

        # loss = -torch.log(pos_term / (pos_term + neg_term)).mean()
        # return loss
