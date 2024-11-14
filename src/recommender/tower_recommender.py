from abc import ABC
from collections import defaultdict
from enum import Enum
from typing import List, Literal, Tuple
import time

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import tqdm
import os
import json

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
    build_all_embedders,
    fetch_auxiliary_data,
    recursive_to_device,
)
from utils.data.testbench import TestBench


class ScoringModifications(Enum):
    UPRANK = {
        "function": str,
        "score_weight": float,
        "uprank_weight": float,
    }
    NONE = {}

    def __new__(cls, parameters):
        obj = object.__new__(cls)
        obj.parameters = parameters
        return obj


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
        loss="bce",  # bce, info_nce, contrastive
        sim="cosine",  # cosine, dot
        scoring_modification=["none", {}],
        lr=1e-3,
        epochs=10,
        batch_size=32,
        dataset_args={},
        dataset_wrapper_args={},
        load_from="",
        save_to="",
        save_model=False,
        device="cuda",
        test_temp=1.0,
        sampling=None,
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
        self.loss = loss
        self.sim = sim
        self.scoring_modification_type = scoring_modification[0]
        self.scoring_modification_parameters = scoring_modification[1]
        self.test_temp = test_temp
        self.sampling = sampling

        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.load_from = load_from
        self.save_to = save_to
        self.save_model = save_model
        if self.save_to:
            os.makedirs(os.path.dirname(self.save_to), exist_ok=True)
            with open(f"{self.save_to}.log", "a") as f:
                relevant_args = {
                    "embedding_dimension": embedding_dimension,
                    "user_embedder": user_embedder,
                    "item_embedder": item_embedder,
                    "temperature": temperature,
                    "n_pos": n_pos,
                    "n_neg": n_neg,
                    "loss": loss,
                    "sim": sim,
                    "lr": lr,
                    "epochs": epochs,
                    "batch_size": batch_size,
                }
                f.write(f"(hyperparameters){json.dumps(relevant_args, indent=None)}\n")
        self.device = device

        self.model = TowerModel(
            n_users=n_users,
            n_anime=n_anime,
            embedding_dimension=embedding_dimension,
            user_embedders_spec=user_embedder,
            item_embedders_spec=item_embedder,
            temperature=temperature,
            loss=loss,
            sim=sim,
        ).to(self.device)
        self.testbench = TestBench(datamodule, should_return_ids=True)

        if self.load_from:
            self.model.load_state_dict(torch.load(self.load_from, weights_only=True))
        self.dataset = DatasetWrapper(
            datamodule=datamodule,
            user_embedder=user_embedder,
            item_embedder=item_embedder,
            n_pos=n_pos,
            n_neg=n_neg,
            auxiliary_data=dataset_wrapper_args,
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

        self.anime_dataset = AnimeEnumerator(
            datamodule, item_embedder, self.dataset.auxiliary_data
        )
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
        start_time = time.time()
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
                user = recursive_to_device(user, self.device)
                positive = recursive_to_device(positive, self.device)
                negative = recursive_to_device(negative, self.device)

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

            current_time = time.time()
            elapsed_time = current_time - start_time
            estimated_time = elapsed_time / (epoch + 1) * self.epochs
            estimated_remainder = estimated_time - elapsed_time
            print(
                f"[{epoch}] Loss: {epoch_loss:0.4f}. Time: {elapsed_time:0.2f} s / {estimated_time:0.2f} s. ETA: {estimated_remainder:0.2f} s"
            )
            metrics = self.testbench.full_evaluation(self, return_scores=False)
            print(f">>{epoch}|{loss}|{json.dumps(metrics, indent=None)}<<")
            with open(f"{self.save_to}.log", "a") as f:
                f.write(f">>{epoch}|{loss}|{json.dumps(metrics, indent=None)}<<\n")

        if self.save_model and self.save_to:
            print(f"Saving model to {self.save_to}")
            torch.save(self.model.state_dict(), self.save_to)
        with open(f"{self.save_to}.log", "a") as f:
            f.write(f"(Fin)\n")
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
            users, history_tuples = features
            user_features = [
                [
                    featurize_singular_user(
                        user, history_tuple, embedder, self.dataset.auxiliary_data
                    )
                    for embedder in self.user_embedder
                ]
                for user, history_tuple in zip(users, history_tuples)
            ]

            user_embeddings = []
            for batch in tqdm.tqdm(
                batchify(user_features, self.batch_size), desc="User Embeddings..."
            ):
                collated_user_features = collate_user_feature(batch, self.user_embedder)
                collated_user_features = recursive_to_device(
                    collated_user_features, self.device
                )
                batch_user_embedding = self.model.embed_user(collated_user_features)
                user_embeddings.append(batch_user_embedding)
            user_embeddings = torch.vstack(user_embeddings)

            anime_embeddings = []
            for batch in tqdm.tqdm(self.anime_dataloader, desc="Anime Embeddings..."):
                collated_anime_features = [feature.to(self.device) for feature in batch]
                batch_anime_embedding = self.model.embed_feats(collated_anime_features)
                anime_embeddings.append(batch_anime_embedding)
            anime_embeddings = torch.vstack(anime_embeddings)

            user_interaction_mask = np.zeros((len(users), self.n_anime), dtype=bool)
            print('history_tuples',len(history_tuples))
            for i, (user, history) in enumerate(zip(users, history_tuples)):
                anime_ids = np.array([anime.id for anime in history[0]])
                user_interaction_mask[i, anime_ids] = True
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

            scores = self.reweight(scores)
            scores[user_interaction_mask] = -np.inf
            print(f"Commence Big Sort Energy")
            if self.sampling == "uni":
                shows = np.argsort(-scores, axis=1)
                shows = shows[:, :3*k]
                k_recommended = np.zeros((len(users), k), dtype=int)
                for i, user in enumerate(users):
                    k_recommended[i] = np.random.choice(shows[i], k, replace=False)
            elif self.sampling == "boltzmann":
                shows = np.argsort(-scores, axis=1)
                shows = shows[:, :3*k]
                k_recommended = np.zeros((len(users), k), dtype=int)
                for i, user in enumerate(users):
                    probs = np.exp(scores[i, shows[i]] / self.test_temp)
                    probs /= probs.sum()
                    k_recommended[i] = np.random.choice(shows[i], k, p=probs, replace=False)
            else:
                shows = np.argsort(-scores, axis=1)
                k_recommended = shows[:, :k]
            return scores, k_recommended

    def reweight(self, scores):
        scoring_modification = ScoringModifications[
            self.scoring_modification_type.upper()
        ]
        match scoring_modification:
            case ScoringModifications.NONE:
                return scores
            case ScoringModifications.UPRANK:
                assert "function" in self.scoring_modification_parameters
                assert "weight" in self.scoring_modification_parameters
                match scoring_modification.parameters["function"]:
                    case "log":
                        function = np.log
                    case "sqrt":
                        function = np.sqrt
                    case _:
                        raise ValueError(f"Unrecognized function {function}")
                score_weight = self.scoring_modification_parameters["score_weight"]
                uprank_weight = self.scoring_modification_parameters["uprank_weight"]
                community_sizes = function(
                    np.array(
                        [
                            self.datamodule.canonical_anime_mapping[
                                cuid
                            ].membership_count
                            for cuid in range(self.datamodule.max_anime_count)
                        ]
                    )
                )

                # TODO James


class AnimeEnumerator(data.Dataset):
    def __init__(
        self,
        datamodule: DataModule,
        item_embedder: List[Tuple[str, dict]],
        auxiliary_data: dict,
    ):
        super().__init__()
        self.datamodule = datamodule
        self.item_embedder = item_embedder
        self.auxiliary_data = auxiliary_data

    def __len__(self):
        return self.datamodule.max_anime_count

    def __getitem__(self, idx):
        anime = self.datamodule.canonical_anime_mapping[idx]

        return [
            featurize_item_or_items(anime, embedder, self.auxiliary_data)
            for embedder in self.item_embedder
        ]


class DatasetWrapper(data.Dataset):
    def __init__(
        self,
        datamodule: DataModule,
        user_embedder: List[Tuple[str, dict]],
        item_embedder: List[Tuple[str, dict]],
        n_pos: int = 10,
        n_neg: int = 10,
        auxiliary_data: dict = {},
    ):
        super().__init__()
        self.datamodule = datamodule
        self.user_embedder = user_embedder
        self.item_embedder = item_embedder
        self.n_pos = n_pos
        self.n_neg = n_neg
        self.auxiliary_data = fetch_auxiliary_data(auxiliary_data)

    def __len__(self):
        return self.datamodule.max_user_count

    def __getitem__(self, idx):
        user = self.datamodule.canonical_user_mapping[idx]
        positives, history = user.partition_preserved(self.n_pos)
        positives = [
            self.datamodule.canonical_anime_mapping[anime] for anime in positives
        ]
        history = [self.datamodule.canonical_anime_mapping[anime] for anime in history]
        ratings = user.rating_history[user.preserved]

        history_tuple = (history, ratings)

        negatives = user.sample_negative(self.n_neg)
        negatives = [
            self.datamodule.canonical_anime_mapping[anime] for anime in negatives
        ]

        user_feature = [
            featurize_singular_user(user, history_tuple, embedder, self.auxiliary_data)
            for embedder in self.user_embedder
        ]
        pos_feature = [
            featurize_item_or_items(positives, embedder, self.auxiliary_data)
            for embedder in self.item_embedder
        ]
        neg_feature = [
            featurize_item_or_items(negatives, embedder, self.auxiliary_data)
            for embedder in self.item_embedder
        ]
        return user_feature, pos_feature, neg_feature


class TowerModel(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_anime: int,
        embedding_dimension: int,
        user_embedders_spec: str,
        item_embedders_spec: List[Tuple[str, dict]],
        temperature: float = 1,
        loss: Literal["bce", "contrastive", "info_nce"] = "bce",
        sim: Literal["cosine", "dot"] = "cosine",
    ):
        super().__init__()
        self.n_users = n_users
        self.n_anime = n_anime
        self.embedding_dimension = embedding_dimension
        self.temperature = temperature
        self.loss = loss
        self.sim = sim

        user_embedders, item_embedders = build_all_embedders(
            user_embedders_spec=user_embedders_spec,
            item_embedders_spec=item_embedders_spec,
            n_users=n_users,
            n_anime=n_anime,
        )
        self.user_embedders = user_embedders
        self.item_embedders = item_embedders

    def embed_user(self, feats: Tuple[torch.Tensor | Tuple[torch.Tensor]]):
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

        if self.sim == "cosine":
            pos_feature_embedding = pos_feature_embedding / torch.linalg.norm(
                pos_feature_embedding, dim=-1, keepdim=True, ord=2
            )
            neg_feature_embedding = neg_feature_embedding / torch.linalg.norm(
                neg_feature_embedding, dim=-1, keepdim=True, ord=2
            )

        negative_sim = (
            torch.linalg.vecdot(user_embedding, neg_feature_embedding, dim=-1)
            / self.temperature
        )
        positive_sim = (
            torch.linalg.vecdot(user_embedding, pos_feature_embedding, dim=-1)
            / self.temperature
        )

        # BCE
        if self.loss == "bce":
            data = torch.cat([positive_sim, negative_sim], dim=1)
            labels = torch.cat(
                [torch.ones_like(positive_sim), torch.zeros_like(negative_sim)],
                dim=1,
            ).to(user_embedding.device)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(data, labels)
            return {
                "loss": loss,
                "positive_term": positive_sim.mean().item(),
                "negative_term": negative_sim.mean().item(),
            }

        # Contrastive loss
        if self.loss == "contrastive":
            negative_sim_exp = torch.exp(negative_sim)
            positive_sim_exp = torch.exp(positive_sim)

            pos_term = positive_sim_exp.sum(dim=1)
            neg_term = negative_sim_exp.sum(dim=1)

            loss = -torch.log(pos_term / (pos_term + neg_term)).mean()
            return {
                "loss": loss,
                "positive_term": positive_sim.mean().item(),
                "negative_term": negative_sim.mean().item(),
            }

        # Info NCE
        if self.loss == "nce":
            negative_sim_exp = torch.exp(negative_sim)
            positive_sim_exp = torch.exp(positive_sim)

            neg_term = negative_sim_exp.sum(dim=1, keepdim=True)
            terms = -torch.log(positive_sim_exp / (positive_sim_exp + neg_term))
            sum_terms = terms.sum(dim=1)
            loss = sum_terms.mean()

            return {
                "loss": loss,
                "positive_term": positive_sim.mean().item(),
                "negative_term": negative_sim.mean().item(),
            }

        # return loss
