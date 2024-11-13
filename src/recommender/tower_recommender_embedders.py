from enum import Enum
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn

from utils.data.data_classes import Anime, User


class UserEmbedders(Enum):
    USER_EMBEDDING = "user_embedding"
    SHARED_PRESERVED_AVG = "shared_preserved_avg"
    DIFFERENT_PRESERVED_AVG = "different_preserved_avg"


class ItemEmbedders(Enum):
    ID_EMBEDDING = "id_embedding"
    TITLE_PRETRAINED_EMBEDDING = "title_pretrained_embedding"


def collate_item_feature(
    features: Tuple[torch.Tensor], item_embedder: List[Tuple[str, dict]]
):
    collated_features = []
    for (feature_name, feature_metadata), feature in zip(item_embedder, zip(*features)):
        assembled_feature = None
        match feature_name:
            case ItemEmbedders.ID_EMBEDDING.value:
                assembled_feature = torch.stack(feature, dim=0)
            case ItemEmbedders.TITLE_PRETRAINED_EMBEDDING.value:
                assembled_feature = torch.stack(feature, dim=0)
            case _:
                raise NotImplementedError(f"Unrecognized feature {item_embedder}")
        collated_features.append(assembled_feature)
    return collated_features


def collate_user_feature(
    features: Tuple[torch.Tensor], user_embedder: List[Tuple[str, dict]]
):
    collated_features = []
    zipped_features_over_batch = zip(*features)
    for (feature_name, feature_metadata), feature in zip(
        user_embedder, zipped_features_over_batch
    ):
        assembled_feature = None
        match feature_name:
            case UserEmbedders.USER_EMBEDDING.value:
                assembled_feature = torch.stack(feature, dim=0)
            case UserEmbedders.SHARED_PRESERVED_AVG.value:
                # tuple of list of histories
                max_len = max([len(hist) for hist in feature])
                padded_histories = [
                    torch.nn.functional.pad(
                        hist, (0, max_len - len(hist)), "constant", 0
                    )
                    for hist in feature
                ]
                padding_mask = [
                    torch.cat(
                        (
                            torch.ones(len(hist), dtype=torch.bool),
                            torch.zeros(max_len - len(hist), dtype=torch.bool),
                        )
                    )
                    for hist in feature
                ]
                assembled_feature = (
                    torch.vstack(padded_histories),
                    torch.vstack(padding_mask),
                )

            case _:
                raise NotImplementedError(f"Unrecognized feature {user_embedder}")
        collated_features.append(assembled_feature)
    return collated_features


def user_pos_neg_collator(
    data: List[Tuple[torch.Tensor]],
    user_embedder: List[Tuple[str, dict]],
    item_embedder: List[Tuple[str, dict]],
):
    users, pos_feats, neg_feats = zip(*data)
    collated_users = collate_user_feature(users, user_embedder)
    collated_pos_feats = collate_item_feature(pos_feats, item_embedder)
    collated_neg_feats = collate_item_feature(neg_feats, item_embedder)

    return collated_users, collated_pos_feats, collated_neg_feats


def item_collator(
    data: List[Tuple[torch.Tensor]], item_embedder: List[Tuple[str, dict]]
):
    return collate_item_feature(data, item_embedder)


def featurize_item_or_items(
    animes: List[Anime], item_embedder: Tuple[str, dict], auxiliary_data: dict
):
    if type(animes) == Anime:
        animes = [animes]
    feature_name, feature_metadata = item_embedder
    match feature_name:
        case ItemEmbedders.ID_EMBEDDING.value:
            return torch.tensor([anime.id for anime in animes])
        case ItemEmbedders.TITLE_PRETRAINED_EMBEDDING.value:
            assert (
                "npy_title_pretrained_embedding" in auxiliary_data
            ), "Title embeddings not found in auxiliary data"
            return torch.vstack(
                [
                    torch.from_numpy(
                        auxiliary_data["npy_title_pretrained_embedding"][anime.id]
                    )
                    for anime in animes
                ]
            )
        case _:
            raise NotImplementedError(f"Unrecognized feature {feature_name}")


def featurize_singular_user(
    user: User,
    history: Tuple[List[Anime], np.ndarray[float]],
    user_embedder: Tuple[str, dict],
    auxiliary_data: dict,
):
    feature_name, feature_metadata = user_embedder
    match feature_name:
        case UserEmbedders.USER_EMBEDDING.value:
            return torch.tensor([user.cuid])
        case UserEmbedders.SHARED_PRESERVED_AVG.value:
            return torch.tensor([anime.id for anime in history[0]], dtype=torch.long)
        case _:
            raise NotImplementedError(f"Unrecognized user embedder {user_embedder}")


class HistoryEmbedder(nn.Module):
    def __init__(self, embedder: nn.Module):
        super().__init__()
        self.embedder = embedder

    def forward(self, data: Tuple[torch.Tensor, torch.Tensor]):
        """
        Assume that the values to embed are in the last dimension
        """
        batch_history: torch.Tensor = data[0]
        batch_mask: torch.Tensor = data[1]

        # mask: (n, ..., s)
        # (n, ..., s) -> (n, ..., s, d)
        embeddings = self.embedder(batch_history)
        # (n, ..., s, d) * (n, ..., s, 1)
        masked_embeddings = embeddings * batch_mask.unsqueeze(-1)
        # (n, ..., s) -> (n, ...) -> (n, ..., 1)
        avg_constants = torch.sum(batch_mask, axis=-1, keepdim=True)
        avg_constants = torch.maximum(avg_constants, torch.ones_like(avg_constants))
        # (n, ..., s, d) -> (n, ..., d)
        summed_embeddings = torch.sum(masked_embeddings, axis=-2)
        # (n, ..., d) / (n, ..., 1) -> (n, ..., d)
        avg_embeddings = summed_embeddings / avg_constants
        # print(f"{type(data)=} {len(data)=}")
        # print(f"{type(batch_history)=} {batch_history.shape=}")
        # print(f"{type(batch_mask)=} {batch_mask.shape=}")
        # print(f"{avg_embeddings.shape=}")

        return avg_embeddings.unsqueeze(1)


def build_all_embedders(
    user_embedders_spec: List[Tuple[str, dict]],
    item_embedders_spec: List[Tuple[str, dict]],
    n_users: int,
    n_anime: int,
):
    lut = {}
    user_embedders = []
    item_embedders = []

    for embedder_type, embedder_metadata in item_embedders_spec:
        embedder = None
        match embedder_type:
            case ItemEmbedders.ID_EMBEDDING.value:
                embedder = nn.Embedding(
                    n_anime, embedder_metadata["embedding_dimension"]
                )
            case ItemEmbedders.TITLE_PRETRAINED_EMBEDDING.value:
                assert "model" in embedder_metadata
                embedder = build_arbitrary_mlp(embedder_metadata["model"])
            case _:
                raise NotImplementedError(
                    f"Embedder type {embedder_type} not implemented"
                )
        item_embedders.append(embedder)
        lut[embedder_type] = embedder

    # Process user embedders afterwards
    for embedder_type, embedder_metadata in user_embedders_spec:
        embedder = None
        match embedder_type:
            case UserEmbedders.USER_EMBEDDING.value:
                embedder = nn.Embedding(
                    n_users, embedder_metadata["embedding_dimension"]
                )
            case UserEmbedders.SHARED_PRESERVED_AVG.value:
                assert ItemEmbedders.ID_EMBEDDING.value in lut
                anime_id_embedder = lut[ItemEmbedders.ID_EMBEDDING.value]
                embedder = HistoryEmbedder(embedder=anime_id_embedder)
            case _:
                raise NotImplementedError(
                    f"Embedder type {embedder_type} not implemented"
                )
        user_embedders.append(embedder)
        lut[embedder_type] = embedder

    user_embedders = nn.ModuleList(user_embedders)
    item_embedders = nn.ModuleList(item_embedders)
    return user_embedders, item_embedders


def fetch_auxiliary_data(auxiliary_data: dict):
    res = {}
    for k, v in auxiliary_data.items():
        if k.startswith("npy_"):
            res[k] = np.load(v)
        else:
            raise NotImplementedError(f"Cannot fetch auxiliary data for key {k}")
    return res


def build_arbitrary_mlp(definition: list):
    modules = []
    if not definition:
        return nn.Identity()
    for val in definition:
        if type(val) == str:
            match val:
                case "relu":
                    modules.append(nn.ReLU())
                case _:
                    raise ValueError(f"Invalid module definition {val}")
        elif type(val) == list:
            if len(val) != 2:
                raise ValueError(f"Invalid module definition {val}")
            inp, out = val
            modules.append(nn.Linear(inp, out))
        else:
            raise ValueError(f"Type of {val=} {type(val)=} not supported")
    return nn.Sequential(*modules)


def recursive_to_device(data: torch.Tensor | list, device: SystemError):
    if type(data) == torch.Tensor:
        return data.to(device)
    elif type(data) == list or type(data) == tuple:
        return [recursive_to_device(d, device) for d in data]
    else:
        raise NotImplementedError(f"Data type {type(data)} not supported")
