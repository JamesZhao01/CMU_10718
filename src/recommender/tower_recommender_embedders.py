from enum import Enum
from typing import List, Tuple

import torch
import torch.nn as nn

from utils.data.data_classes import Anime, User


class UserEmbedders(Enum):
    USER_EMBEDDING = "user_embedding"
    SHARED_PRESERVED_AVG = "shared_preserved_avg"
    DIFFERENT_PRESERVED_AVG = "different_preserved_avg"


class ItemEmbedders(Enum):
    ID_EMBEDDING = "id_embedding"


def collate_item_feature(
    features: Tuple[torch.Tensor], item_embedder: List[Tuple[str, dict]]
):
    collated_features = []
    for (feature_name, feature_metadata), feature in zip(item_embedder, zip(*features)):
        assembled_feature = None
        match feature_name:
            case ItemEmbedders.ID_EMBEDDING.value:
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


def featurize_item_or_items(animes: List[Anime], item_embedder: Tuple[str, dict]):
    if type(animes) == Anime:
        animes = [animes]
    feature_name, feature_metadata = item_embedder
    match feature_name:
        case ItemEmbedders.ID_EMBEDDING.value:
            return torch.tensor([anime.id for anime in animes])
        case _:
            raise NotImplementedError(f"Unrecognized feature {feature_name}")


def featurize_singular_user(
    user: User, history: List[Anime], user_embedder: Tuple[str, dict]
):
    feature_name, feature_metadata = user_embedder
    match feature_name:
        case UserEmbedders.USER_EMBEDDING.value:
            return torch.tensor([user.cuid])
        case _:
            raise NotImplementedError(f"Unrecognized user embedder {user_embedder}")


def build_user_embedder(
    embedder_type: str, embedder_metadata: dict, n_users: int, **kwargs
):
    match embedder_type:
        case UserEmbedders.USER_EMBEDDING.value:
            return nn.Embedding(n_users, embedder_metadata["embedding_dimension"])
        case _:
            raise NotImplementedError(f"Embedder type {embedder_type} not implemented")


def build_item_embedder(
    embedder_type: str, embedder_metadata: dict, n_anime: int, **kwargs
):
    match embedder_type:
        case ItemEmbedders.ID_EMBEDDING.value:
            return nn.Embedding(n_anime, embedder_metadata["embedding_dimension"])
        case _:
            raise NotImplementedError(f"Embedder type {embedder_type} not implemented")
