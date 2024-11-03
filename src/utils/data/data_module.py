from dataclasses import dataclass
import hashlib
import json
import os
import time
from typing import List, Optional

import numpy as np
import pandas as pd
import tqdm
from sklearn.metrics import ndcg_score
from utils.data.data_classes import Serializable

from utils.data.data_classes import Anime, Genre, Type, User

"""
    DataModule is a class that allows for unified parsing of data.
    
    It is also serializable / unserializable for consistency / reduced load on repeated
    runs of the same data. 
"""


@dataclass
class DataModule(Serializable):
    def to_dict(self) -> dict:
        return {
            "k": self.k,
            "path": self.path,
            "normalize_unrated": self.normalize_unrated,
            "threshold_watch_history": self.threshold_watch_history,
            "verbose": self.verbose,
            "holdout_type": self.holdout_type,
            "user_mapping": {k: v.to_dict() for k, v in self.user_mapping.items()},
            "anime_mapping": {k: v.to_dict() for k, v in self.anime_mapping.items()},
            "canonical_anime_mapping": {
                k: v.to_dict() for k, v in self.canonical_anime_mapping.items()
            },
            "canonical_user_mapping": {
                k: v.to_dict() for k, v in self.canonical_user_mapping.items()
            },
            "anime_id_to_caid": self.anime_id_to_caid,
            "caid_to_anime_id": self.caid_to_anime_id,
            "user_id_to_cuid": self.user_id_to_cuid,
            "cuid_to_user_id": self.cuid_to_user_id,
            "user_ids": self.user_ids,
            "max_anime_count": self.max_anime_count,
            "max_user_count": self.max_user_count,
            "train_indices": self.train_indices,
            "val_indices": self.val_indices,
            "test_indices": self.test_indices,
        }

    def from_dict(cls, data: dict):
        return cls(**data)

    def assign_propoperties(self, data: dict):
        self.user_mapping = {}
        self.anime_mapping = {}
        self.canonical_anime_mapping = {}
        self.canonical_user_mapping = {}
        self.anime_id_to_caid = {}
        self.caid_to_anime_id = {}
        self.user_id_to_cuid = {}
        self.cuid_to_user_id = {}
        self.user_ids = []

        self.max_anime_count: int = -1
        self.max_user_count: int = -1
        self.user_ids: List[int] = []

        self.train_indices: Optional[List[int]] = []
        self.val_indices: Optional[List[int]] = []
        self.test_indices: Optional[List[int]] = []

    def __init__(
        self,
        path: str,
        k: int = 10,
        normalize_unrated=True,
        threshold_watch_history=20,
        verbose=True,
        holdout_type="user",
        *args,
        **kwargs,
    ):
        self.k: int = k
        self.path: str = path
        self.threshold_watch_history: int = threshold_watch_history
        self.normalize_unrated: bool = normalize_unrated
        self.threshold_watch_history: int = threshold_watch_history
        self.verbose = verbose
        assert holdout_type in ["user", "item"]
        self.holdout_type = holdout_type
        self.rng: np.random.Generator = np.random.default_rng(42)

        self.user_mapping = {}
        self.anime_mapping = {}
        self.canonical_anime_mapping = {}
        self.canonical_user_mapping = {}
        self.anime_id_to_caid = {}
        self.caid_to_anime_id = {}
        self.user_id_to_cuid = {}
        self.cuid_to_user_id = {}
        self.user_ids = []

        self.max_anime_count: int = -1
        self.max_user_count: int = -1
        self.user_ids: List[int] = []

        self.train_indices: Optional[List[int]] = []
        self.val_indices: Optional[List[int]] = []
        self.test_indices: Optional[List[int]] = []

    def construct(self):
        animes = pd.read_csv(os.path.join(self.path, "anime.csv"))
        users = pd.read_csv(os.path.join(self.path, "rating.csv"))

        # Gets all the information about the animes
        anime_iterator = enumerate(animes.iterrows())
        if self.verbose:
            anime_iterator = tqdm.tqdm(
                anime_iterator,
                "parsing animes...",
                total=len(animes),
            )
        for canonical_anime_id, (anime_id, anime_df) in anime_iterator:
            genres = []
            if not pd.isna(anime_df["genre"]):
                for unprocessed_string in anime_df["genre"].split(","):
                    stripped = unprocessed_string.strip()
                    if stripped == "Sci-Fi":
                        genre = "Sci_Fi"
                    elif stripped == "Super Power":
                        genre = "Super_Power"
                    elif stripped == "Slice of Life":
                        genre = "Slice_of_Life"
                    elif stripped == "Shounen Ai":
                        genre = "Shounen_Ai"
                    elif stripped == "Shoujo Ai":
                        genre = "Shoujo_Ai"
                    elif stripped == "Martial Arts":
                        genre = "Martial_Arts"
                    else:
                        genre = stripped
                    genres.append(getattr(Genre, genre))

            if pd.isna(anime_df["type"]):
                type = "Unknown"
            else:
                type = anime_df["type"]
            type = getattr(Type, type)

            if pd.isna(anime_df["rating"]):
                rating = 6.57  # median value
            else:
                rating = anime_df["rating"]

            if pd.isna(anime_df["members"]):
                membership_count = -1
            else:
                membership_count = anime_df["members"]

            anime = Anime(
                id=canonical_anime_id,
                name=anime_df["name"],
                genres=genres,
                type=type,
                episodes=anime_df["episodes"],
                rating=rating,
                membership_count=membership_count,
            )
            self.anime_mapping[anime_df["anime_id"]] = anime
            self.canonical_anime_mapping[canonical_anime_id] = anime
            self.anime_id_to_caid[anime_df["anime_id"]] = canonical_anime_id
            self.caid_to_anime_id[canonical_anime_id] = anime_df["anime_id"]
        masking_set = np.zeros(max(self.anime_mapping.keys()) + 1, dtype=bool)
        masking_set[list(self.anime_mapping.keys())] = True

        # Gets all the user watch history
        user_iterator = users.groupby("user_id", sort=True)
        if self.verbose:
            user_iterator = tqdm.tqdm(
                user_iterator,
                "parsing users...",
                total=user_iterator.count(),
            )
        next_canonical_user_id = 0
        for user_id, user_df in user_iterator:
            anime_list: np.ndarray[int] = user_df["anime_id"].to_numpy(dtype=int)
            rating_list: np.ndarray[float] = user_df["rating"].to_numpy(dtype=float)

            assert len(anime_list) == len(rating_list)

            # Filters out ratings that aren't valid with our anime_list
            valid_indices = masking_set[anime_list]
            anime_list: np.ndarray[int] = anime_list[valid_indices]
            rating_list: np.ndarray[float] = rating_list[valid_indices]

            # Filter/Preprocess the data
            filtered_watch_history: list[int] = []
            filtered_rating_history: list[int] = []
            imputed_rating_history: list[int] = []
            for anime_id, rating in zip(anime_list, rating_list):
                # Filter out NSFW
                current_anime = self.anime_mapping[anime_id]
                if Genre.Hentai not in current_anime.genres:
                    filtered_watch_history.append(self.anime_id_to_caid[anime_id])
                    filtered_rating_history.append(
                        self.anime_mapping[anime_id].rating
                        if normalize_unrated and rating == -1
                        else rating
                    )
                    imputed_rating_history.append(
                        self.anime_mapping[anime_id].rating if rating == -1 else rating
                    )
            enough_watch_history = (
                np.count_nonzero(filtered_watch_history) >= self.threshold_watch_history
            )

            # If not enough history, exclude user
            if not enough_watch_history:
                continue
            canonical_user_id = next_canonical_user_id
            next_canonical_user_id += 1
            user = User(
                user_id,
                canonical_user_id,
                np.array(filtered_watch_history),
                np.array(filtered_rating_history),
                np.array(imputed_rating_history),
                self.k,
                self.max_anime_count,
                lazy_store=False,
            )
            self.user_mapping[user.id] = user
            self.canonical_user_mapping[user.cuid] = user
            self.user_id_to_cuid[user.id] = user.cuid
            self.cuid_to_user_id[user.cuid] = user.id
            user.generate_masked_history()

        self.max_anime_count = len(self.anime_mapping)
        self.max_user_count = len(self.user_mapping)

        # Split Data
        self.user_ids = list(sorted(self.user_mapping.keys()))

        if self.holdout_type == "user":
            train_size = int(0.8 * len(self.user_ids))
            val_size = int(0.1 * len(self.user_ids))

            self.train_indices = self.user_ids[:train_size]
            self.val_indices = self.user_ids[train_size : train_size + val_size]
            self.test_indices = self.user_ids[train_size + val_size :]
        elif self.holdout_type == "interaction":
            self.train_indices = self.user_ids
            self.val_indices = None
            self.test_indices = None

        if self.verbose:
            data_hash = hashlib.md5(
                json.dumps(self.user_ids).encode("utf-8")
            ).hexdigest()
            print(
                f"Number of Users: {len(self.user_ids)}, Hash[:8]: {data_hash[:6]}, Hash: {data_hash}"
            )
            print(
                f"Total Animes: {self.max_anime_count}, Total Users: {self.max_user_count}"
            )
