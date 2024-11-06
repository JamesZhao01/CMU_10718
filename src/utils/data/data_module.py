from dataclasses import dataclass, field
import hashlib
import json
import os
import time
from typing import Dict, List, Literal, Optional

import numpy as np
import pandas as pd
import tqdm
from sklearn.metrics import ndcg_score
from dataclass_wizard import JSONWizard
from utils.data.data_classes import Anime, Genre, Type, User

"""
    DataModule is a class that allows for unified parsing of data.
    
    It is also serializable / unserializable for consistency / reduced load on repeated
    runs of the same data. 
    
    cuid = canonical user id
    caid = cnaonical anime id
    
    a canonical id is a remapping of ids to a dense increasing sequence 

    E.x.
    anime_id | canonical_anime_id|
    ------------------------------
    34       | 0
    22       | 1
    97       | 2
"""


@dataclass
class DataModule(JSONWizard):
    k: int
    path: str
    thresholded_watch_history: int
    normalize_unrated: bool
    verbose: bool

    holdout_type: Literal["user", "interaction"]
    rng: np.random.Generator = field(default_factory=np.random.default_rng(42))

    user_mapping: Dict[int, User] = field(default_factory=dict)
    anime_mapping: Dict[int, Anime] = field(default_factory=dict)
    canonical_user_mapping: Dict[int, User] = field(default_factory=dict)
    canonical_anime_mapping: Dict[int, Anime] = field(default_factory=dict)
    anime_id_to_caid: Dict[int, int] = field(default_factory=dict)
    caid_to_anime_id: Dict[int, int] = field(default_factory=dict)
    user_id_to_cuid: Dict[int, int] = field(default_factory=dict)
    cuid_to_user_id: Dict[int, int] = field(default_factory=dict)
    cuids: List[int] = field(default_factory=list)

    max_anime_count: int = -1
    max_user_count: int = -1

    train_cuids: List[int] = field(default_factory=list)
    val_cuids: List[int] = field(default_factory=list)
    test_cuids: List[int] = field(default_factory=list)

    initialized: bool = False

    def __post_init__(self):
        assert self.holdout_type in ["interaction"]
        if self.initialized:
            return
        self.initialized = True
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
                        if self.normalize_unrated and rating == -1
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
                id=user_id,
                cuid=canonical_user_id,
                watch_history=np.array(filtered_watch_history),
                rating_history=np.array(filtered_rating_history),
                imputed_history=np.array(imputed_rating_history),
                k=self.k,
                max_anime_count=self.max_anime_count,
                lazy_store=False,
            )
            self.user_mapping[user.id] = user
            self.canonical_user_mapping[user.cuid] = user
            self.user_id_to_cuid[user.id] = user.cuid
            self.cuid_to_user_id[user.cuid] = user.id

        self.max_anime_count = len(self.anime_mapping)
        self.max_user_count = len(self.user_mapping)
        self.cuids = range(self.max_user_count)

        if self.holdout_type == "user":
            train_size = int(0.8 * len(self.user_ids))
            val_size = int(0.1 * len(self.user_ids))

            self.train_cuids = self.cuids[:train_size]
            self.val_cuids = self.cuids[train_size : train_size + val_size]
            self.test_cuids = self.cuids[train_size + val_size :]
        elif self.holdout_type == "interaction":
            self.train_cuids = self.cuids
            self.val_cuids = []
            self.test_cuids = []

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
