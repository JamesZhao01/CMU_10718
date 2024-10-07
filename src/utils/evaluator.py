import math
import os
import random
import time
from enum import Enum

import numpy as np
import pandas as pd
import tqdm
from sklearn.metrics import ndcg_score


class Type(Enum):
    TV = 1
    OVA = 2
    Movie = 3
    Special = 4
    ONA = 5
    Music = 6
    Unknown = 7


class Genre(Enum):
    Drama = 1
    Romance = 2
    School = 3
    Supernatural = 4
    Action = 5
    Adventure = 6
    Fantasy = 7
    Magic = 8
    Military = 9
    Shounen = 10
    Comedy = 11
    Historical = 12
    Parody = 13
    Samurai = 14
    Sci_Fi = 15
    Thriller = 16
    Sports = 17
    Super_Power = 18
    Space = 19
    Slice_of_Life = 20
    Mecha = 21
    Music = 22
    Mystery = 23
    Seinen = 24
    Martial_Arts = 25
    Vampire = 26
    Shoujo = 27
    Horror = 28
    Police = 29
    Psychological = 30
    Demons = 31
    Ecchi = 32
    Josei = 33
    Shounen_Ai = 34
    Game = 35
    Dementia = 36
    Harem = 37
    Cars = 38
    Kids = 39
    Shoujo_Ai = 40
    Hentai = 41
    Yaoi = 42
    Yuri = 43


class Anime:
    def __init__(
        self,
        id: int,
        name: str,
        genres: set[Genre],
        type: Type,
        episodes: int,
        rating: float,
        membership_count: int,
    ):
        self.id: int = id
        self.name: str = name
        self.genres: set[Genre] = genres
        self.type: Type = type
        self.episodes: int = episodes
        self.rating: float = rating
        self.membership_count: int = membership_count


class User:
    def __init__(
        self,
        id: int,
        watch_history: np.ndarray[int],
        rating_history: np.ndarray[float],
        imputed_rating_history: np.ndarray[float],
        k: int,
        max_anime_count: int,
    ):
        self.id: int = id
        self.watch_history: np.ndarray[int] = watch_history
        # Usually, ratings are ints, but if we normalize the unrated shows to the average
        # they will be a float.
        # self.rating_history:list[float] = rating_history
        self.rating_history: np.ndarray[float] = rating_history
        self.imputed_rating_history: np.ndarray[float] = imputed_rating_history
        self.k = k
        self.masked_watch_history: np.ndarray[int] = np.array([])
        self.masked_rating_history: np.ndarray[float] = np.array([])
        self.masked_imputed_history: np.ndarray[float] = np.array([])
        self.max_anime_count: int = max_anime_count

    def generate_masked_history(self):
        """
        Generates the masked history based on the current watch/rating history.
        Note that this should be called after the filtering process, so that
        the masked information does not include any filtered out data.
        """
        if len(self.masked_watch_history) == 0:
            # Randomly select indices to mask (subsample)
            self.mask = np.random.choice(
                len(self.watch_history), size=self.k, replace=False
            )
            preserved_data = np.setdiff1d(np.arange(len(self.watch_history)), self.mask)
            self.preserved_canonical_ids = self.watch_history[preserved_data]

            # Subsample both arrays using the selected indices
            self.masked_watch_history = self.watch_history[preserved_data]
            self.masked_rating_history = self.rating_history[preserved_data]
            self.masked_imputed_history = self.imputed_rating_history[preserved_data]

    def get_history(self):
        """
        Returns a one-hot encoding of the rating history of the user.
        The index is the anime id.
        (Note that index 0 is always just 0 since anime id starts with 1).
        """

        one_hot_encoding = np.zeros(self.max_anime_count, dtype=float)
        # Populate the array with ratings at indices corresponding to anime_id
        one_hot_encoding[self.watch_history] = self.rating_history
        return one_hot_encoding

    def get_imputed_history(self):
        one_hot_encoding = np.zeros(self.max_anime_count, dtype=float)
        # Populate the array with ratings at indices corresponding to anime_id
        one_hot_encoding[self.watch_history] = self.imputed_rating_history
        return one_hot_encoding

    def get_masked_history(self):
        """
        Same as get_history, but some of the information is masked out.
        This assumes that generate_masked_history has been called earlier.
        """

        one_hot_encoding = np.zeros(self.max_anime_count, dtype=float)
        # Populate the one-hot list with ratings at indices corresponding to anime_id

        one_hot_encoding[self.masked_watch_history] = self.masked_rating_history
        return one_hot_encoding

    def get_imputed_masked_history(self):
        one_hot_encoding = np.zeros(self.max_anime_count, dtype=float)
        one_hot_encoding[self.masked_watch_history] = self.masked_imputed_history
        return one_hot_encoding


class Evaluator:
    def __init__(self, path: str, normalize_unrated=True):
        self.animes = pd.read_csv(os.path.join(path, "anime.csv"))
        self.users = pd.read_csv(os.path.join(path, "rating.csv"))
        self.k = 10
        self.threshold_watch_history = 20
        self.user_mapping = {}
        self.anime_mapping = {}
        self.canonical_anime_mapping = {}
        self.anime_id_to_canonical_id = {}
        self.canonical_id_to_anime_id = {}
        self.user_ids = []
        self.current_idx = 0
        self.normalize_unrated = normalize_unrated

        print(f"{normalize_unrated=}")
        np.random.seed(42)
        t = 0
        # Gets all the information about the animes
        for anime_id, anime_df in tqdm.tqdm(
            self.animes.iterrows(), "parsing animes...", total=len(self.animes)
        ):
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
                id=t,
                name=anime_df["name"],
                genres=genres,
                type=type,
                episodes=anime_df["episodes"],
                rating=rating,
                membership_count=membership_count,
            )
            self.anime_mapping[anime_df["anime_id"]] = anime
            self.canonical_anime_mapping[t] = anime
            self.anime_id_to_canonical_id[anime_df["anime_id"]] = t
            self.canonical_id_to_anime_id[t] = anime_df["anime_id"]
            t += 1
        self.max_anime_count = len(self.anime_mapping)

        masking_set = np.zeros(max(self.anime_mapping.keys()) + 1, dtype=bool)
        masking_set[list(self.anime_mapping.keys())] = True

        # Gets all the user watch history
        for user_id, user_df in tqdm.tqdm(
            self.users.groupby("user_id"), "parsing users..."
        ):
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
                    filtered_watch_history.append(
                        self.anime_id_to_canonical_id[anime_id]
                    )
                    filtered_rating_history.append(
                        self.anime_mapping[anime_id].rating
                        if normalize_unrated and rating == -1
                        else rating
                    )
                    imputed_rating_history.append(
                        self.anime_mapping[anime_id].rating if rating == -1 else rating
                    )
            user = User(
                user_id,
                np.array(filtered_watch_history),
                np.array(filtered_rating_history),
                np.array(imputed_rating_history),
                self.k,
                self.max_anime_count,
            )

            # Filters out users that have too little watch history.
            enough_watch_history = (
                np.count_nonzero(user.watch_history) >= self.threshold_watch_history
            )
            if enough_watch_history:
                self.user_mapping[user.id] = user
                # We need the history -- James
                user.generate_masked_history()
        print(f"Total Animes: {len(self.anime_mapping)}")
        print(f"Total Users: {len(self.user_mapping)}")
        self.idxs = list(self.user_mapping.keys())

        # Split the data into train/test/val
        train_size = int(0.8 * len(self.idxs))
        val_size = int(0.1 * len(self.idxs))

        self.train_indices = self.idxs[:train_size]
        self.val_ids = self.idxs[train_size : train_size + val_size]
        self.test_ids = self.idxs[train_size + val_size :]

    def get_anime_info(self, id: int):
        return self.anime_mapping[id]

    ###################################################################################
    # This section is for evaluating on a single item. It's probably not what we'll use
    # in the long term, but can work for baselines and initial debugging.
    def start_eval_on_single_test(self):
        """
        Run this method to start the evaluation program.
        You MUST call end_eval_runtime() as soon as possible to get evaluation results.
        """
        user_id = self.test_ids[self.current_idx]
        user: User = self.user_mapping[user_id]
        user.generate_masked_history()
        user_masked_watch_history = user.get_imputed_masked_history()
        self.start_time = time.time()
        return user_masked_watch_history[None, ...], self.k

    def end_eval_on_single_test(self, k_recommended_shows: np.ndarray[int]):
        """
        Run this method to end the evaluation program.
        I expect k_recommended_shows to be of shape [test_set_size, k]. Test size should be 1.
        The columns should be ordered by ranking with 0 being the highest and 1 being
        the lowest.
        """
        total_runtime = time.time() - self.start_time

        k_recommended_shows = k_recommended_shows.squeeze()
        user_id = self.test_ids[self.current_idx]
        user: User = self.user_mapping[user_id]
        complete_watch_history = user.get_history()
        user_masked_watch_history = user.get_masked_history()
        ground_truth_of_masked_history = (
            complete_watch_history - user_masked_watch_history
        )

        pred = np.zeros(MAX_ANIME_COUNT, dtype=int)
        # Sort the indices and generate decreasing values
        decreasing_values = np.arange(len(k_recommended_shows), 0, -1)

        # Assign the decreasing values to the respective indices
        pred[k_recommended_shows] = decreasing_values

        score = ndcg_score(
            ground_truth_of_masked_history[None, ...], pred[None, ...], k=self.k
        )
        print(f"This model took {total_runtime} seconds.")
        print(f"Out of an optimal score of 1.0, you scored {score}.")
        return total_runtime, score

    ####################################################################################

    def start_eval_test_set(self):
        """
        Returns:
        - user_masked_watch_history: Masked user watch history of shape [num_of_users, MAX_ANIME_COUNT]
        - k: number of anime recommendations your model should present
        """

        user_masked_watch_history = np.zeros((len(self.test_ids), self.max_anime_count))
        for i, id in enumerate(self.test_ids):
            user: User = self.user_mapping[id]
            user.generate_masked_history()
            user_masked_watch_history[i] = user.get_masked_history()

        self.start_time = time.time()
        return user_masked_watch_history, self.k

    def end_eval_test_set(self, k_recommended_shows: np.ndarray[int]):
        """
        Run this method to end the evaluation program.
        I expect k_recommended_shows to be of shape [test_set_size, k].
        The columns should be ordered by ranking with 0 being the highest and 1 being
        the lowest.

        if is_anime_id is true, then each row of k_recommended_shows should be a list of anime ids,
            in ranked order
        """
        total_runtime = time.time() - self.start_time

        complete_watch_history = np.zeros((len(self.test_ids), self.max_anime_count))
        user_masked_watch_history = np.zeros((len(self.test_ids), self.max_anime_count))
        ground_truth_of_masked_history = np.zeros(
            (len(self.test_ids), self.max_anime_count)
        )
        for i, id in enumerate(self.test_ids):
            user: User = self.user_mapping[id]
            complete_watch_history[i] = user.get_imputed_history()
            user_masked_watch_history[i] = user.get_imputed_masked_history()
            ground_truth_of_masked_history[i] = (
                complete_watch_history[i] - user_masked_watch_history[i]
            )

        pred = np.zeros((len(self.test_ids), self.max_anime_count), dtype=int)

        # Sort the indices and generate decreasing values
        decreasing_values = np.arange(k_recommended_shows.shape[1], 0, -1)

        # Assign the decreasing values to the respective indices
        # pred[np.array(k_recommended_shows)] = decreasing_values
        pred[np.arange(len(self.test_ids))[:, None], k_recommended_shows] = (
            decreasing_values
        )

        score = ndcg_score(ground_truth_of_masked_history, pred, k=self.k)
        print(f"This model took {total_runtime:0.4f} seconds.")
        print(f"Out of an optimal score of 1.0, you scored {score:0.4f}.")
        return total_runtime, score
