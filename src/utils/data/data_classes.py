from enum import Enum
from typing import Tuple

import numpy as np


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
    """
    Copilot Docs:

    User class that holds information about the user's watch history, rating history, and imputed
    rating history. The class also provides methods to generate feature vectors for the user's
    watch history and imputed watch history. The class also provides methods to reshuffle the
    watch history and generate masked history for training.

    Additional Notes:

    The "masked" history is designed to be TEST-time visible only. It should not be accessed
        conventionally during training.
    The "preserved" history is the training-time visible history. Utility methods are provided
        to allow for sampling from within this pool (e.x. positives / negatives)
    The "negative" history is the pool of items that the user has not interacted with -- and can be
        used for sampling negatives during training, outside of the "masked" and "preserved" pool.

    """

    def __init__(
        self,
        id: int,
        canonical_user_id: int,
        watch_history: np.ndarray[int],
        rating_history: np.ndarray[float],
        imputed_history: np.ndarray[float],
        k: int,
        max_anime_count: int,
        # only generate feature vectors on the fly. This prevents storing excessive amounts of
        # sparse vectors to save memory
        lazy_store: bool = False,
    ):
        self.id: int = id
        self.cuid: int = canonical_user_id
        self.watch_history: np.ndarray[int] = watch_history
        self.rating_history: np.ndarray[float] = rating_history
        self.imputed_rating_history: np.ndarray[float] = imputed_history
        self.k = k
        self.max_anime_count: int = max_anime_count
        self.rng = np.random.default_rng(seed=id)
        self.lazy_store = lazy_store

        # Featurized Transformations
        self.mask: np.ndarray[int] = None  # heldout indices [0, len(watch_history))
        self.preserved: np.ndarray[int] = None  # visible indices [0, len(watch_history)

        # Masked and Preserved Id's
        self.masked_cais: np.ndarray[int] = None
        self.preserved_cais: np.ndarray[int] = None
        self.negative_cais: np.ndarray[int] = None  # negative pool for training

        # Many-Hot Encoding of features
        self.preserved_features: np.ndarray[int] = None
        self.preserved_imputed_features: np.ndarray[float] = None
        self.masked_features: np.ndarray[int] = None
        self.masked_imputed_features: np.ndarray[float] = None

    def reseed(self):
        self.rng = np.random.default_rng(seed=id)

    def reshuffle(self):
        """
        Reshuffle the holdout, selecting self. indices to mask out and updating
        corresponding feature vectors
        """
        # Randomly select indices to mask (subsample)
        permutation = self.rng.permutation(len(self.watch_history))
        self.mask = permutation[: self.k]
        self.preserved = permutation[self.k :]
        self.negative_cais = np.setdiff1d(
            np.arange(self.max_anime_count), self.watch_history, assume_unique=True
        )

        # Subsample both arrays using the selected indices
        self.masked_cais = self.watch_history[self.mask]
        self.preserved_cais = self.watch_history[self.preserved]

        # Generate feature vectors (each of size self.max_anime_count)
        if not self.lazy_store:
            self.preserved_features = self.get_preserved_features(False)
            self.preserved_imputed_features = self.get_preserved_features(True)
            self.masked_features = self.get_masked_features(False)
            self.masked_imputed_features = self.get_masked_features(True)

    def generate_masked_history(self):
        """
        Generates the masked history based on the current watch/rating history.
        Note that this should be called after the filtering process, so that
        the masked information does not include any filtered out data.

        Doesn't do anything if already shuffled
        """
        if self.mask is None:
            self.reshuffle()
        else:
            print(
                "Warning: generate_masked_history() called on shuffled data, skipping"
            )

    def sample_negative(self, n: int) -> np.ndarray[int]:
        """
        Sample n negative items from the negative pool
        """
        if n <= 0n > len(self.negative_cais):
            raise ValueError(
                f"Cannot sample {n} negative items from pool of size {len(self.negative_cais)} "
                + f"for user {self.id}"
            )
        return self.rng.choice(self.negative_cais, n, replace=False)

    def partition_preserved(self, n) -> Tuple[np.ndarray[int], np.ndarray[int]]:
        """
        Sample n positive items from within the preserved pool. Returns the postiive and negative
        split
        """
        if n <= 0 or n > len(self.preserved_cais):
            raise ValueError(
                f"Cannot sample {n} positive items from pool of size {len(self.preserved_cais)} "
                + f"for user {self.id}"
            )
        perm = self.rng.choice(self.preserved_cais, len(self.preserved_cais), replace=False)
        return perm[:n], perm[n:]


    def get_features(self, imputed=False) -> np.ndarray[float]:
        """
        Returns feature vector of the rating history of the user.
        """
        one_hot_encoding = np.zeros(self.max_anime_count, dtype=float)
        one_hot_encoding[self.watch_history] = (
            self.imputed_rating_history if imputed else self.rating_history
        )
        return one_hot_encoding

    def get_preserved_features(self, imputed=False) -> np.ndarray[float]:
        """
        Returns feature vector of visible rating history for the user
        """
        assert self.preserved_cais is not None
        one_hot_encoding = np.zeros(self.max_anime_count, dtype=float)
        one_hot_encoding[self.preserved_cais] = (
            self.imputed_rating_history if imputed else self.rating_history
        )[self.preserved]
        return one_hot_encoding

    def get_masked_features(self, imputed=False):
        """
        Returns feature vector of invisible rating history for the user
        """
        assert self.masked_cais is not None
        one_hot_encoding = np.zeros(self.max_anime_count, dtype=float)
        one_hot_encoding[self.masked_cais] = (
            self.imputed_rating_history if imputed else self.masked_rating_history
        )[self.mask]
        return one_hot_encoding
