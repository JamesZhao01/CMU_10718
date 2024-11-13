from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import functools
from typing import Tuple
from dataclass_wizard import JSONWizard, json_field
import numpy as np


def encode_ndarray(arr: np.ndarray):
    return arr.tolist()


def decode_ndarray(arr: list):
    return np.array(arr)


def encode_set(data: set):
    return [v.name for v in data]


def decode_set(arr: list, cls: type):
    return {cls[v] for v in arr}


def make_ndarray_field():
    field(
        metadata={
            "dataclass_wizard": {"encoder": encode_ndarray, "decoder": decode_ndarray}
        }
    )


def make_optional_ndarray_field():
    field(
        metadata={
            "dataclass_wizard": {"encoder": encode_ndarray, "decoder": decode_ndarray}
        },
        default=None,
    )


def make_set_enum_field(cls: type):
    field(
        metadata={
            "dataclass_wizard": {
                "encoder": encode_set,
                "decoder": functools.partial(decode_set, cls=cls),
            }
        },
        default_factory=set,
    )


def make_enum_field(cls: type):
    field(
        metadata={
            "dataclass_wizard": {
                "encoder": lambda x: x.name,
                "decoder": lambda x: cls[x],
            }
        },
        default_factory=set,
    )


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


@dataclass
class Anime(JSONWizard):
    id: int
    name: str
    episodes: int
    rating: float
    membership_count: int
    type: Type = make_enum_field(Type)
    genres: set[Genre] = make_set_enum_field(Genre)


@dataclass
class User(JSONWizard):
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
    To allow for dynamically sampling "positives" within the "preserved" ids, utility for paritioning
        is defined in `partition_preserved()`
    """

    id: int
    cuid: int
    k: int
    max_anime_count: int
    masked_is_negative: bool
    watch_history: np.ndarray[int] = make_ndarray_field()
    rating_history: np.ndarray[float] = make_ndarray_field()
    imputed_history: np.ndarray[float] = make_ndarray_field()
    rng: np.random.Generator | None = field(
        default=None, metadata={"dataclass_wizard": {"exclude": True}}
    )
    lazy_store: bool = False
    

    mask: np.ndarray[int] | None = make_optional_ndarray_field()
    preserved: np.ndarray[int] | None = make_optional_ndarray_field()

    masked_cais: np.ndarray[int] | None = make_optional_ndarray_field()
    preserved_cais: np.ndarray[int] | None = make_optional_ndarray_field()
    negative_cais: np.ndarray[int] | None = make_optional_ndarray_field()

    preserved_features: np.ndarray[int] | None = make_optional_ndarray_field()
    preserved_imputed_features: np.ndarray[int] | None = make_optional_ndarray_field()
    masked_features: np.ndarray[int] | None = make_optional_ndarray_field()
    masked_imputed_features: np.ndarray[int] | None = make_optional_ndarray_field()

    def __post_init__(self):
        self.rng = np.random.default_rng(seed=self.id)
        if self.mask is None:
            self.generate_masked_history()
            self.reseed()

    def reseed(self):
        self.rng = np.random.default_rng(seed=self.id)

    def reshuffle(self):
        """
        Reshuffle the holdout, selecting self. indices to mask out and updating
        corresponding feature vectors
        """
        # Randomly select indices to mask (subsample)
        permutation = self.rng.permutation(len(self.watch_history))
        self.mask = permutation[: self.k]
        self.preserved = permutation[self.k :]

        # Subsample both arrays using the selected indices
        self.masked_cais = self.watch_history[self.mask]
        self.preserved_cais = self.watch_history[self.preserved]
        if self.masked_is_negative:
            self.negative_cais = np.setdiff1d(
                np.arange(self.max_anime_count), self.preserved_cais
            )
        else:
            self.negative_cais = np.setdiff1d(
                np.arange(self.max_anime_count), self.watch_history
            )

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
        if n <= 0 or n > len(self.negative_cais):
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
        perm = self.rng.choice(
            self.preserved_cais, len(self.preserved_cais), replace=False
        )
        return perm[:n], perm[n:]

    def get_features(self, imputed=False) -> np.ndarray[float]:
        """
        Returns feature vector of the entire history of the user.
        """
        one_hot_encoding = np.zeros(self.max_anime_count, dtype=float)
        one_hot_encoding[self.watch_history] = (
            self.imputed_history if imputed else self.rating_history
        )
        return one_hot_encoding

    def get_preserved_features(self, imputed=False) -> np.ndarray[float]:
        """
        Returns feature vector of visible rating history for the user
        """
        assert self.preserved_cais is not None
        one_hot_encoding = np.zeros(self.max_anime_count, dtype=float)
        one_hot_encoding[self.preserved_cais] = (
            self.imputed_history if imputed else self.rating_history
        )[self.preserved]
        return one_hot_encoding

    def get_masked_features(self, imputed=False):
        """
        Returns feature vector of invisible rating history for the user
        """
        assert self.masked_cais is not None
        one_hot_encoding = np.zeros(self.max_anime_count, dtype=float)
        one_hot_encoding[self.masked_cais] = (
            self.imputed_history if imputed else self.rating_history
        )[self.mask]
        return one_hot_encoding
