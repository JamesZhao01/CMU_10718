from abc import ABC
import numpy as np

from utils.data.data_module import DataModule
from utils.data.testbench import TestBench


class GenericRecommender(ABC):
    def __init__(self, datamodule: DataModule):
        super().__init__()
        self.datamodule = datamodule

    def train(self):
        raise NotImplementedError("train() not implemented")

    def infer(self, ratings_features: np.ndarray[float], k: int) -> np.ndarray[int]:
        """Takes in a ratings float tensor of size (n_test, n_anime) and returns an int tensor of
            size (n_test, k), the canonical anime ids to rank

        Args:
            k: number of items to recommend
            ratings_features: tensor of dimension (n_test, n_anime)

        Returns:
            np.ndarray[int]: tensor of size (n_test, n_anime)
        """
        raise NotImplementedError("infer() not implemented")
