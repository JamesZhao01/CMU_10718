import numpy as np
import tqdm
from recommender.generic_recommender import GenericRecommender
from utils.data.data_module import DataModule
from scipy.sparse import lil_matrix


class JaccardRecommender(GenericRecommender):
    def __init__(
        self,
        datamodule: DataModule,
        sim_matrix_normalization="row",  # first step processing
        sim_threshold: float = 0.0,  # any post-normalized score below threshold is zeroed
        do_weighted_average=False,  # whether weighted averaging
        normalizing_clip: float = 0,  # lower bound for normalization constant
        normalizing_reg: float = 0,  # constant added to normalizing constant
        score_reg: float = 0,  # constant added to scores
        **kwargs,
    ):
        super().__init__(datamodule)

        self.n_anime = len(self.datamodule.canonical_anime_mapping)
        self.n_users = len(self.datamodule.canonical_user_mapping)
        assert sim_matrix_normalization in ["row", "global", "none"]
        self.sim_matrix_normalization = sim_matrix_normalization
        self.sim_threshold = sim_threshold
        self.do_weighted_average = do_weighted_average
        self.normalizing_clip = normalizing_clip
        self.normalizing_reg = normalizing_reg
        self.score_reg = score_reg

    def train(self, *args, **kwargs):
        # Sparse matrix, anime x user
        # Intersection: anime @ anime.T
        interaction_matrix = lil_matrix((self.n_anime, self.n_users), dtype=np.float32)
        for cuid in tqdm.tqdm(
            self.datamodule.cuids, desc="Building interaction matrix..."
        ):
            user = self.datamodule.canonical_user_mapping[cuid]
            interaction_matrix[user.preserved_cais, cuid] = 1
        interaction_matrix = interaction_matrix.tocsr()
        intersection = (interaction_matrix @ interaction_matrix.T).toarray()
        sum_counts = np.array(interaction_matrix.sum(axis=1))
        union = (sum_counts.reshape(-1, 1) + sum_counts.reshape(1, -1)) - intersection
        self.sim_matrix = intersection / np.where(union == 0, 1, union)

        if self.sim_matrix_normalization == "row":
            row_sums = self.sim_matrix.sum(axis=1, keepdims=True)
            self.sim_matrix /= np.where(row_sums == 0, 1, row_sums)
        elif self.sim_matrix_normalization == "global":
            self.sim_matrix /= np.sum(self.sim_matrix)
        elif self.sim_matrix_normalization == "none":
            pass
        else:
            raise NotImplementedError(
                f"{self.sim_matrix_normalization} not implemented"
            )
        self.sim_matrix = np.where(
            self.sim_matrix < self.sim_threshold, 0, self.sim_matrix
        )

    def infer(self, ratings_features: np.ndarray[float], k: int) -> np.ndarray[int]:
        ratings_mask = (~np.isclose(ratings_features, 0).T).astype(np.float32)
        scores = self.sim_matrix @ ratings_features.T

        if self.do_weighted_average:
            normalizing_constants = (
                self.sim_matrix @ ratings_mask + self.normalizing_reg
            )
            normalizing_constants = normalizing_constants.clip(
                min=self.normalizing_clip
            )
            scores = (scores + self.score_reg) / normalizing_constants
        scores = np.where(ratings_mask, -float("inf"), scores)
        order = np.argsort(-scores, axis=0)
        results = order[:k].T  # (k, t) -> (t, k)
        return scores, results
