import os
from collections import defaultdict

import numpy as np
import tqdm as tqdm
from scipy.sparse import csr_matrix, lil_matrix

from utils.evaluator import Evaluator


class JaccardRecommender:
    def __init__(
        self,
        data_path: str,
        output_dir: str,
        normalize_unrated=False,
        thresholded_watch_history=20,
        true_weighted_average=False,
        sim_matrix_normalization="row", # normalization method
        sim_threshold=0.0, # lower bound for similarity to be considered
        normalizing_constant_clip=0, # lower bound for normalization constant
        normalizing_constant_regularization = 0, # constant added to normalizing constants
        score_regularization = 0, # constant added to scores
        save=True,
        evaluator = None,
        *args,
        **kwargs,
    ):
        print(
            f"{normalize_unrated=} {thresholded_watch_history=} {true_weighted_average=} {sim_matrix_normalization=}"
        )
        if not evaluator:
            self.evaluator = Evaluator(
                data_path,
                normalize_unrated=normalize_unrated,
                threshold_watch_history=thresholded_watch_history,
            )
        else:
            self.evaluator = evaluator
        self.n_anime = len(self.evaluator.anime_mapping)
        self.n_users = len(self.evaluator.user_mapping)

        self.output_dir = output_dir
        assert sim_matrix_normalization in ["row", "global", "none"]
        self.sim_matrix_normalization = sim_matrix_normalization
        self.true_weighted_average = true_weighted_average
        self.save = save
        
        self.sim_threshold = sim_threshold
        self.normalizing_constant_clip = normalizing_constant_clip
        self.normalizing_constant_regularization = normalizing_constant_regularization
        self.score_regularization = score_regularization

        os.makedirs(output_dir, exist_ok=True)

    def build_matrix(self) -> None:
        user_id_to_canonical_id = {
            k: i for i, (k, _) in enumerate(self.evaluator.user_mapping.items())
        }
        # Sparse matrix, anime x user
        # Intersection: anime @ anime.T
        interaction_matrix = lil_matrix((self.n_anime, self.n_users), dtype=np.float32)
        for user_id in tqdm.tqdm(
            self.evaluator.train_indices, desc="building interaction matrix..."
        ):
            user = self.evaluator.user_mapping[user_id]
            canonical_user_id = user_id_to_canonical_id[user_id]
            interaction_matrix[user.masked_watch_history, canonical_user_id] = 1
        interaction_matrix = interaction_matrix.tocsr()
        intersection = (interaction_matrix @ interaction_matrix.T).toarray()
        sum_counts = np.array(interaction_matrix.sum(axis=1))
        union = (sum_counts.reshape(-1, 1) + sum_counts.reshape(1, -1)) - intersection
        self.sim_matrix = intersection / np.where(union == 0, 1, union)
        if self.save:
            np.save(
                os.path.join(self.output_dir, "item_similarity_matrix.npy"),
                self.sim_matrix,
            )

    def load_matrix(self) -> None:
        self.sim_matrix = os.path.join(self.output_dir, "item_similarity_matrix.npy")

    def bulk_recommend(
        self,
        ratings_tensor: np.ndarray[float],  # (n, t)
        k: int,
    ) -> np.ndarray[int]:
        ratings_mask = (~np.isclose(ratings_tensor, 0)).astype(np.float32)  # (n, t)
        scores = self.sim_matrix @ ratings_tensor  # (n, n) x (n, t) -> (n, t)
        assert ratings_mask.shape == scores.shape == ratings_tensor.shape
        if self.true_weighted_average:
            # (n, n) x (n, t) -> (n, t)
            normalizing_constants = self.sim_matrix @ ratings_mask + self.normalizing_constant_regularization
            normalizing_constants = normalizing_constants.clip(min=self.normalizing_constant_clip)
            scores = (scores + self.score_regularization) / normalizing_constants
        scores = np.where(ratings_mask, -float("inf"), scores)
        order = np.argsort(-scores, axis=0)
        results = order[:k].T
        self.scores = scores
        return results

    def inference(self) -> None:
        print(
            f"Percentage Zeroes: {(np.sum(self.sim_matrix == 0)) / (len(self.sim_matrix)**2):0.2f}"
        )

        if self.sim_matrix_normalization == "row":
            row_sums = self.sim_matrix.sum(axis=1, keepdims=True)
            self.sim_matrix /= np.where(row_sums == 0, 1, row_sums)
        elif self.sim_matrix_normalization == "global":
            self.sim_matrix /= np.sum(self.sim_matrix)
        else:
            pass
        # apply similarity thresholding
        self.sim_matrix = np.where(self.sim_matrix < self.sim_threshold, 0, self.sim_matrix)

        # Perform recommendations
        ratings_tensor, k = self.evaluator.start_eval_test_set()
        self.rating_tensor = ratings_tensor
        self.k_recommended_shows = self.bulk_recommend(ratings_tensor.T, k)
        return self.evaluator.end_eval_test_set(self.k_recommended_shows)
