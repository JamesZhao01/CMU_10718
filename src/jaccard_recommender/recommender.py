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
        *args,
        **kwargs,
    ):
        print(
            f"{normalize_unrated=} {thresholded_watch_history=} {true_weighted_average=}"
        )
        self.evaluator = Evaluator(
            data_path,
            normalize_unrated=normalize_unrated,
            threshold_watch_history=thresholded_watch_history,
        )
        self.n_anime = len(self.evaluator.anime_mapping)
        self.n_users = len(self.evaluator.user_mapping)

        self.output_dir = output_dir
        self.true_weighted_average = true_weighted_average

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
        np.save(
            os.path.join(self.output_dir, "item_similarity_matrix.npy"), self.sim_matrix
        )

    def load_matrix(self) -> None:
        self.sim_matrix = os.path.join(self.output_dir, "item_similarity_matrix.npy")

    def recommend(
        self,
        items: list[int],
        ratings: list[float],
        k: int,
    ) -> np.ndarray[int]:
        ratings_vector = np.zeros((self.n_anime, 1))
        ratings_vector[items, 0] = ratings
        if self.true_weighted_average:
            sim_matrix = np.zeros_like(self.sim_matrix)
            sim_matrix[:, items] = self.sim_matrix[:, items]
            row_sum = np.sum(sim_matrix, axis=1, keepdims=True)
            sim_matrx /= np.where(row_sum == 0, 1, row_sum)
        else:
            sim_matrix = self.sim_matrix
        scores = (sim_matrix @ ratings_vector).flatten()
        scores[items] = -1  # mask out items rated by user
        order = np.argsort(-scores).flatten()
        return order[:k]

    def bulk_recommend(
        self,
        ratings_tensor: np.ndarray[float],  # (n, t)
        k: int,
    ) -> np.ndarray[int]:
        ratings_mask = (~np.isclose(ratings_tensor, 0)).astype(np.float32)  # (n, t)
        scores = self.sim_matrix @ ratings_tensor  # (n, n) x (n, t) -> (n, t)
        assert ratings_mask.shape == scores.shape == ratings_tensor.shape
        print(f"{scores.dtype}")
        if self.true_weighted_average:
            normalizing_constants = (
                self.sim_matrix @ ratings_mask
            )  # (n, n) x (n, t) -> (n, t)
            normalizing_constants = np.where(
                normalizing_constants == 0, 1, normalizing_constants
            )
            assert scores.shape == normalizing_constants.shape
            print(
                f"Pre Normalizatoin: {len(scores)=} {np.mean(scores)=} {np.std(scores)=}"
            )
            scores = scores / normalizing_constants
            print(
                f"Post Normalization: {len(scores)=} {np.mean(scores)=} {np.std(scores)=}"
            )
        scores = np.where(ratings_mask, -float("inf"), scores)
        order = np.argsort(-scores, axis=0)
        results = order[:k].T
        return results

    def inference(self) -> None:
        print(
            f"Percentage Zeroes: {(np.sum(self.sim_matrix == 0)) / (len(self.sim_matrix)**2):0.2f}"
        )
        # Relative normalization
        row_sums = self.sim_matrix.sum(axis=1, keepdims=True)
        self.sim_matrix /= np.where(row_sums == 0, 1, row_sums)
        print(np.sum(np.isclose(np.sum(self.sim_matrix, axis=1), 1)))
        print(np.sum(np.isclose(np.sum(self.sim_matrix, axis=1), 0)))
        print(len(self.sim_matrix))

        # Perform recommendations
        ratings_tensor, k = self.evaluator.start_eval_test_set()
        k_recommended_shows = self.bulk_recommend(ratings_tensor.T, k)
        print(k_recommended_shows.shape)
        return self.evaluator.end_eval_test_set(k_recommended_shows)
