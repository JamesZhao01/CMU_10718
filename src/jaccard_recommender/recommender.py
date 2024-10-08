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
        n=20,
        true_weithed_average=False,
        *args,
        **kwargs,
    ):
        self.evaluator = Evaluator(data_path, normalize_unrated=normalize_unrated, n=n)
        self.n_anime = len(self.evaluator.anime_mapping)
        self.n_users = len(self.evaluator.user_mapping)

        self.output_dir = output_dir
        self.true_weighted_average = true_weithed_average

        os.makedirs(output_dir, exist_ok=True)

    def build_matrix(self) -> None:
        user_id_to_canonical_id = {
            k: i for i, (k, _) in enumerate(self.evaluator.user_mapping.items())
        }
        # Sparse matrix, anime x user
        # Intersection: anime @ anime.T
        interaction_matrix = lil_matrix((self.n_anime, self.n_users), dtype=np.float32)
        for user_id in tqdm.tqdm(self.evaluator.train_indices):
            user = self.evaluator.user_mapping[user_id]
            canonical_user_id = user_id_to_canonical_id[user_id]
            interaction_matrix[user.preserved_canonical_ids, canonical_user_id] = 1
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
        ratings_vector = np.zeros((self.n_items, 1))
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
        items: list[list[int]],
        ratings: list[list[float]],
        k: int,
    ) -> np.ndarray[int]:
        n_test = len(items)
        ratings_tensor = np.zeros((self.n_items, len(items)), dtype=np.float32)
        for i, (item, rating) in enumerate(zip(items, ratings)):
            ratings_tensor[item, i] = rating
        unnormalized_scores = self.sim_matrix @ ratings_tensor
        ratings[np.arange(n_test)[:, 1],]

    def inference(self) -> None:
        print(
            f"Percentage Zeroes: {(np.sum(self.sim_matrix == 0)) / (len(self.sim_matrix)**2):0.2f}"
        )
        # Relative normalization
        row_sums = self.sim_matrix.sum(axis=1, keepdims=True)
        self.sim_matrix /= np.where(row_sums == 0, 1, row_sums)

        # Perform recommendations
        user_history, k = self.start_eval_test_set()
        k_recommended_shows = np.zeros((len(user_history), k), dtype=np.int32)
        for i, masked_history in tqdm.tqdm(
            enumerate(user_history), total=len(user_history)
        ):
            items = masked_history.nonzero()[0]
            ratings = masked_history[items]
            recommended = self.recommend(items, ratings, k)
            k_recommended_shows[i] = recommended
        self.evaluator.end_eval_test_set(k_recommended_shows)
