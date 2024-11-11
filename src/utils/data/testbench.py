"""
TestBench

Test Bench contains all the configuration to evaluates the model. All test benchmarks will be 
reliant on using imputed scores for the results. We can have several potential benchmarks

# Interaction Holdout:

This holdout method describes a typical recommender context, in which we have interaction histor
of a user, and want to begin recommending items to them. 

In training and test datasets, we have the same users for both datasets. The test dataset
will withold k=10 interactions from each user. The models will be evaluated on how well they rank
the heldout items.

# User Holdout:

This holdout method describes a context in which we have the interaction history of a user, but
not a personalized embedding (in the form of a lookup table). Training and test datasets will have 
different user splits (80/10/10).

For Matrix Factorization models, user embeddings will need to be re-trained during test time
For Two-Tower models, models will need to made agnostic to user embeddings
"""

import time
from datetime import datetime
import tqdm

import numpy as np
from sklearn.metrics import ndcg_score

from recommender.generic_recommender import GenericRecommender
from utils.data.data_module import DataModule


class TestBench:
    def __init__(self, data_module: DataModule):
        self.data_module = data_module

        self.k = data_module.k
        self.n_test = len(self.data_module.test_cuids)
        self.n_anime = self.data_module.max_anime_count

    def batch_iterator(self, data, batch_size: int):
        n = len(data)
        for i in range(0, n, batch_size):
            yield data[i : min(i + batch_size, n)]

    def get_preserved_feature_tensor(self) -> np.ndarray[float]:
        """_summary_

        Returns:
            np.ndarray[float]: _description_
        """
        preserved_histories = [
            self.data_module.canonical_user_mapping[cuid].get_preserved_features()
            for cuid in tqdm.tqdm(
                self.data_module.test_cuids, desc="Preserved Features"
            )
        ]
        arr = np.vstack(preserved_histories)
        assert arr.shape == (self.n_test, self.n_anime)
        return arr

    def get_masked_feature_tensor(self) -> np.ndarray[float]:
        """_summary_

        Returns:
            np.ndarray[float]: _description_
        """
        masked_features = [
            self.data_module.canonical_user_mapping[cuid].get_masked_features(
                imputed=True
            )
            for cuid in self.data_module.test_cuids
        ]
        arr = np.vstack(masked_features)
        assert arr.shape == (self.n_test, self.n_anime)
        return arr

    def get_masked_top_k_tensor(self) -> np.ndarray[int]:
        k_predictions = np.zeros((self.n_test, self.k), dtype=int)
        for cuid in self.data_module.test_cuids:
            user = self.data_module.canonical_user_mapping[id]
            ground_truth_of_masked_history = user.get_masked_features(imputed=True)
            ids = np.nonzero(ground_truth_of_masked_history)[0]
            ratings = ground_truth_of_masked_history[ids]

            sorted_indices = np.argsort(-ratings)
            sorted_ids = ids[sorted_indices]
            assert len(sorted_ids) == self.k
            k_predictions[cuid] = sorted_ids
        return k_predictions

    def start_eval_test_set(self):
        """
        Returns:
        - user_preserved_watch_history: Masked user watch history of shape [num_of_users, MAX_ANIME_COUNT]
        - k: number of anime recommendations your model should present
        """

        user_preserved_watch_history = self.get_preserved_feature_tensor()
        self.start_time = time.time()
        str_time = datetime.fromtimestamp(self.start_time).strftime("%Y-%m-%d %H:%M:%S")
        print(f"Start Time: {str_time}")
        return user_preserved_watch_history, self.k

    def calculate_ndcg(self, k_recommended_shows: np.ndarray[int]):
        masked_feature_ground_truth_scores = self.get_masked_feature_tensor()
        pred = np.zeros((self.n_test, self.n_anime), dtype=int)
        # Sort the indices and generate decreasing values
        decreasing_values = np.arange(k_recommended_shows.shape[1], 0, -1)

        # Assign the decreasing values to the respective indices
        pred[np.arange(self.n_test)[:, None], k_recommended_shows] = decreasing_values

        score = ndcg_score(masked_feature_ground_truth_scores, pred, k=self.k)
        return score

    def end_eval_test_set(self, k_recommended_shows: np.ndarray[int]):
        """
        Run this method to end the evaluation program.
        I expect k_recommended_shows to be of shape [test_set_size, k].
        The columns should be ordered by ranking with 0 being the highest and 1 being
        the lowest.

        if is_anime_id is true, then each row of k_recommended_shows should be a list of anime ids,
            in ranked order
        """
        end_time = time.time()
        str_time = datetime.fromtimestamp(end_time).strftime("%Y-%m-%d %H:%M:%S")
        print(f"End Time: {str_time}")
        total_runtime = time.time() - self.start_time
        score = self.calculate_ndcg(k_recommended_shows)

        print(f"This model took {total_runtime:0.4f} seconds.")
        print(f"Out of an optimal score of 1.0, you scored {score:0.4f}.")
        return total_runtime, score

    def full_evaluation(self, recommender: GenericRecommender):
        preserved_features, k = self.start_eval_test_set()
        k_recommended_shows = recommender.infer(preserved_features, k)
        assert k_recommended_shows.shape == (self.n_test, k)
        total_runtime, score = self.end_eval_test_set(k_recommended_shows)

        return {
            "total_runtime": total_runtime,
            "score": score,
            "k": k,
            "k_recommended_shows": k_recommended_shows,
            "preserved_features": preserved_features,
        }
