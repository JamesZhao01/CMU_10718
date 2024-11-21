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

from functools import cache
import math
import time
from datetime import datetime
import tqdm

import numpy as np
from sklearn.metrics import ndcg_score

from recommender.generic_recommender import GenericRecommender
from utils.data.data_classes import Anime
from utils.data.data_module import DataModule


class TestBench:
    def __init__(
        self,
        datamodule: DataModule,
        should_return_ids: bool = False,
        calibration_buckets=10,
        sigmoid_scores=False,
        **kwargs,
    ):
        self.datamodule = datamodule
        self.should_return_ids = should_return_ids

        self.k = self.datamodule.k
        self.calibration_buckets = calibration_buckets
        self.sigmoid_scores = sigmoid_scores
        self.n_test = len(self.datamodule.test_cuids)
        self.n_anime = self.datamodule.max_anime_count

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
            self.datamodule.canonical_user_mapping[cuid].get_preserved_features()
            for cuid in tqdm.tqdm(self.datamodule.test_cuids, desc="Preserved Features")
        ]
        arr = np.vstack(preserved_histories)
        assert arr.shape == (self.n_test, self.n_anime)
        return arr

    @cache
    def get_masked_feature_tensor(self) -> np.ndarray[float]:
        """_summary_

        Returns:
            np.ndarray[float]: _description_
        """
        masked_features = [
            self.datamodule.canonical_user_mapping[cuid].get_masked_features(
                imputed=True
            )
            for cuid in self.datamodule.test_cuids
        ]
        arr = np.vstack(masked_features)
        assert arr.shape == (self.n_test, self.n_anime)
        return arr

    def get_masked_top_k_tensor(self) -> np.ndarray[int]:
        k_predictions = np.zeros((self.n_test, self.k), dtype=int)
        for cuid in self.datamodule.test_cuids:
            user = self.datamodule.canonical_user_mapping[id]
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

        Can also return (list[User], [(list[Anime], ratings_ndarray)..]) if should_return_ids is True
        """

        if not self.should_return_ids:
            user_preserved_watch_history = self.get_preserved_feature_tensor()
            to_return = user_preserved_watch_history
        else:
            users = [
                self.datamodule.canonical_user_mapping[test_cuid]
                for test_cuid in self.datamodule.test_cuids
            ]
            user_histories = [
                (
                    [
                        self.datamodule.canonical_anime_mapping[caid]
                        for caid in self.datamodule.canonical_user_mapping[
                            test_cuid
                        ].preserved_cais
                    ],
                    self.datamodule.canonical_user_mapping[test_cuid].rating_history[
                        self.datamodule.canonical_user_mapping[test_cuid].preserved
                    ],
                )
                for test_cuid in self.datamodule.test_cuids
            ]
            to_return = users, user_histories
        self.start_time = time.time()
        str_time = datetime.fromtimestamp(self.start_time).strftime("%Y-%m-%d %H:%M:%S")
        print(f"Start Time: {str_time}")
        return to_return, self.k

    def calculate_ndcg(self, k_recommended_shows: np.ndarray[int]):
        masked_feature_ground_truth_scores = self.get_masked_feature_tensor()
        pred = np.zeros((self.n_test, self.n_anime), dtype=int)
        # Sort the indices and generate decreasing values
        decreasing_values = np.arange(k_recommended_shows.shape[1], 0, -1)

        # Assign the decreasing values to the respective indices
        pred[np.arange(self.n_test)[:, None], k_recommended_shows] = decreasing_values

        score = ndcg_score(masked_feature_ground_truth_scores, pred, k=self.k)
        return score

    def calculate_r_precision(
        self, k_recommended_shows: np.ndarray[int], k_truth: np.ndarray[int]
    ):
        """
        What percentage of truth was recommended
        """
        pass

    def calculate_diversity_by_community_count(
        self, k_recommended_shows: np.ndarray[int]
    ):
        """
        Gets the membership/community count of each anime that was recommended and sums
        them up, ignoring duplicates.

        This prioritizes both popularity and diversity, as popular shows will increase the count,
        but duplicates will not.

        TODO: Consider functions to downgrade the importance of popularity (log, sqrt)
        """

        total_community_count = 0
        recommended = set()
        for row_of_watch_history in k_recommended_shows:
            for caid in row_of_watch_history:
                anime: Anime = self.datamodule.canonical_anime_mapping[caid]
                if caid not in recommended:
                    total_community_count += math.log(anime.membership_count)
                    recommended.add(caid)
        return total_community_count

    def pseudo_iou(self, k_recommended_shows: np.ndarray[int]):
        """
        Gets the membership/community count of each anime that was recommended and sums
        them up, ignoring duplicates.

        This prioritizes both popularity and diversity, as popular shows will increase the count,
        but duplicates will not.

        TODO: Consider functions to downgrade the importance of popularity (log, sqrt)
        """
        recommended = set()
        total_posssible = k_recommended_shows.shape[0] * k_recommended_shows.shape[1]
        for row_of_watch_history in k_recommended_shows:
            for id in row_of_watch_history:
                if id not in recommended:
                    recommended.add(id)
        ##count elements in recommended
        total_num = len(recommended)
        score = total_num / total_posssible
        return score

    def binary_calibration(self, scores: np.ndarray[int], buckets=10):
        """
        Calibration score (ECE) for binary classification
        Within each bucket, find the actual success rate and compare it with the predicted probabilities
        """
        masked_feature_matrix = self.get_masked_feature_tensor().flatten()
        masked_interaction_matrix = ~np.isclose(masked_feature_matrix, 0)
        n = len(masked_feature_matrix)
        ct_positive = np.sum(masked_interaction_matrix)

        predicted_probabilities = np.array(scores.flatten())
        assert len(predicted_probabilities) == n == len(masked_interaction_matrix)
        if self.sigmoid_scores:
            predicted_probabilities = 1 / (1 + np.exp(-predicted_probabilities))

        # print(f"Binary Calibration: {n=} {ct_positive=}")
        # print(np.min(predicted_probabilities), np.max(predicted_probabilities))

        bucket_idx = np.floor(predicted_probabilities * buckets).astype(np.int32)
        bucket_frequencies = np.zeros((buckets,))
        bucket_bincounts = np.bincount(bucket_idx)
        for i in range(len(bucket_bincounts)):
            bucket_frequencies[i] = bucket_bincounts[i]
        bucket_n_success = np.array(
            [sum(masked_interaction_matrix[bucket_idx == i]) for i in range(buckets)]
        )
        bucket_accuracies = np.where(
            bucket_frequencies == 0, 0, bucket_n_success / bucket_frequencies
        )
        bucket_average_probabilities = np.array(
            [
                (
                    np.mean(predicted_probabilities[bucket_idx == i])
                    if bucket_frequencies[i] > 0
                    else 0
                )
                for i in range(buckets)
            ]
        )
        assert np.sum(bucket_frequencies) == n
        assert np.sum(bucket_n_success) == ct_positive
        ece = (
            np.sum(
                bucket_frequencies
                * np.abs(bucket_accuracies - bucket_average_probabilities)
            )
            / n
        )
        return {
            "ece": ece,
            "bucket_frequencies": bucket_frequencies,
            "bucket_accuracies": bucket_accuracies,
            "bucket_average_probabilities": bucket_average_probabilities,
        }

    def end_eval_test_set(
        self, scores: np.ndarray[int], k_recommended_shows: np.ndarray[int]
    ):
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
        ndcg_score = self.calculate_ndcg(k_recommended_shows)
        diversity_score = self.calculate_diversity_by_community_count(
            k_recommended_shows
        )
        pseudo_iou = self.pseudo_iou(k_recommended_shows)
        calibration_scores = self.binary_calibration(scores, self.calibration_buckets)

        print(f"This model took {total_runtime:0.4f} seconds.")
        print(f"Out of an optimal score of 1.0, you scored {ndcg_score:0.4f}.")
        print(f"Your DEI score is {diversity_score:0.4f}.")
        print(f"Your Pseudo-IOU score is {pseudo_iou:0.4f}.")
        print(f"Your calibration score is {calibration_scores['ece']:0.4f}.")
        calibration_breakdown = list(
            zip(
                calibration_scores["bucket_frequencies"],
                calibration_scores["bucket_accuracies"],
                calibration_scores["bucket_average_probabilities"],
            )
        )
        print(f"Your calibration breakdown: ")
        title_a, title_b, title_c = (
            "Bucket Frequency",
            "Bucket Accuracy",
            "Bucket Average Probability",
        )
        print(f"{title_a:<20s}|{title_b:<20s}|{title_c:<20s}")
        for a, b, c in calibration_breakdown:
            print(f"{a:^20.2f}|{b:^20.4f}|{c:^20.4f}")
        return {
            "runtime": total_runtime,
            "ndcg": ndcg_score,
            "diversity_score": diversity_score,
            "pseudo_iou": pseudo_iou,
        } | calibration_scores

    def full_evaluation(self, recommender: GenericRecommender, return_scores=False):
        preserved_features, k = self.start_eval_test_set()
        scores, k_recommended_shows = recommender.infer(preserved_features, k)
        assert k_recommended_shows.shape == (self.n_test, k)
        results = self.end_eval_test_set(scores, k_recommended_shows)
        auxiliary_results = (
            {
                "k_recommended_shows": k_recommended_shows,
                "preserved_features": preserved_features,
                "scores": scores,
            }
            if return_scores
            else {}
        )
        return results | auxiliary_results
