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


class TestBench:
    def get_anime_info(self, id: int):
        return self.anime_mapping[id]

    def get_ground_truth_ratings_tensor(self) -> np.ndarray[float]:
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
        return ground_truth_of_masked_history

    def get_ground_truth_k_predictions(self) -> np.ndarray[int]:
        k_predictions = np.zeros((len(self.test_ids), self.k), dtype=int)
        for i, id in enumerate(self.test_ids):
            user: User = self.user_mapping[id]
            complete_watch_history = user.get_imputed_history()
            user_masked_watch_history = user.get_imputed_masked_history()
            ground_truth_of_masked_history = (
                complete_watch_history - user_masked_watch_history
            )
            ids = np.nonzero(ground_truth_of_masked_history)[0]

            ratings = ground_truth_of_masked_history[ids]
            sorted_indices = np.argsort(-ratings)
            sorted_ids = ids[sorted_indices]
            assert len(sorted_ids) == self.k
            k_predictions[i] = sorted_ids
        return k_predictions

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

    def calculate_ndcg(self, k_recommended_shows: np.ndarray[int]):
        ground_truth_of_masked_history = self.get_ground_truth_ratings_tensor()
        pred = np.zeros((len(self.test_ids), self.max_anime_count), dtype=int)
        # Sort the indices and generate decreasing values
        decreasing_values = np.arange(k_recommended_shows.shape[1], 0, -1)

        # Assign the decreasing values to the respective indices
        pred[np.arange(len(self.test_ids))[:, None], k_recommended_shows] = (
            decreasing_values
        )

        score = ndcg_score(ground_truth_of_masked_history, pred, k=self.k)
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
        total_runtime = time.time() - self.start_time
        score = self.calculate_ndcg(k_recommended_shows)

        print(f"This model took {total_runtime:0.4f} seconds.")
        print(f"Out of an optimal score of 1.0, you scored {score:0.4f}.")
        return total_runtime, score
