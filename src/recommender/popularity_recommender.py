from recommender.generic_recommender import GenericRecommender
import numpy as np


class PopularityRecommender(GenericRecommender):
    def __init__(self, datamodule, popularity_mode="membership_count", **kwargs):
        super().__init__(datamodule)
        self.popularity_mode = popularity_mode

    def train(self):
        popularity = [
            (
                caid,
                getattr(anime, self.popularity_mode),
            )
            for caid, anime in self.datamodule.canonical_anime_mapping.items()
        ]
        popularity.sort(key=lambda x: -x[1])
        ranked_cuids = [x[0] for x in popularity]
        self.ranked_items = np.array(ranked_cuids)

    def infer(self, ratings_features: np.ndarray[float], k: int) -> np.ndarray[int]:
        n_test, n_anime = ratings_features.shape
        predictions = []
        for i in range(n_test):
            history = ratings_features[i].nonzero()[0]
            chooseable_mask = np.isin(self.ranked_items, history, invert=True)
            predictions.append(self.ranked_items[chooseable_mask][:k])
        return None, np.vstack(predictions)
