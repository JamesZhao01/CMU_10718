import os
import pandas as pd
from enum import Enum
import random
import numpy as np
import time
from sklearn.metrics import ndcg_score
import math

MAX_ANIME_COUNT = 12_294

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

class Anime:
    def __init__(self, id:int, name:str, genres:set[Genre], type:Type, episodes:int, rating:float, membership_count:int):
        self.id:int = id
        self.name:str = name
        self.genres:set[Genre] = genres
        self.type:Type = type
        self.episodes:int = episodes
        self.rating:float = rating
        self.membership_count:int = membership_count

class User:
    def __init__(self, id:int, watch_history:np.ndarray[int], rating_history:np.ndarray[float], k):
        self.id:int = id
        self.watch_history:np.ndarray[int] = watch_history
        # Usually, ratings are ints, but if we normalize the unrated shows to the average
        # they will be a float.
        # self.rating_history:list[float] = rating_history
        self.rating_history:np.ndarray[float] = rating_history
        self.k = k
        self.masked_watch_history:np.ndarray[int] = np.array([])
        self.masked_rating_history:np.ndarray[float] = np.array([])

    def generate_masked_history(self):
        """
        Generates the masked history based on the current watch/rating history.
        Note that this should be called after the filtering process, so that
        the masked information does not include any filtered out data.
        """
        
        # Randomly select indices to mask (subsample)
        mask = np.random.choice(len(self.watch_history), size=self.k, replace=False)
        preserved_data = np.setdiff1d(np.arange(len(self.watch_history)), mask)

        # Subsample both arrays using the selected indices
        self.masked_watch_history = self.watch_history[preserved_data]
        self.masked_rating_history = self.rating_history[preserved_data]

    def get_history(self):
        """
        Returns a one-hot encoding of the rating history of the user.
        The index is the anime id. 
        (Note that index 0 is always just 0 since anime id starts with 1).
        """

        one_hot_encoding = np.zeros(MAX_ANIME_COUNT, dtype=float)
        # Populate the array with ratings at indices corresponding to anime_id
        one_hot_encoding[self.watch_history] = self.rating_history
        return one_hot_encoding
    
    def get_masked_history(self):
        """
        Same as get_history, but some of the information is masked out.
        This assumes that generate_masked_history has been called earlier.
        """

        one_hot_encoding = np.zeros(MAX_ANIME_COUNT, dtype=float)
        # Populate the one-hot list with ratings at indices corresponding to anime_id
        one_hot_encoding[self.masked_watch_history] = self.masked_rating_history
        return one_hot_encoding
        

class Evaluator:
    def __init__(self, path: str, normalize_unrated = True):
        self.animes = pd.read_csv(os.path.join(path, 'anime.csv'))
        self.users = pd.read_csv(os.path.join(path, 'rating.csv'))
        self.k = 10
        self.threshold_watch_history = 20
        self.user_mapping = {}
        self.anime_mapping = {}
        self.user_ids = []
        self.current_idx = 0
        self.normalize_unrated = normalize_unrated
        np.random.seed(42)

        # Gets all the information about the animes
        for anime_id, anime_df in self.animes.iterrows():

            genres = []
            if not pd.isna(anime_df['genre']):
                for unprocessed_string in anime_df['genre'].split(','):
                    stripped = unprocessed_string.strip()
                    if stripped == 'Sci-Fi':
                        genre = 'Sci_Fi'
                    elif stripped == 'Super Power':
                        genre = 'Super_Power'
                    elif stripped == 'Slice of Life':
                        genre = 'Slice_of_Life'
                    elif stripped == 'Shounen Ai':
                        genre = 'Shounen_Ai'
                    elif stripped == 'Shoujo Ai':
                        genre = 'Shoujo_Ai'
                    elif stripped == 'Martial Arts':
                        genre = 'Martial_Arts'
                    else:
                        genre = stripped
                    genres.append(getattr(Genre,genre))

            if pd.isna(anime_df['type']):
                type = 'Unknown'
            else:
                type = anime_df['type']
            type = getattr(Type, type)

            if pd.isna(anime_df['rating']):
                rating = 6.57 #median value
            else:
                rating = anime_df['rating']

            if pd.isna(anime_df['members']):
                membership_count = -1
            else:
                membership_count = anime_df['members']

            anime = Anime(id = anime_df['anime_id']-1,
                  name = anime_df['name'],
                  genres = genres,
                  type = type,
                  episodes = anime_df['episodes'],
                  rating = rating,
                  membership_count = membership_count
                  )
            self.anime_mapping[anime_id-1] = anime

        # Gets all the user watch history
        for user_id, user_df in self.users.groupby('user_id'):
            anime_list:np.ndarray[int] = np.array(user_df['anime_id'].tolist(), dtype=int)
            anime_list = anime_list - 1
            rating_list:np.ndarray[float] = np.array(user_df['rating'].tolist(), dtype=float)

            assert len(anime_list) == len(rating_list)

            # Filters out ratings that aren't valid with our anime_list
            valid_indices = np.where(anime_list < MAX_ANIME_COUNT)[0]
            anime_list:np.ndarray[int] = anime_list[valid_indices]
            rating_list:np.ndarray[float] = rating_list[valid_indices]

            user = User(user_id, anime_list, rating_list, self.k)

            # Filters out users that have too little watch history.
            enough_watch_history = np.count_nonzero(anime_list) >= self.threshold_watch_history
            if enough_watch_history:
                self.user_mapping[user.id] = user
                

        self.idxs = list(self.user_mapping.keys())

        # Filter/Preprocess the data
        for idx in self.user_mapping.keys():
            user:User = self.user_mapping[idx]
            filtered_watch_history:list[int] = []
            filtered_rating_history:list[int] = []

            for i,anime_id in enumerate(user.watch_history):

                # Filter out NSFW
                if Genre.Hentai not in self.anime_mapping[anime_id].genres:
                    filtered_watch_history.append(anime_id)
                    filtered_rating_history.append(user.rating_history[i])

                # Set unrated, watched shows to the average rating of the anime
                if normalize_unrated and filtered_rating_history[-1] == -1:
                    filtered_rating_history[-1] = self.anime_mapping[anime_id].rating

            user.watch_history = np.array(filtered_watch_history)
            user.rating_history = np.array(filtered_rating_history)
        
        # Split the data into train/test/val
        train_size = int(0.8 * len(self.idxs))
        val_size = int(0.1 * len(self.idxs))

        self.train_indices = self.idxs[:train_size]
        self.val_ids = self.idxs[train_size:train_size + val_size]
        self.test_ids = self.idxs[train_size + val_size:]

    def get_anime_info(self, id:int):
        return self.anime_mapping[id]

    ###################################################################################
    # This section is for evaluating on a single item. It's probably not what we'll use
    # in the long term, but can work for baselines and initial debugging.
    def start_eval_on_single_test(self):
        """
        Run this method to start the evaluation program.
        You MUST call end_eval_runtime() as soon as possible to get evaluation results.
        """
        user_id = self.test_ids[self.current_idx]
        user:User = self.user_mapping[user_id]
        user.generate_masked_history()
        user_masked_watch_history = user.get_masked_history()
        self.start_time = time.time()
        return user_masked_watch_history[None, ...], self.k

    def end_eval_on_single_test(self, k_recommended_shows: np.ndarray[int]):
        """
        Run this method to end the evaluation program.
        I expect k_recommended_shows to be of shape [test_set_size, k]. Test size should be 1.
        The columns should be ordered by ranking with 0 being the highest and 1 being
        the lowest.
        """
        total_runtime = time.time() - self.start_time

        k_recommended_shows = k_recommended_shows.squeeze()
        user_id = self.test_ids[self.current_idx]
        user:User = self.user_mapping[user_id]
        complete_watch_history = user.get_history()
        user_masked_watch_history = user.get_masked_history()
        ground_truth_of_masked_history = complete_watch_history - user_masked_watch_history
        

        pred = np.zeros(MAX_ANIME_COUNT, dtype=int)
        # Sort the indices and generate decreasing values
        decreasing_values = np.arange(len(k_recommended_shows), 0, -1)

        # Assign the decreasing values to the respective indices
        pred[k_recommended_shows] = decreasing_values

        score = ndcg_score(ground_truth_of_masked_history[None,...], pred[None,...], k=self.k)
        print(f'This model took {total_runtime} seconds.')
        print(f'Out of an optimal score of 1.0, you scored {score}.')
        return total_runtime, score
    ####################################################################################
    
    
    def start_eval_test_set(self):
        """
        Returns: 
        - user_masked_watch_history: Masked user watch history of shape [num_of_users, MAX_ANIME_COUNT]
        - k: number of anime recommendations your model should present
        """
        
        user_masked_watch_history = np.zeros((len(self.test_ids), MAX_ANIME_COUNT))
        user_watch_history = np.zeros((len(self.test_ids), MAX_ANIME_COUNT))
        for i,id in enumerate(self.test_ids):
            user = self.user_mapping[id]
            user.generate_masked_history()
            user_masked_watch_history[i] = user.get_masked_history()
            user_watch_history[i] = user.get_history()

        user_heldout_watch_history = user_watch_history - user_masked_watch_history

        self.start_time = time.time()
        return user_masked_watch_history, user_heldout_watch_history, self.k
    
    
    def end_eval_test_set(self, k_recommended_shows: np.ndarray[int]):
        """
        Run this method to end the evaluation program.
        I expect k_recommended_shows to be of shape [test_set_size, k].
        The columns should be ordered by ranking with 0 being the highest and 1 being
        the lowest.
        """
        total_runtime = time.time() - self.start_time

        complete_watch_history = np.zeros((len(self.test_ids), MAX_ANIME_COUNT))
        user_masked_watch_history = np.zeros((len(self.test_ids), MAX_ANIME_COUNT))
        ground_truth_of_masked_history = np.zeros((len(self.test_ids), MAX_ANIME_COUNT))
        for i,id in enumerate(self.test_ids):
            user = self.user_mapping[id]
            complete_watch_history[i] = user.get_history()
            user_masked_watch_history[i] = user.get_masked_history()
            ground_truth_of_masked_history[i] = complete_watch_history[i] - user_masked_watch_history[i]
        

        pred = np.zeros((len(self.test_ids), MAX_ANIME_COUNT), dtype=int)
        # Sort the indices and generate decreasing values
        decreasing_values = np.arange(k_recommended_shows.shape[1], 0, -1)


        # Assign the decreasing values to the respective indices
        # pred[np.array(k_recommended_shows)] = decreasing_values
        pred[np.arange(len(self.test_ids))[:, None], k_recommended_shows] = decreasing_values

        score = ndcg_score(ground_truth_of_masked_history, pred, k=self.k)
        print(f'This model took {total_runtime} seconds.')
        print(f'Out of an optimal score of 1.0, you scored {score}.')
        return total_runtime, score
