import os
import pandas as pd
from enum import Enum
import random


class Type(Enum):
    TV = 1
    OVA = 2
    Movie = 3
    Special = 4
    ONA = 5
    Music = 6

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
    def __init__(self, id:int, watch_history:list[int], rating_history:list[int], k):
        self.id:int = id
        self.watch_history:list[int] = watch_history
        # Usually, ratings are ints, but if we normalize the unrated shows to the average
        # they will be a float.
        self.rating_history:list[float] = rating_history
        self.k = k
        self.masked_watch_history:list[int] = []
        self.masked_rating_history:list[int] = []

    def generate_masked_history(self):
        """
        Generates the masked history based on the current watch/rating history.
        Note that this should be called after the filtering process, so that
        the masked information does not include any filtered out data.
        """
        
        # Randomly select indices to mask (subsample)
        selected_indices = random.sample(range(len(self.watch_history)), self.k)

        # Subsample both lists using the selected indices
        self.masked_watch_history = [self.watch_history[i] for i in selected_indices]
        self.masked_rating_history = [self.rating_history[i] for i in selected_indices]

    def get_history(self):
        """
        Returns a one-hot encoding of the rating history of the user.
        The index is the anime id. 
        (Note that index 0 is always just 0 since anime id starts with 1).
        """

        one_hot_encoding = [0.0] * 12_294
        # Populate the one-hot list with ratings at indices corresponding to anime_id
        for anime_id, rating in zip(self.watch_history, self.rating_history):
            one_hot_encoding[anime_id] = rating

        return one_hot_encoding
    
    def get_masked_history(self):
        """
        Same as get_history, but some of the information is masked out.
        """
        one_hot_encoding = [0.0] * 12_294
        # Populate the one-hot list with ratings at indices corresponding to anime_id
        for anime_id, rating in zip(self.masked_watch_history, self.masked_rating_history):
            one_hot_encoding[anime_id] = rating

        return one_hot_encoding
        

class Evaluator:
    def __init__(self, path: str, normalize_unrated = True):
        self.animes = pd.read_csv(os.path.join(path, 'anime.csv'))
        self.users = pd.read_csv(os.path.join(path, 'rating.csv'))
        self.k = 10
        self.threshold_watch_history = 20
        self.user_mapping = {}
        self.anime_mapping = {}
        self.idxs = []
        self.normalize_unrated = normalize_unrated
        random.seed(42)

        # Gets all the information about the animes
        for anime_id, anime_df in self.animes.iterrows():
            anime = Anime(id = anime_df['id'],
                  name = anime_df['name'],
                  genres = [getattr(Genre,genre.strip()) for genre in anime_df['genre'].split(',')],
                  type = getattr(Type,anime_df['type']),
                  episodes = anime_df['episodes'],
                  rating = anime_df['rating'],
                  membership_count = anime_df['members']
                  )
            self.anime_mapping[anime_id] = anime

        # Gets all the user watch history
        for user_id, user_df in self.users.groupby('user_id'):
            anime_list = user_df['anime_id'].tolist()
            rating_list = user_df['rating'].tolist()

            assert len(anime_list) == len(rating_list)

            user = User(user_id, anime_list, rating_list)
            
            # Filters out users that have too little watch history.
            if anime_list >= self.threshold_watch_history:
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
                if normalize_unrated and user.rating_history[i] == -1:
                    user.rating_history[i] = self.anime_mapping[anime_id].rating

            user.watch_history = filtered_watch_history
            user.rating_history = filtered_rating_history
        
        # Split the data into train/test/val
        train_size = int(0.8 * len(self.idxs))
        val_size = int(0.1 * len(self.idxs))

        self.train_indices = self.idxs[:train_size]
        self.val_indices = self.idxs[train_size:train_size + val_size]
        self.test_indices = self.idxs[train_size + val_size:]

Evaluator('./data/',normalize_unrated=True)