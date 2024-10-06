import argparse
import sys
import os

from src.utils import evaluator

def get_data():
    EVAL = evaluator.Evaluator(path=args.data_path)
    history_test_masked, history_test_heldout, test_k = EVAL.start_eval_test_set()

    user_dict = dict()
    for i in range(len(history_test_masked)):
        user_dict[i] = {
            'history': dict(),
            'predict': dict()
        }
        for j in history_test_masked[i].nonzero()[0]:
            anime_id = j.item()
            user_dict[i]['history'][anime_id] = history_test_masked[i][anime_id].item()
        for j in history_test_heldout[i].nonzero()[0]:
            anime_id = j.item()
            user_dict[i]['predict'][anime_id] = history_test_heldout[i][anime_id].item()

    anime_dict = dict()
    for i in EVAL.anime_mapping.keys():
        anime_dict[i] = {
            'Anime Name': EVAL.anime_mapping[i].name,
            'Genre': [genre.name for genre in EVAL.anime_mapping[i].genres],
            'Type': EVAL.anime_mapping[i].type.name,
            'Episodes': EVAL.anime_mapping[i].episodes,
            'Rating': EVAL.anime_mapping[i].rating,
        }
    import pdb; pdb.set_trace()
    return user_dict, anime_dict

def sort_data():
    pass    

def get_embedder():
    pass

def train_embedder():    
    pass

def save_embedder():
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default='/data/taeyoun_kim/jailbreak_work/attack_mlip/big_attack/data/copperunion')
    parser.add_argument("--embedder_name", type=str, default='mteb/gte-base')
    args = parser.parse_args()
    print(args)

    user_dict, anime_dict = get_data()
    model, tokenizer = get_embedder()
    train_embedder()
    save_embedder()
