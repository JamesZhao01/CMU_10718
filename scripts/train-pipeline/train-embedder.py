import argparse
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from src.utils import evaluator

def get_data():
    EVAL = evaluator.Evaluator(path=args.data_path)
    import pdb; pdb.set_trace()
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
    parser.add_argument("--data_path", type=str, default='/data/taeyoun_kim/jailbreak_work/attack_mlip/big_attack/data')
    parser.add_argument("--embedder_name", type=str, default=None)
    args = parser.parse_args()
    print(args)

    get_data()
    sort_data()
    get_embedder()
    train_embedder()
    save_embedder()
