import argparse

def get_data():
    pass

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
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--embedder_name", type=str, default=None)
    args = parser.parse_args()
    print(args)

    get_data()
    sort_data()
    get_embedder()
    train_embedder()
    save_embedder()
