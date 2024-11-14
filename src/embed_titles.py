from transformers import AutoModel, AutoTokenizer
from torch import Tensor
import torch
import os, json
from utils.data.data_module import DataModule
import tqdm
import numpy as np

model = AutoModel.from_pretrained("thenlper/gte-base").to("cuda")
tokenizer = AutoTokenizer.from_pretrained("thenlper/gte-base")


def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def process_batch(batch_of_text):
    model.eval()
    with torch.no_grad():
        batch_dict = tokenizer(
            batch_of_text,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        batch_dict = {k: v.to("cuda") for k, v in batch_dict.items()}
        outputs = model(**batch_dict)
        embeddings = average_pool(
            outputs.last_hidden_state, batch_dict["attention_mask"]
        )
        return embeddings

if os.getcwd().endswith("src"):
    os.chdir("..")

dataset_config = {}
with open("configs/datasets/id_dataset.json", "r") as f:
    dataset_config_2 = json.load(f)
    dataset_config.update(dataset_config_2)
datamodule = DataModule(**dataset_config)

all_embeddings = []
n_anime = datamodule.max_anime_count
for i in tqdm.tqdm(range(0, n_anime, 64)):
    samples = [
        datamodule.canonical_anime_mapping[i + j].name
        for j in range(64)
        if i + j < n_anime
    ]
    all_embeddings.append(process_batch(samples).detach().cpu().numpy())

stacked_embeddings = np.vstack(all_embeddings)
os.makedirs("data/embeddings", exist_ok=True)
np.save("data/embeddings/gte-base_titles.npy", stacked_embeddings)

print(stacked_embeddings.shape)
print(stacked_embeddings.dtype)