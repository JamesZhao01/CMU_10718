import argparse
import os
import json
import numpy as np
from enum import Enum

from recommender.popularity_recommender import PopularityRecommender
from recommender.jaccard_recommender import JaccardRecommender
from recommender.tower_recommender import TowerRecommender
from utils.data.data_module import DataModule
from utils.data.testbench import TestBench


class Models(Enum):
    POPULARITY = PopularityRecommender
    JACCARD = JaccardRecommender
    TOWER = TowerRecommender

    @classmethod
    def from_string(cls, name):
        try:
            return cls[name].value
        except KeyError:
            raise ValueError(f"No Model with name '{name}'")


def main(args: dict):
    dataset_config = {}
    model_config = {}
    output_dir = args["output_dir"]
    if args["dataset_config"]:
        with open(args["dataset_config"], "r") as f:
            dataset_config_2 = json.load(f)
            dataset_config.update(dataset_config_2)
    if args["model_config"]:
        with open(args["model_config"], "r") as f:
            model_config = json.load(f)

    os.makedirs(output_dir, exist_ok=True)
    datamodule = DataModule(**dataset_config)

    testbench = TestBench(datamodule, **dataset_config)
    auxiliary_args = {
        "n_users": datamodule.max_user_count,
        "n_anime": datamodule.max_anime_count,
    }
    model_config = model_config + auxiliary_args
    model = Models.from_string(args["model"])(data_module=datamodule, **model_config)

    model.train()
    metrics = testbench.full_evaluation(model)

    with open(os.path.join(output_dir, "output.txt"), "w") as f:
        for k, v in metrics.items():
            if type(v) == np.ndarray:
                v = v.tolist()
            f.write(f"{k}: {v}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_config", type=str, default="")
    parser.add_argument("--model_config", type=str, default="")
    parser.add_argument(
        "--model", type=str, default="POPULARITY", choices=[val.name for val in Models]
    )
    parser.add_argument("--output_dir", type=str, default="models/default")
    args = vars(parser.parse_args())
    main(args)
