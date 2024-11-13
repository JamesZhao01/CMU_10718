import argparse
import itertools
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


def sweep(
    datamodule: DataModule,
    model_type: str,
    base_model_config: dict,
    sweep_config: dict,
    output_dir: str,
):
    keys = list(sweep_config.keys())
    values = list(sweep_config.values())
    for i, sweep_values in enumerate(itertools.product(*values)):
        sweep_config_i = dict(zip(keys, sweep_values))
        full_config = base_model_config | sweep_config_i
        save_to = os.path.join(output_dir, f"model_{i}.pt")
        full_config = full_config | {"save_to": save_to}
        print(f"# Sweep {i} with config {json.dumps(sweep_config_i, indent=None)}")
        model = Models.from_string(model_type)(datamodule=datamodule, **full_config)
        model.train()


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
    if args["sweep_config"]:
        with open(args["sweep_config"], "r") as f:
            sweep_config = json.load(f)

    os.makedirs(output_dir, exist_ok=True)
    datamodule = DataModule(**dataset_config)
    auxiliary_args = {
        "n_users": datamodule.max_user_count,
        "n_anime": datamodule.max_anime_count,
    }
    model_config = model_config | auxiliary_args

    sweep(
        datamodule=datamodule,
        model_type=args["model"],
        base_model_config=model_config,
        sweep_config=sweep_config,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_config", type=str, required=True)
    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--sweep_config", type=str, required=True)
    parser.add_argument(
        "--model", type=str, default="TOWER", choices=[val.name for val in Models]
    )
    parser.add_argument("--output_dir", type=str, default="models/sweep/default")
    args = vars(parser.parse_args())
    main(args)
