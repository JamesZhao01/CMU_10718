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
from datetime import datetime, timedelta, timezone


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
    num_chunks: int,
    chunk_idx: int,
    force_retrain: bool,
    do_train: bool,
    do_test: bool,
):
    keys = list(sweep_config.keys())
    values = list(sweep_config.values())
    iteration = list(itertools.product(*values))
    total_experiments = len(iteration)

    est_offset = timezone(timedelta(hours=-5))
    current_est_time = datetime.now(est_offset).strftime("%Y-%m-%d %H:%M:%S")

    chunk_size = (total_experiments + num_chunks - 1) // num_chunks
    start = chunk_idx * chunk_size
    end_exclusive = min((chunk_idx + 1) * chunk_size, total_experiments)
    preface = f"=== {current_est_time} [{chunk_idx}/{num_chunks}]"
    print(
        f"{preface} Sweeping [{start}, {end_exclusive}) out of {total_experiments} ==="
    )

    for i in range(start, end_exclusive):
        sweep_values = iteration[i]
        sweep_config_i = dict(zip(keys, sweep_values))
        full_config = base_model_config | sweep_config_i
        save_to = os.path.join(output_dir, f"model_{i}.pt")
        os.makedirs(output_dir, exist_ok=True)
        log_file = save_to + ".log"
        if do_train:
            full_config = full_config | {"save_to": save_to}
        inner_preface = f"{preface} {i=}"

        if os.path.exists(log_file):
            with open(log_file, "r") as f:
                contents = f.read().strip()
                if contents.endswith("(Fin)"):
                    if not force_retrain:
                        print(
                            f"{inner_preface} File {log_file} already ends with (Fin)"
                        )
                        continue
                    else:
                        print(
                            f"{inner_preface} Retraining despite {log_file} already ends with (Fin)"
                        )

        print(f"{inner_preface} Config {json.dumps(sweep_config_i, indent=None)}")
        with open(save_to + ".log", "a") as f:
            f.write(
                f"{inner_preface} Config {json.dumps(sweep_config_i, indent=None)}\n"
            )

        if do_train:
            print(f"{inner_preface} Training")
            model = Models.from_string(model_type)(datamodule=datamodule, **full_config)
            model.train()
        else:
            if do_test and os.path.exists(save_to):
                print(f"{inner_preface} Skipping training, loading {save_to}")
                full_config = full_config | {"load_from": save_to}
                model = Models.from_string(model_type)(
                    datamodule=datamodule, **full_config
                )
            elif not do_test:
                print(f"{inner_preface} Warn: Skipping training and testing ...?")
                continue
            else:
                print(f"{inner_preface} Error: No checkpoint to load {save_to}")
                continue
        if do_test:
            testbench = TestBench(datamodule, should_return_ids=True)
            metrics = testbench.full_evaluation(model, return_scores=False)
            with open(save_to + ".metrics.log", "w") as f:
                f.write(json.dumps(metrics, indent=2))


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

    num_chunks = args["num_chunks"]
    chunk_idx = args["chunk_idx"]
    force_retrain = args["force_retrain"]
    do_train = args["do_train"]
    do_test = args["do_test"]
    assert num_chunks > 0
    assert chunk_idx >= 0 and chunk_idx < num_chunks
    sweep(
        datamodule=datamodule,
        model_type=args["model"],
        base_model_config=model_config,
        sweep_config=sweep_config,
        output_dir=output_dir,
        num_chunks=num_chunks,
        chunk_idx=chunk_idx,
        force_retrain=force_retrain,
        do_train=do_train,
        do_test=do_test,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_config", type=str, required=True)
    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--sweep_config", type=str, required=True)
    parser.add_argument("--num_chunks", type=int, required=True)
    parser.add_argument("--chunk_idx", type=int, required=True)
    parser.add_argument("--force_retrain", action="store_true")
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_test", action="store_true")

    parser.add_argument(
        "--model", type=str, default="TOWER", choices=[val.name for val in Models]
    )
    parser.add_argument("--output_dir", type=str, default="models/sweep/default")
    args = vars(parser.parse_args())
    main(args)
