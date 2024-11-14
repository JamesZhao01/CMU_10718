#!/bin/bash
#SBATCH --job-name=rec-sys
#SBATCH --output=slurmjobs/%j.out
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:A6000:1
#SBATCH --time=1:59:00
#SBATCH --mem=64G
#SBATCH --partition=general

export PYTHONPATH="src"
python src/recommender/run_pipeline.py \
  --dataset_config configs/datasets/id_dataset.json \
  --model_config configs/twotower/4_user_id_anime_id_title_fresher.json \
  --model TOWER \
  --should_return_ids