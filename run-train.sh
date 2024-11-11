#!/bin/bash
#SBATCH --job-name=attack-eval
#SBATCH --output=slurmjobs/%j.out
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=64G
#SBATCH --time=5:59:00
#SBATCH --exclude=babel-0-23,babel-3-25,babel-15-32,babel-0-37,babel-1-23,babel-1-31,shire-1-1,babel-11-9,babel-3-21,babel-4-17,babel-6-9,babel-7-9,babel-13-1,babel-13-13,babel-13-17,babel-13-25,babel-14-1,babel-14-37,shire-1-6,flame-8-21,shire-2-9,flame-9-21,flame-10-21,babel-12-21,babel-5-19,babel-4-33
#SBATCH --partition=preempt

##echo nodelist name
echo $SLURM_JOB_NODELIST

scontrol show job -d $SLURM_JOBID | grep GRES
source activate mlip
python train-embedder.py "$@" 
echo "job finished";