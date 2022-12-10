#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --time=00-01:30:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=64GB
#SBATCH --account=eecs595f22_class

# set up job
module load python3.9-anaconda  
# pushd essentially sets pwd
pushd /home/musicer/595/Joke-Generator/
source activate jokes

python3 appr2_joke_generator.py --train_file datasets/data/reddit_preprocessed/appr2/train.tsv --test_file datasets/data/reddit_preprocessed/appr2/test.tsv --model_file model_appr2_bart.torch
