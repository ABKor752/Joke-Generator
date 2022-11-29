#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --time=00-01:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=64GB
#SBATCH --account=eecs595f22_class

# set up job
module load python3.9-anaconda  
# pushd essentially sets pwd
pushd /home/akorot/Joke-Generator/
source activate jokes

python3 simple_joke_generator.py --train_file datasets/data/reddit_preprocessed/unfunny.tsv --test_file datasets/data/reddit_preprocessed/test_unfunny.tsv --model_file model_unfunny_bart.torch
