#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --time=00-06:00:00
#SBATCH --gpus=2
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=64GB
#SBATCH --account=eecs595f22_class

# set up job
module load python3.9-anaconda  
# pushd essentially sets pwd
pushd /home/musicer/595/Joke-Generator/
source activate jokes

python3 simple_joke_generator.py
