#!/bin/bash

#SBATCH -J "h3_156b_fast"
#SBATCH --nodes=1                
#SBATCH --ntasks=4               
#SBATCH --mem=16G                
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:2

# Exit immediately if a command exits with a non-zero status
set -euo pipefail

source /resnick/groups/CS156b/from_central/2026/h3/cs156b/venv/bin/activate
python src/predict.py
