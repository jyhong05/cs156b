#!/bin/bash

#SBATCH -J "h3_156b_fast"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_l40s:1

# Exit immediately if a command exits with a non-zero status
set -euo pipefail

source /resnick/groups/CS156b/from_central/2026/h3/cs156b/venv/bin/activate
CONFIG_PATH=${CONFIG_PATH:-configs/customcnn.json}

python src/train.py --config "$CONFIG_PATH"
python src/predict.py --config "$CONFIG_PATH"
