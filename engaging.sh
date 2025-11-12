#!/bin/bash
#SBATCH --job-name=context_in_time
#SBATCH --output=logs/slurm_logs/slurm_%j.out
#SBATCH --error=logs/slurm_logs/slurm_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=12:00:00

module load miniforge
source .venv/bin/activate

python main.py configs/sequence_instructed.yaml \
    training.log_dir=logs/sequence_instructed