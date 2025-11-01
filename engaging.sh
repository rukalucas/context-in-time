#!/bin/bash
#SBATCH --job-name=parallel_multitask
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00

module load miniforge
source .venv/bin/activate

# Set number of threads for PyTorch
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Number of CPU threads: $OMP_NUM_THREADS"

# Run training with memory-reduced settings
python main.py configs/parallel_multitask.yaml \
    training.log_dir=logs/parallel_multitask_$(date +%Y%m%d_%H%M%S)