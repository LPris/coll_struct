#!/usr/bin/env bash
#
# Slurm arguments
#
#SBATCH --job-name=zayas_heur            # Name of the job
#SBATCH --export=ALL                # Export all environment variables
#SBATCH --output=logs/%j.log   # Log-file (important!)
#SBATCH --cpus-per-task=4           # Number of CPU cores to allocate
#SBATCH --mem=4G                   # Memory to allocate per allocated CPU core
#SBATCH --time=24:00:00                # Max execution time
#SBATCH --partition=batch

# Run your Python script
module load Python/3.7.4-GCCcore-8.3.0
./run_heur.sh