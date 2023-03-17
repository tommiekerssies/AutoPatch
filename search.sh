#!/bin/bash
#SBATCH -p oahu
#SBATCH --gres=gpu:1
#SBATCH --error=logs/%j.log
#SBATCH --output=logs/%j.log
#SBATCH -t 167:00:00
#SBATCH --mem 16gb

source /home/tommie_kerssies/miniconda3/etc/profile.d/conda.sh
conda activate AutoPatch

srun python search.py --accelerator auto --study_name $1 --n_trials $2 --k $3 --seed $4 --category $5 --test_set_search $6
