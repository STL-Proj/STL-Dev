#!/bin/bash
#SBATCH --job-name=run_comp_sep
#SBATCH --exclude=node0257,node0258
#SBATCH --account=ens
#SBATCH --partition=ens
#SBATCH --cluster=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --qos=gpu_ens_vlong

python -u comp_sep_5channels.py