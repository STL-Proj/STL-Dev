#!/bin/bash
#SBATCH --job-name=run_benchmark
#SBATCH --exclude=node0257,node0258
#SBATCH --account=ens
#SBATCH --partition=ens
#SBATCH --cluster=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --qos=gpu_ens_vlong

python -u benchmark_comp_sep.py