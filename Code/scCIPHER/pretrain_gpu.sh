#!/bin/bash
#SBATCH --account=zitnik_mz189              # associated Slurm account
#SBATCH --job-name=pretrain_hgt_%j          # assign job name
#SBATCH --ntasks-per-node=1
#SBATCH -c 8                                # request cores
#SBATCH -t 2-00:00                          # runtime in D-HH:MM format
#SBATCH -p gpu_quad	                        # partition to run in
#SBATCH --mem=100G                          # memory for all cores
#SBATCH --gres=gpu:rtx8000:1
#SBATCH -o /n/data1/hms/dbmi/zitnik/lab/users/an252/NeuroKG/neuroKG/Results/slurm/pretrain_hgt_%j.out   # file to which STDOUT will be written, including job ID (%j)
#SBATCH -e /n/data1/hms/dbmi/zitnik/lab/users/an252/NeuroKG/neuroKG/Results/slurm/pretrain_hgt_%j.err   # file to which STDERR will be written, including job ID (%j)

# change working directory
cd /n/data1/hms/dbmi/zitnik/lab/users/an252/NeuroKG/neuroKG

# load modules
module load gcc/9.2.0 cuda/11.7 python/3.9.14

# activate environment
source neuroKG_env/bin/activate

# change working directory
cd Code/pretrain

# run script
# wandb agent ayushnoori/cipher-pretraining/XXXXX --count 1
python pretrain.py

# run with sbatch pretrain_gpu.sh
