#!/bin/bash

#SBATCH -N 1                             # number of compute nodes
#SBATCH -n 1                             # number of CPU cores to reserve on this compute node

#SBATCH -J dl_final_p                    # Job name
#SBATCH -o slurm.%j.out                  # STDOUT (%j = JobId)
#SBATCH -e slurm.%j.err                  # STDERR (%j = JobId)
#SBATCH -p physicsgpu1
#SBATCH -q wildfire 
#SBATCH --gres=gpu:2                     # Request two GPUs
#SBATCH --export=ALL
#SBATCH -t 0-48:00                       # wall time (D-HH:MM)
#SBATCH --mail-user=mkhoshle@asu.edu     # email address
#SBATCH --mail-type=all                  # type of mail to send

#The next line is required if the user has more than one project
# #SBATCH -A A-yourproject # <-- Allocation name to charge job against

module load cuda/9.1.85 
module load gcc/8.2.0
module load cudnn/7.0
# Private conda environment
source activate ml-gpu

hostnodes=`scontrol show hostnames $SLURM_NODELIST`
echo $hostnodes

SCHEDULER=`hostname`
echo SCHEDULER: $SCHEDULER

#python -c 'import torch; print(torch.rand(2,3).cuda())'
#python -c 'import torch; print(torch.cuda.is_available())'
my_file=Training
#my_file=collect_env
python $SLURM_SUBMIT_DIR/$my_file.py


