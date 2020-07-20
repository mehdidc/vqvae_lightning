#!/bin/bash -x
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --output=out
#SBATCH --error=err
#SBATCH --time=02:00:00
#SBATCH --partition=gpus
#SBATCH --gres=gpu:1
source ~/pyenv
ml purge
ml Stages/Devel-2019a
ml GCC/8.3.0
ml ParaStationMPI/5.4.4-1
ml CUDA/10.1.105
ml NCCL/2.4.6-1-CUDA-10.1.105
ml cuDNN/7.5.1.10-CUDA-10.1.105
export NCCL_DEBUG=INFO
rm results/test -fr
srun python -u cli.py train-vqvae configs/test_horovod.yaml #--checkpoint=results/test/model.th
