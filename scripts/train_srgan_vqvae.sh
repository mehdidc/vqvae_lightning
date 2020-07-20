#!/bin/bash -x
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=10
#SBATCH --output=out
#SBATCH --error=err
#SBATCH --time=02:00:00
source ~/pyenv=
ml purge
ml use $OTHERSTAGES
ml Stages/2019a
ml GCC/8.3.0
ml ParaStationMPI/5.4.4-1-CUDA
#ml MVAPICH2/2.3.3-GDR
ml CUDA/10.1.105
ml NCCL/2.4.6-1-CUDA-10.1.105
ml cuDNN/7.5.1.10-CUDA-10.1.105
export NCCL_DEBUG=INFO
export NCCL_IB_CUDA_SUPPORT=0
export NCCL_IB_DISABLE=1
rm -fr results/srgan/vqvae
srun python -u cli.py train-vqvae configs/srgan_vqvae.yaml
