#!/bin/bash -x
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --output=gen_out
#SBATCH --error=gen_err
#SBATCH --time=00:15:00
source ~/pyenv
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
python -u cli.py transformer-generate $1 --nb-examples=64 --device=cuda --temperature=1
