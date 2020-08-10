#!/bin/bash -x
#SBATCH --nodes=48
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --output=out
#SBATCH --error=err
#SBATCH --time=08:00:00
#SBATCH --partition=gpus
#SBATCH --gres=gpu:1
source ~/pyenv
ml purge
ml Stages/Devel-2019a
ml GCC/8.3.0
ml ParaStationMPI/5.4.4-1
ml CUDA/10.1.105
ml cuDNN/7.5.1.10-CUDA-10.1.105
export NCCL_DEBUG=INFO
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
#export HOROVOD_AUTOTUNE=1 
export HOROVOD_HIERARCHICAL_ALLGATHER=0 
export HOROVOD_HIERARCHICAL_ALLREDUCE=0 
export NCCL_IB_CUDA_SUPPORT=0
export NCCL_IB_DISABLE=1
#rm -fr results/imagenet03AUG/vqvae
srun --cpu-bind=none,v --accel-bind=gn python -u cli.py train-transformer-generator configs/transformer.yaml $*
