#!/bin/bash -x
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --output=out
#SBATCH --error=err
#SBATCH --time=02:00:00
source ~/pyenv
ml purge
ml use $OTHERSTAGES
ml Stages/2019a
ml GCC/8.3.0
ml MVAPICH2/2.3.3-GDR
ml CUDA/10.1.105
ml NCCL/2.4.6-1-CUDA-10.1.105
ml cuDNN/7.5.1.10-CUDA-10.1.105
export NCCL_DEBUG=INFO
srun python -u cli.py train-transformer-generator configs/test_horovod_gen.yaml 
