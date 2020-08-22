#!/bin/bash -x
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --output=out
#SBATCH --error=err
#SBATCH --time=08:00:00
#SBATCH --partition=gpus
#SBATCH --gres=gpu:1
#SBATCH --account=ccstdl
source ~/pyenv
conda activate jusuf
ml purge
ml Stages/Devel-2019a
ml GCC/8.3.0
ml ParaStationMPI/5.2.2-1
ml CUDA/10.2.89
export NCCL_DEBUG=INFO
export NCCL_NET_GDR_LEVEL=3 # enable GDR over IB, Level 3
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
#export OMP_NUM_THREADS=1
export HOROVOD_AUTOTUNE=1
#export HOROVOD_AUTOTUNE_LOG=autotune_log.csv
#rm -fr results/imagenet03AUG/vqvae
srun --cpu-bind=none,v --accel-bind=gn python -u cli.py train-transformer-generator configs/transformer.yaml $*
