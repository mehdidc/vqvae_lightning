#!/bin/bash -x
#SBATCH --partition=develgpus
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --output=out
#SBATCH --error=err
#SBATCH --time=02:00:00
source ~/pyenv
which python
ml CUDA
#ml MVAPICH2
which python
#rm -fr results/imagenet/vqvae
python -u cli.py train-vqvae configs/imagenet.yaml --checkpoint=results/imagenet/vqvae/model.th
