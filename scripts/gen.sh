#!/bin/bash -x
#SBATCH --partition=develgpus
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --output=gen_out
#SBATCH --error=gen_err
#SBATCH --time=00:10:00
source ~/pyenv
which python
ml CUDA
#ml MVAPICH2
which python
python -u cli.py transformer-generate results/imagenet/generator/model.th --device=cuda --nb-examples=100 --single-cond --custom-cond="monitor" --temperature=0.1 --top-k=50
