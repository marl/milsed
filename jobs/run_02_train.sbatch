#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --job-name=02train
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB
#SBATCH --time=12:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=justin.salamon@gmail.com
#SBATCH --output="02train-%j.out"

source ~/.bashrc
source activate py35milsed
cd /home/$USER/dev/milsed/models/

module purge
module load sox/intel/14.4.2
module load ffmpeg/intel/3.2.2
module load rubberband/intel/1.8.1
module load cuda/8.0.44
module load cudnn/8.0v5.1

python 02_train.py /beegfs/js7561/datasets/dcase2017/task4_official/train/features_silence/ --max_samples 1

