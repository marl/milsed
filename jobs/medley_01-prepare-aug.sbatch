#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --job-name=01prepare_medley_aug
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=12:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=brian.mcfee@nyu.edu
#SBATCH --output="01prepare-aug-%j.out"

source ~/.bashrc
source activate py35milsed
cd /home/$USER/dev/milsed/models_medley

module purge
module load sox/intel/14.4.2
module load rubberband/intel/1.8.1
module load ffmpeg/intel/3.2.2
module load cuda/8.0.44
module load cudnn/8.0v5.1

python -u 01_prepare.py /scratch/bm106/MedleyDB_downmix/aug /scratch/bm106/MedleyDB_downmix/features --sample-rate 44100 --hop-length 1024 --nfft 2048 --nmels 128 --jobs 4 --overwrite

