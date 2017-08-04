#!/bin/bash

#SBATCH --job-name=prepareDRC
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=30GB
#SBATCH --time=12:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=justin.salamon@gmail.com
#SBATCH --output="01prepare-%j.out"

source ~/.bashrc
cd /home/$USER/dev/milsed/models
source activate py35milsed

module purge
module load sox/intel/14.4.2
module load rubberband/intel/1.8.1
module load ffmpeg/intel/3.2.2

python 01_prepare.py /beegfs/js7561/datasets/dcase2017/task4_official/train/audio_augmented /beegfs/js7561/datasets/dcase2017/task4_official/combined/features_silence --sample-rate 44100 --hop-length 1024 --nfft 2048 --nmels 128 --jobs 10 --loadpump

