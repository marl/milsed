#!/bin/bash

#SBATCH --job-name=setupdrc
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=4GB
#SBATCH --time=12:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=justin.salamon@gmail.com
#SBATCH --output="00setup-%j.out"

source ~/.bashrc
cd /home/$USER/dev
source activate py35milsed

module purge
module load sox/intel/14.4.2
module load rubberband/intel/1.8.1
module load ffmpeg/intel/3.2.2

python ~/dev/milsed/models/00_setup_drc.py '/beegfs/js7561/datasets/dcase2017/task4_official/train/audio_silence/' '/beegfs/js7561/datasets/dcase2017/task4_official/train/audio_augmented/' --audio-ext wav --jams-ext jams --jobs 10 --start_index 10

