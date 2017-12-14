#!/bin/bash

#SBATCH --job-name=urbansed-00setup
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=4GB
#SBATCH --time=12:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=brian.mcfee@nyu.edu
#SBATCH --output="00setup-%j.out"

source ~/.bashrc
source activate py35milsed
cd /home/$USER/dev/milsed/models_urbansed

module purge
module load sox/intel/14.4.2
module load rubberband/intel/1.8.1
module load ffmpeg/intel/3.2.2

python -u 00-setup.py /scratch/$USER/URBAN-SED/audio/train /scratch/$USER/URBAN-SED/audio_aug/train --semitones 3 --audio-ext wav --jams-ext jams --jobs 10

