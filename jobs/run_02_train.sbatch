#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --job-name=02train
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB
#SBATCH --time=48:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=justin.salamon@gmail.com
#SBATCH --output="02train-%j.out"

source ~/.bashrc
source activate py35milsed
# cd /home/$USER/dev/milsed/models/

SRCDIR=$HOME/dev/milsed/models
RUNDIR=$BEEGFS/jobs/milsed/run-$(SLURM_JOB_ID)
mkdir $RUNDIR
cd $RUNDIR

module purge
module load sox/intel/14.4.2
module load ffmpeg/intel/3.2.2
module load rubberband/intel/1.8.1
module load cuda/8.0.44
module load cudnn/8.0v5.1

python $SRCDIR/02_train.py crnn1d_smp /beegfs/js7561/datasets/dcase2017/task4_official/combined/features_silence/ /beegfs/js7561/datasets/dcase2017/task4_official/combined/metadata/labels/groundtruth_strong_label_testing_set.csv

