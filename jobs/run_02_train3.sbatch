#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --job-name=02train-smp-log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=30GB
#SBATCH --time=72:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=justin.salamon@gmail.com
#SBATCH --output="02train-smp-log-%j.out"

source ~/.bashrc
source activate py35milsed
# cd /home/$USER/dev/milsed/models/

SRCDIR=$HOME/dev/milsed/models
RUNDIR=$BEEGFS/jobs/milsed/run-$SLURM_JOBID
mkdir $RUNDIR
cd $RUNDIR

module purge
module load sox/intel/14.4.2
module load ffmpeg/intel/3.2.2
module load rubberband/intel/1.8.1
module load cuda/8.0.44
module load cudnn/8.0v5.1

python $SRCDIR/02_train.py crnnL3_7_smp_log model038 $SLURM_JOBID /scratch/js7561/datasets/dcase2017/task4_official/combined/features_silence/ /scratch/js7561/datasets/dcase2017/task4_official/combined/metadata/labels/groundtruth_strong_label_testing_set.csv --augment


