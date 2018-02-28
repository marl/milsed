#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --job-name=train-medley-smp
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --time=72:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=brian.mcfee@nyu.edu
#SBATCH --output="train-medley-smp--%j.out"

source ~/.bashrc
source activate py35milsed
# cd /home/$USER/dev/milsed/models/

SRCDIR=$HOME/dev/milsed/models_medley
RUNDIR=/scratch/bm106/jobs/milsed/run-$SLURM_JOBID
mkdir -p $RUNDIR
cd $RUNDIR

module purge
module load sox/intel/14.4.2
module load ffmpeg/intel/3.2.2
module load rubberband/intel/1.8.1
module load cuda/8.0.44
module load cudnn/8.0v6.0

python -u $SRCDIR/02_train.py cnnL3_7_smp model_bm_l3_smp $SLURM_JOBID /scratch/bm106/MedleyDB_downmix/features $SRCDIR/resources/gt_all.csv


