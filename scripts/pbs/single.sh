#!/bin/bash
#PBS -P nz97
#PBS -q normal
#PBS -l walltime=24:00:00
#PBS -l storage=scratch/nz97
#PBS -l mem=64GB
#PBS -l ncpus=4
#PBS -M antoine.didisheim@unimelb.edu.au
#PBS -N debug2
#PBS -o out/debug2.out
#PBS -e out/debug2.err
#PBS -l wd=/scratch/nz97/ad4734/EightK/

module load python3/3.10.4
source ~/scratch/nz97/ad4734/venv/bin/activate
python train_main.py