#!/bin/bash -l

#SBATCH --account smalamud_complexmodels
#SBATCH --mail-type ALL
#SBATCH --mail-user antoine.didisheim@unil.ch
#SBATCH --array=0-10

#SBATCH --chdir /scratch/adidishe/deep_recovery
#SBATCH --job-name dr_default
#SBATCH --output=/scratch/adidishe/EightK/out/gpu_%a.out
#SBATCH --error=/scratch/adidishe/EightK/out/gpu_%a.err
#SBATCH --chdir=/scratch/adidishe/EightK

#SBATCH -p gpu-a100
#SBATCH --gres=gpu:1

#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 12
#SBATCH --mem 64G
#SBATCH --time 24:00:00


module purge
module load cuda

# Check that the GPU is visible
nvidia-smi


module load gcc/9.3.0 python/3.8.8

source /work/PRTNR/EPFL/CDM/smalamud/complexmodels/venv/bin/activate

/work/FAC/HEC/DF/sscheid1/deep_replication/sq/

python3 /scratch/adidishe/EightK/cross_section_didi/launch_infinite_experiment.py 0 0