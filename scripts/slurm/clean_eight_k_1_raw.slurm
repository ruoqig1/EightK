#!/bin/bash -l
#SBATCH --job-name="clean_eight_k"
#SBATCH --output=/home/adidishe/EightK/out/clean_eight_k%a.out
#SBATCH --error=/home/adidishe/EightK/out/clean_eight_k%a.err
#SBATCH --chdir=/home/adidishe/EightK
#SBATCH --array=0-20

# Partition for the job:
#SBATCH --partition=cascade
#SBATCH --account="punim2039"
#SBATCH --mail-type=ALL # notifications for job start, end, and failure
#SBATCH --mail-user=antoine.didisheim@unimelb.edu.au
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END


#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=15G
#SBATCH --cpus-per-task=4
#SBATCH --time=0-1:10:00


source /home/adidishe/EightK/venv/bin/activate

python3 eightk_01_from_raw_to_txt.py ${SLURM_ARRAY_TASK_ID}
