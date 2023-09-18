#!/bin/bash -l
#SBATCH --job-name clean_eight_k
#SBATCH --output=/home/unimelb.edu.au/adidisheim/EightK/out/clean_eight_k%a.out
#SBATCH --error=/home/unimelb.edu.au/adidisheim/EightK/out/clean_eight_k%a.err
#SBATCH --chdir=/home/unimelb.edu.au/adidisheim/EightK
#SBATCH --array=0-20

#SBATCH --mail-type=ALL # notifications for job start, end, and failure
#SBATCH --mail-user=antoine.didisheim@unimelb.edu.au

#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 15G
#SBATCH --cpus-per-task 4
#SBATCH --time 24:00:00

source /home/unimelb.edu.au/adidisheim/EightK/venv/bin/activate

python3 eightk_01_from_raw_to_txt.py ${SLURM_ARRAY_TASK_ID}
