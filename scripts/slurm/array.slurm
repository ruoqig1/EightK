#!/bin/bash -l
#SBATCH --job-name array
#SBATCH --output=/home/unimelb.edu.au/adidisheim/EightK/out/array%a.out
#SBATCH --error=/home/unimelb.edu.au/adidisheim/EightK/out/array%a.err
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

python3 clean_eightk.py ${SLURM_ARRAY_TASK_ID}
