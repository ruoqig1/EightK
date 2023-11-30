#!/bin/bash -l
#SBATCH --job-name 04_log
#SBATCH --output=/home/unimelb.edu.au/adidisheim/EightK/out/04_log%a.out
#SBATCH --error=/home/unimelb.edu.au/adidisheim/EightK/out/04_log%a.err
#SBATCH --chdir=/home/unimelb.edu.au/adidisheim/EightK
#SBATCH --array=90-100

#SBATCH --mail-type=ALL # notifications for job start, end, and failure
#SBATCH --mail-user=antoine.didisheim@unimelb.edu.au

#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 16G
#SBATCH --cpus-per-task 4
#SBATCH --time 24:00:00

source /home/unimelb.edu.au/adidisheim/venvs/nlp_gpu/bin/activate

python3 04_log_unzip_high_frequ.py ${SLURM_ARRAY_TASK_ID}
