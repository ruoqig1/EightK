#!/bin/bash -l
#SBATCH --job-name 01_log
#SBATCH --output=/home/unimelb.edu.au/adidisheim/EightK/out/01_log%a.out
#SBATCH --error=/home/unimelb.edu.au/adidisheim/EightK/out/01_log%a.err
#SBATCH --chdir=/home/unimelb.edu.au/adidisheim/EightK
#SBATCH --array=0-100

#SBATCH --mail-type=ALL # notifications for job start, end, and failure
#SBATCH --mail-user=antoine.didisheim@unimelb.edu.au

#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 8G
#SBATCH --cpus-per-task 2
#SBATCH --time 24:00:00

source /home/unimelb.edu.au/adidisheim/venvs/nlp_gpu/bin/activate

python3 01_log_unzip_and_process.py ${SLURM_ARRAY_TASK_ID}
