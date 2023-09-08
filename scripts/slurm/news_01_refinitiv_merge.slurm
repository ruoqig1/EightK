#!/bin/bash -l
#SBATCH --job-name news_merge_
#SBATCH --output=/home/unimelb.edu.au/adidisheim/EightK/out/news_merge_%a.out
#SBATCH --error=/home/unimelb.edu.au/adidisheim/EightK/out/news_merge_%a.err
#SBATCH --chdir=/home/unimelb.edu.au/adidisheim/EightK
#SBATCH --array=0-27

#SBATCH --mail-type=ALL # notifications for job start, end, and failure
#SBATCH --mail-user=antoine.didisheim@unimelb.edu.au

#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 15G
#SBATCH --cpus-per-task 4
#SBATCH --time 24:00:00

source /home/unimelb.edu.au/adidisheim/EightK/venv/bin/activate

python3 news_01_refintiv_merge.py ${SLURM_ARRAY_TASK_ID}
