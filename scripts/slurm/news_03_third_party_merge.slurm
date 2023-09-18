#!/bin/bash -l
#SBATCH --job-name third_party
#SBATCH --output=/home/adidishe/EightK/out/third_party_%a.out
#SBATCH --error=/home/adidishe/EightK/out/third_party_%a.err
#SBATCH --chdir=/home/adidishe/EightK
#SBATCH --array=0-28

#SBATCH --mail-type=ALL # notifications for job start, end, and failure
#SBATCH --mail-user=antoine.didisheim@unimelb.edu.au

#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 15G
#SBATCH --cpus-per-task 4
#SBATCH --time 24:00:00

source /home/adidishe/EightK/venv/bin/activate

python3 news_03_third_party_merge.py ${SLURM_ARRAY_TASK_ID}
