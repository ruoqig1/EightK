#!/bin/bash -l
#SBATCH --job-name third_party
#SBATCH --output=/home/unimelb.edu.au/adidisheim/EightK/out/third_party_%a.out
#SBATCH --error=/home/unimelb.edu.au/adidisheim/EightK/out/third_party_%a.err
#SBATCH --chdir=/home/unimelb.edu.au/adidisheim/EightK/
#SBATCH --array=23-25

#SBATCH --mail-type=ALL # notifications for job start, end, and failure
#SBATCH --mail-user=antoine.didisheim@unimelb.edu.au

#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 100G
#SBATCH --cpus-per-task 30
#SBATCH --time 24:00:00

source /home/unimelb.edu.au/adidisheim/EightK/venv/bin/activate

python3 news_03_third_party_merge.py ${SLURM_ARRAY_TASK_ID}
