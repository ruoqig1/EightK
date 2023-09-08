#!/bin/bash -l
#SBATCH --job-name news_bow
#SBATCH --output=/home/unimelb.edu.au/adidisheim/EightK/out/news_bow%a.out
#SBATCH --error=/home/unimelb.edu.au/adidisheim/EightK/out/news_bow%a.err
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

python3 news_02_refinitive_to_bow.py ${SLURM_ARRAY_TASK_ID}
