#!/bin/bash -l
#SBATCH --job-name news_create_list_of_topics
#SBATCH --output=/home/unimelb.edu.au/adidisheim/EightK/out/topics.out
#SBATCH --error=/home/unimelb.edu.au/adidisheim/EightK/out/topics.err
#SBATCH --chdir=/home/unimelb.edu.au/adidisheim/EightK

#SBATCH --mail-type=ALL # notifications for job start, end, and failure
#SBATCH --mail-user=antoine.didisheim@unimelb.edu.au

#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 30G
#SBATCH --cpus-per-task 8
#SBATCH --time 24:00:00

source /home/unimelb.edu.au/adidisheim/EightK/venv/bin/activate

python3 news_create_list_of_topics.py --para=1