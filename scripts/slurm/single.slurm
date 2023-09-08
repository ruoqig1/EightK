#!/bin/bash -l
#SBATCH --job-name single
#SBATCH --output=/home/unimelb.edu.au/adidisheim/EightK/out/single.out
#SBATCH --error=/home/unimelb.edu.au/adidisheim/EightK/out/single.err
#SBATCH --chdir=/home/unimelb.edu.au/adidisheim/EightK

#SBATCH --mail-type=ALL # notifications for job start, end, and failure
#SBATCH --mail-user=antoine.didisheim@unimelb.edu.au

#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 30G
#SBATCH --cpus-per-task 8
#SBATCH --time 24:00:00

source /home/unimelb.edu.au/adidisheim/EightK/venv/bin/activate

python3 wip.py