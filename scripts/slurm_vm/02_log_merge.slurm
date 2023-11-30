#!/bin/bash -l
#SBATCH --job-name 02_log_prime
#SBATCH --output=/home/unimelb.edu.au/adidisheim/EightK/out/02_log_prime.out
#SBATCH --error=/home/unimelb.edu.au/adidisheim/EightK/out/02_log_prime.err
#SBATCH --chdir=/home/unimelb.edu.au/adidisheim/EightK

#SBATCH --mail-type=ALL # notifications for job start, end, and failure
#SBATCH --mail-user=antoine.didisheim@unimelb.edu.au

#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 100G
#SBATCH --cpus-per-task 30
#SBATCH --time 72:00:00

source /home/unimelb.edu.au/adidisheim/venvs/nlp_gpu/bin/activate

python3 02_log_merge.py
