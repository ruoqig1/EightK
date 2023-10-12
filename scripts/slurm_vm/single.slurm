#!/bin/bash -l
#SBATCH --job-name single
#SBATCH --output=/home/unimelb.edu.au/adidisheim/EightK/out/single.out
#SBATCH --error=/home/unimelb.edu.au/adidisheim/EightK/out/single.err
#SBATCH --chdir=/home/unimelb.edu.au/adidisheim/EightK

#SBATCH --mail-type=ALL # notifications for job start, end, and failure
#SBATCH --mail-user=antoine.didisheim@unimelb.edu.au

#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 64G
#SBATCH --cpus-per-task 16
#SBATCH --time 24:00:00

source /home/unimelb.edu.au/adidisheim/venvs/nlp_gpu/bin/activate
python3 eightk_01_from_raw_to_txt.py 9
