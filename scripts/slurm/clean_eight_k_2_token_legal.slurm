#!/bin/bash -l
#SBATCH --job-name="tokenise_legal"
#SBATCH --output=/home/adidishe/EightK/out/tokenise_legal%a.out
#SBATCH --error=/home/adidishe/EightK/out/tokenise_legal%a.err
#SBATCH --chdir=/home/adidishe/EightK
#SBATCH --array=0-20

# Partition for the job:
#SBATCH --partition=cascade
#SBATCH --account="punim2039"

#SBATCH --mail-type=ALL # notifications for job start, end, and failure
#SBATCH --mail-user=antoine.didisheim@unimelb.edu.au
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END


#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --time=0-5:00:00

module load foss/2022a
module load GCC/11.3.0
module load CUDA/11.7.0
module load GCCcore/11.3.0; module load Python/3.10.4

source ~/venvs/nlp_gpu/bin/activate
python3 eightk_02_from_txt_to_bag.py ${SLURM_ARRAY_TASK_ID} --legal=1
