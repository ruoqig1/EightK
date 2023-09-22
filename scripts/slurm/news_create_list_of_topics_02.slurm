#!/bin/bash -l
#SBATCH --job-name="news_create_list_of_topics_02"
#SBATCH --output=/home/adidishe/EightK/out/news_create_list_of_topics_02.out
#SBATCH --error=/home/adidishe/EightK/out/news_create_list_of_topics_02.err
#SBATCH --chdir=/home/adidishe/EightK

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
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=0-12:00:00

module load foss/2022a
module load GCCcore/11.3.0; module load Python/3.10.4
module load  cuDNN/8.4.1.50-CUDA-11.7.0
module load TensorFlow/2.11.0-CUDA-11.7.0
source  ~/venvs/nlp_gpu/bin/activate

python3 news_create_list_of_topics_02.py
