#!/bin/bash

for i in {0..100}
do
cat > job.slurm << EOF
#!/bin/bash -l
#SBATCH --job-name=embedding_process_$i
#SBATCH --output=/home/adidishe/EightK/out/embedding_process_$i.out
#SBATCH --error=/home/adidishe/EightK/out/embedding_process_$i.err
#SBATCH --chdir=/home/adidishe/EightK

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --time=6:00:00

module load gcccore/10.2.0 python/3.8.6
source /data/gpfs/projects/punim2039/envs/nlp/bin/activate

python3 embedding_processing_by_chunks.py $i 0
EOF

sbatch job.slurm
done

