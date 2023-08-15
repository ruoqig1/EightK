#!/bin/bash

for i in {0..511}
do
cat > job.slurm << EOF
#!/bin/bash -l
#SBATCH --job-name=wsj_pre_$i
#SBATCH --output=/home/adidishe/EightK/out/preprocess_$i.out
#SBATCH --error=/home/adidishe/EightK/out/preprocess_$i.err
#SBATCH --chdir=/home/adidishe/EightK

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --time=1:00:00

module load gcccore/10.2.0 python/3.8.6
source /data/gpfs/projects/punim2039/envs/nlp/bin/activate

python3 data.py $i
EOF

sbatch job.slurm
done

