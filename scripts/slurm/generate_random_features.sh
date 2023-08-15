#!/bin/bash

for i in {0..20}
do
cat > job.slurm << EOF
#!/bin/bash -l
#SBATCH --job-name=grf_$i
#SBATCH --output=/home/adidishe/EightK/out/generate_random_features_$i.out
#SBATCH --error=/home/adidishe/EightK/out/generate_random_features_$i.err
#SBATCH --chdir=/home/adidishe/EightK

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00

module load fosscuda/2020b
module load gcc/10.2.0
module load cuda/11.1.1
module load openmpi/4.0.5
module load python/3.8.6
module load tensorflow/2.6.0

source /data/gpfs/projects/punim2039/envs/nlp_gpu/bin/activate

python3 generate_random_features.py $i
EOF

sbatch job.slurm
done

