#!/bin/bash

for i in {0..50}
do
cat > job.slurm << EOF
#!/bin/bash -l
#SBATCH --job-name=GPU_embedding_process_$i
#SBATCH --output=/home/adidishe/EightK/out/GPU5embedding_process_$i.out
#SBATCH --error=/home/adidishe/EightK/out/GPU5embedding_process_$i.err
#SBATCH --chdir=/home/adidishe/EightK


#SBATCH -p gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=120G
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00

module load fosscuda/2020b
module load gcc/10.2.0
module load cuda/11.1.1
module load openmpi/4.0.5
module load python/3.8.6
module load tensorflow/2.6.0

source /data/gpfs/projects/punim2039/envs/nlp_gpu/bin/activate

# Run nvidia-smi to log GPU usage
while sleep 60; do nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv >> /home/adidishe/EightK/out/usage_$i.csv; done &

python3 embedding_processing_by_chunks.py $i 4


EOF

sbatch job.slurm
done

