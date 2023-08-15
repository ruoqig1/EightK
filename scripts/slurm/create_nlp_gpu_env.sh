#!/usr/bin/env bash


module load fosscuda/2020b
module load gcc/10.2.0
module load cuda/11.1.1
module load openmpi/4.0.5
module load python/3.8.6
module load tensorflow/2.6.0

# check it exist! mkdir /data/gpfs/projects/punim2039/envs
virtualenv /data/gpfs/projects/punim2039/envs/nlp_gpu
source /data/gpfs/projects/punim2039/envs/nlp_gpu/bin/activate

pip install --upgrade pip==23.1.2
pip install six==1.15.0
pip install 'requests>=2.21.0,<3'
pip install pandas==1.2.0
pip install didipack==3.2.2
pip install beautifulsoup4==4.12.2
pip install transformers==4.30.2
pip install html2text==2020.1.16
pip install 'urllib3<2.0'
pip install wrds==3.1.6



python -c "import pandas; print(pandas.__version__)"


