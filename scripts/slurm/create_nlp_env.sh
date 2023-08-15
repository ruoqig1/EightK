#!/usr/bin/env bash

module load gcccore/10.2.0 python/3.8.6
# check it exist! mkdir /data/gpfs/projects/punim2039/envs
virtualenv /data/gpfs/projects/punim2039/envs/nlp
source /data/gpfs/projects/punim2039/envs/nlp/bin/activate
pip install --upgrade pip==23.1.2
pip install tensorflow==2.5.0
pip install didipack==3.2.2
pip install numpy==1.19.2
pip install beautifulsoup4==4.12.2
pip install transformers==4.30.2
pip install html2text==2020.1.16