#!/usr/bin/env bash
#echo "$1G"
Sinteractive -m 64G -G 1 -t 2:00:00 -c 16
#source /work/PRTNR/EPFL/CDM/smalamud/complexmodels/venv_opt/bin/activate
#module load gcc python cuda cudnn
#ray start --head
#export TRANSFORMERS_CACHE=/work/PRTNR/EPFL/CDM/smalamud/complexmodels/wsj/cache_hugging_face/export TRANSFORMERS_CACHE=/work/PRTNR/EPFL/CDM/smalamud/complexmodels/wsj/cache_hugging_face/
#salloc -p interactive -N 1 -c 8 --mem 64G --gres=gpu:2 --ntasks-per-gpu=1 -t 2:00:00

