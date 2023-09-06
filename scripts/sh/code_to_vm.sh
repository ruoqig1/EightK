#!/usr/bin/env bash
scp *.py ADIDISHEIM@vm-172-26-151-77.desktop.cloud.unimelb.edu.au:/home/unimelb.edu.au/adidisheim/EightK/
scp utils_local/*.py ADIDISHEIM@vm-172-26-151-77.desktop.cloud.unimelb.edu.au:/home/unimelb.edu.au/adidisheim/EightK/utils_local
scp scripts/slurm/* ADIDISHEIM@vm-172-26-151-77.desktop.cloud.unimelb.edu.au:/home/unimelb.edu.au/adidisheim/EightK/
# scp summary_stat/* ADIDISHEIM@vm-172-26-151-77.desktop.cloud.unimelb.edu.au:/home/unimelb.edu.au/adidisheim/EightK/summary_stat/


#scp data/raw/crsp_monthly.csv adidishe@spartan.hpc.unimelb.edu.au:/data/gpfs/projects/punim2039/EightK/data/raw/
