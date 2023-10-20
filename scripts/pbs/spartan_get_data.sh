#!/usr/bin/env bash
# scp -r ~/.ssh/  adidishe@spartan.hpc.unimelb.edu.au:/data/gpfs/projects/punim2039/EightK/data/ /scratch/nz97/ad4734/EightK/
scp -r ~/.ssh/  adidishe@spartan.hpc.unimelb.edu.au:/data/gpfs/projects/punim2039/EightK/data/training_norm/OPT_13b/EIGHT_LEGAL/* data/training_norm/OPT_13b/EIGHT_LEGAL/


scp -r ~/.ssh/  adidishe@spartan.hpc.unimelb.edu.au:/data/gpfs/projects/punim2039/EightK/data/p/abn_ev_monly.p /scratch/nz97/ad4734/EightK/data/p/



scp -r ~/.ssh/ ./data/tfidf/* adidishe@spartan.hpc.unimelb.edu.au:/data/gpfs/projects/punim2039/EightK//data/tfidf/
scp -r ~/.ssh/ adidishe@spartan.hpc.unimelb.edu.au:/data/gpfs/projects/punim2039/EightK/data/training_norm/OPT_13b/NEWS_REF_ON_EIGHT_K/*  ./
scp -r ~/.ssh/ ./res/temp/vec_pred/* adidishe@spartan.hpc.unimelb.edu.au:/data/gpfs/projects/punim2039/EightK/res/temp/vec_pred/


# nohup scp -r ~/.ssh/  adidishe@spartan.hpc.unimelb.edu.au:/data/gpfs/projects/punim2039/EightK/data/p/*.p /scratch/nz97/ad4734/EightK/da &