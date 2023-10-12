#!/usr/bin/env bash
# scp -r ~/.ssh/  adidishe@spartan.hpc.unimelb.edu.au:/data/gpfs/projects/punim2039/EightK/data/ /scratch/nz97/ad4734/EightK/
scp -r ~/.ssh/  adidishe@spartan.hpc.unimelb.edu.au:/data/gpfs/projects/punim2039/EightK/data/training_norm/OPT_13b/EIGHT_LEGAL/* data/training_norm/OPT_13b/EIGHT_LEGAL/


scp -r ~/.ssh/  adidishe@spartan.hpc.unimelb.edu.au:/data/gpfs/projects/punim2039/EightK/data/cleaned/eight_k_first_process/*2013* /scratch/nz97/ad4734/EightK/data/cleaned/eight_k_first_process


scp -r ~/.ssh/ ./data/tfidf/* adidishe@spartan.hpc.unimelb.edu.au:/data/gpfs/projects/punim2039/EightK//data/tfidf/
scp -r ~/.ssh/ ./data/temp_cosine/* adidishe@spartan.hpc.unimelb.edu.au:/data/gpfs/projects/punim2039/EightK//data/temp_cosine/


# nohup scp -r ~/.ssh/  adidishe@spartan.hpc.unimelb.edu.au:/data/gpfs/projects/punim2039/EightK/data/p/*.p /scratch/nz97/ad4734/EightK/da &