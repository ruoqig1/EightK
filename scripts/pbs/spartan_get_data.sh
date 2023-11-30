#!/usr/bin/env bash
# scp -r ~/.ssh/  adidishe@spartan.hpc.unimelb.edu.au:/data/gpfs/projects/punim2039/EightK/data/ /scratch/nz97/ad4734/EightK/
scp -r ~/.ssh/  adidishe@spartan.hpc.unimelb.edu.au:/data/gpfs/projects/punim2039/EightK/data/training_norm/OPT_13b/EIGHT_LEGAL_ATI_TRAIN/* data/training_norm/OPT_13b/EIGHT_LEGAL_ATI_TRAIN/


scp -r ~/.ssh/  adidishe@spartan.hpc.unimelb.edu.au:/data/gpfs/projects/punim2039/EightK/data/p/abn_ev_monly.p /scratch/nz97/ad4734/EightK/data/p/



scp -r ~/.ssh/ ./data/cosine/* adidishe@spartan.hpc.unimelb.edu.au:/data/gpfs/projects/punim2039/EightK//data/cosine/
scp -r ~/.ssh/ ./data/cosine_final/* adidishe@spartan.hpc.unimelb.edu.au:/data/gpfs/projects/punim2039/EightK//data/cosine_final/
scp -r ~/.ssh/ adidishe@spartan.hpc.unimelb.edu.au:/data/gpfs/projects/punim2039/EightK/data/training/OPT_13b/EIGHT_LEGAL_ATI_TRAIN/*  ./data/training/OPT_13b/EIGHT_LEGAL_ATI_TRAIN/

scp -r ~/.ssh/ ./res/temp/vec_pred/* adidishe@spartan.hpc.unimelb.edu.au:/data/gpfs/projects/punim2039/EightK/res/temp/vec_pred/




scp -r ~/.ssh/ ./data/p/load_control_coverage.p adidishe@spartan.hpc.unimelb.edu.au:/data/gpfs/projects/punim2039/EightK//data/p/
scp -r ~/.ssh/ ./data/p/load_bryan_data.p adidishe@spartan.hpc.unimelb.edu.au:/data/gpfs/projects/punim2039/EightK//data/p/
scp -r ~/.ssh/ ./data/p/rel_max.p adidishe@spartan.hpc.unimelb.edu.au:/data/gpfs/projects/punim2039/EightK//data/p/


scp -r ~/.ssh/  adidishe@spartan.hpc.unimelb.edu.au:/data/gpfs/projects/punim2039/EightK/data/p/rel_max.p /scratch/nz97/ad4734/EightK/data/p/
scp -r ~/.ssh/  adidishe@spartan.hpc.unimelb.edu.au:/data/gpfs/projects/punim2039/EightK/data/p/load_bryan_data.p /scratch/nz97/ad4734/EightK/data/p/
scp -r ~/.ssh/  adidishe@spartan.hpc.unimelb.edu.au:/data/gpfs/projects/punim2039/EightK/data/p/load_control_coverage.p /scratch/nz97/ad4734/EightK/data/p/

scp -r ~/.ssh/  adidishe@spartan.hpc.unimelb.edu.au:/data/gpfs/projects/punim2039/EightK/data/p/load_icf_ati_filter.p /mnt/layline/project/eightk/data/p


# nohup scp -r ~/.ssh/  adidishe@spartan.hpc.unimelb.edu.au:/data/gpfs/projects/punim2039/EightK/data/p/*.p /scratch/nz97/ad4734/EightK/da &