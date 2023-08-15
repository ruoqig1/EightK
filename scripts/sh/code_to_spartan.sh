#!/usr/bin/env bash
scp *.py adidishe@spartan.hpc.unimelb.edu.au:/home/adidishe/EightK/
scp utils/*.py adidishe@spartan.hpc.unimelb.edu.au:/home/adidishe/EightK/utils/
scp scripts/slurm/* adidishe@spartan.hpc.unimelb.edu.au:/home/adidishe/EightK/
scp summary_stats/* adidishe@spartan.hpc.unimelb.edu.au:/home/adidishe/EightK/
scp create_rf/* adidishe@spartan.hpc.unimelb.edu.au:/home/adidishe/EightK/create_rf/


#scp data/raw/crsp_monthly.csv adidishe@spartan.hpc.unimelb.edu.au:/data/gpfs/projects/punim2039/EightK/data/raw/

# Print the current time
echo "Pushed at $(date +'%H:%M')"