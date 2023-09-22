#!/usr/bin/env bash
scp *.py ad4734@gadi.nci.org.au:/scratch/nz97/ad4734/EightK/
scp utils_local/*.py ad4734@gadi.nci.org.au:/scratch/nz97/ad4734/EightK/utils_local/
scp clean/*.py ad4734@gadi.nci.org.au:/scratch/nz97/ad4734/EightK/
scp scripts/slurm/* ad4734@gadi.nci.org.au:/scratch/nz97/ad4734/EightK/
scp summary_stat/* ad4734@gadi.nci.org.au:/scratch/nz97/ad4734/EightK/


# Print the current time
echo "Pushed at $(date +'%H:%M')"


