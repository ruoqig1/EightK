#!/usr/bin/env bash
#echo "$1G"
qsub -I -l ncpus=4,mem="$1GB",walltime=01:00:00


