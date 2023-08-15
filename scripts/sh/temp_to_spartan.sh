#!/usr/bin/env bash
scp data/wsj_tar/* adidishe@spartan.hpc.unimelb.edu.au:/home/adidishe/EightK/data/wsj_tar/

echo "code pushed:"
date | cut -d " " -f5 | cut -d ":" -f1-2