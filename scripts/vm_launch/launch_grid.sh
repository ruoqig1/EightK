#!/bin/bash

# Create the output directory if it doesn't exist
mkdir -p out

# Maximum number of simultaneous jobs
k=8

# Initialize a counter for running jobs
running_jobs=0

# Loop through the 20 jobs you want to launch
for i in {1..20}; do
  # Check the number of running jobs
  running_jobs=$(jobs -rp | wc -l)

  # Wait for the running jobs to drop below k
  while [ "$running_jobs" -ge "$k" ]; do
    sleep 1
    running_jobs=$(jobs -rp | wc -l)
  done
  # activate vm nlp bin
  source venv/bin/activate

  # Run the Python script and send it to background
  # Redirect stdout and stderr to files in the 'out' directory
  python clean_eightk.py $i > "out/output_$i.txt" 2> "out/error_$i.txt" &

done

# Wait for all background jobs to finish
wait
