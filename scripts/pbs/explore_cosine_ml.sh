#!/bin/bash

# Make the script executable
chmod +x $0

# Define the upper limit as a variable
lower_limit=0
upper_limit=0
# Loop from 0 to 134
for i in $(seq $lower_limit $upper_limit); do

  # Define a temporary PBS script name
  TEMP_PBS="temp_script.pbs"

  # Initialize the email flag to 'n'
  email_flag="n"

  # Generate the PBS script
  if [ $i -eq $lower_limit ]; then
    # First iteration, send an email when it starts
    email_flag="b"
  elif [ $i -eq $upper_limit ]; then
    # Last iteration, send an email when it ends
    email_flag="e"
  fi

  # Overwrite the content of the existing temporary script
  cat <<EOL > $TEMP_PBS
#!/bin/bash
#PBS -P nz97
#PBS -q normal
#PBS -l walltime=24:00:00
#PBS -l storage=scratch/nz97
#PBS -l mem=32GB
#PBS -l ncpus=32
#PBS -M antoine.didisheim@unimelb.edu.au
#PBS -m ${email_flag}
#PBS -N explore_cosine_ml_${i}
#PBS -o out/explore_cosine_ml_${i}.out
#PBS -e out/explore_cosine_ml_${i}.err

cd /scratch/nz97/ad4734/EightK/
module load python3/3.10.4
source ~/scratch/nz97/ad4734/venv/bin/activate
python explore_cosine_ml.py ${i}
EOL

  # Submit the script
  qsub $TEMP_PBS

  # Sleep for a moment if needed to avoid overwhelming the job scheduler
  sleep 2
done
