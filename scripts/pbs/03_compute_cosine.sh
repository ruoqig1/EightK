#!/bin/bash

# Make the script executable
chmod +x $0

# Define the upper limit as a variable
lower_limit=0
upper_limit=19
# Loop from 0 to 134
for i in $(seq $lower_limit $upper_limit); do

  # Define a temporary PBS script name
  TEMP_PBS="temp_script.pbs"
  echo $i

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
#PBS -q hugemem
#PBS -l walltime=12:00:00
#PBS -l storage=scratch/nz97
#PBS -l mem=380GB
#PBS -l ncpus=64
#PBS -M antoine.didisheim@unimelb.edu.au
#PBS -m ${email_flag}
#PBS -N hugemem${i}
#PBS -o out/hugemem${i}.out
#PBS -e out/hugemem${i}.err

cd /scratch/nz97/ad4734/EightK/
module load python3/3.10.4
source ~/scratch/nz97/ad4734/venv/bin/activate
python 03_compute_cosine.py ${i}
EOL
  # Submit the script
  qsub $TEMP_PBS
  # Sleep for a moment if needed to avoid overwhelming the job scheduler
  sleep 3
done

