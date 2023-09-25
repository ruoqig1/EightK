#!/bin/bash

# Make the script executable
chmod +x $0

# Loop from 0 to 134
for i in $(seq 0 90); do

  # Define a temporary PBS script name
  TEMP_PBS="temp_script.pbs"

  # Initialize the email flag to 'n'
  email_flag="n"

  # Generate the PBS script
  if [ $i -eq 0 ]; then
    # First iteration, send an email when it starts
    email_flag="b"
  elif [ $i -eq 14 ]; then
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
#PBS -l mem=95GB
#PBS -l ncpus=4
#PBS -M antoine.didisheim@unimelb.edu.au
#PBS -m ${email_flag}
#PBS -N train_main_${i}
#PBS -o out/train_main_${i}.out
#PBS -e out/train_main_${i}.err

cd /scratch/nz97/ad4734/EightK/
module load python3/3.10.4
source ~/scratch/nz97/ad4734/venv/bin/activate
python train_main.py ${i}
EOL

  # Submit the script
  qsub $TEMP_PBS

  # Sleep for a moment if needed to avoid overwhelming the job scheduler
  sleep 5
done
