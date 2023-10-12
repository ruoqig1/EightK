#!/bin/bash

# Make the script executable
chmod +x $0
# Define the upper limit as a variable 0 18 for full
lower_limit=9
upper_limit=9
# Loop from lower_limit to upper_limit
for i in $(seq $lower_limit $upper_limit); do
  # Define a temporary PBS script name
  TEMP_PBS="temp_script_vec.pbs"

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
#PBS -l walltime=12:00:00
#PBS -l storage=scratch/nz97
#PBS -l mem=64GB
#PBS -l ncpus=4
#PBS -M antoine.didisheim@unimelb.edu.au
#PBS -m ${email_flag}
#PBS -N vec_press_${i}
#PBS -o out/vec_press_${i}.out
#PBS -e out/vec_press_${i}.err

cd /scratch/nz97/ad4734/EightK/
module load python3/3.10.4
source ~/scratch/nz97/ad4734/venv/bin/activate
python vec_main.py ${i} --legal=0 --eight=1 --bow=1
EOL

  # Submit the script
  qsub $TEMP_PBS

  # Sleep for a moment if needed to avoid overwhelming the job scheduler
  # sleep 1
done
