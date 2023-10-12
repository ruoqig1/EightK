#!/bin/bash

# Make the script executable
chmod +x $0

# Define the upper limit as a variable
lower_limit=0
upper_limit=65
# Loop from 0 to 134
for i in $(seq $lower_limit $upper_limit); do
    echo $i
done
