#!/bin/bash
# Check if an argument is provided
if [ $# -eq 0 ]; then
    # Default value if no argument is provided
    SCRIPT=train.sh
else
    # Use the provided argument
    SCRIPT=$1
fi
submit_batch_job() {
    echo "Submitting ${SCRIPT}..."
    sbatch ${SCRIPT}  
    # sbatch strain_multi.sh
}

while true; do
    submit_batch_job
    sleep 4h 5m
done