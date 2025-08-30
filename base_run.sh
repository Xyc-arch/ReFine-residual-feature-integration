#!/bin/bash

# ============================================================================
# base_run.sh
# Run all base_* experiments in more_baselines/ sequentially
# Logs everything to base_output.txt
# ============================================================================

# Define log file
LOG_FILE="base_output.txt"

# Define working directory (this script is placed in /home/ubuntu/refine)
WORKING_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Change to the working directory
cd "$WORKING_DIR" || { echo "Error: Cannot change to directory $WORKING_DIR"; exit 1; }

# Function to run a script sequentially
run_sequential() {
    local script_name=$1
    echo "Starting $script_name..." | tee -a "$LOG_FILE"
    python -u "$script_name" >> "$LOG_FILE" 2>&1
    exit_status=$?

    if [ $exit_status -ne 0 ]; then
        echo "$script_name failed with exit status $exit_status." | tee -a "$LOG_FILE"
        exit $exit_status
    else
        echo "$script_name completed successfully." | tee -a "$LOG_FILE"
    fi
}

# Clear previous log file
> "$LOG_FILE"

# Define an array of Python scripts to run sequentially
PYTHON_SCRIPTS=(

    ## stress test with CNN on CIFAR-10 and CIFAR-100:
    # more_baselines/base_noise.py
    more_baselines/base_adversarial.py
    more_baselines/base_mismatch.py
    more_baselines/base_noise_test100.py
    more_baselines/base_adversarial_test100.py
    more_baselines/base_imb_test100.py
    more_baselines/base_mismatch_test100.py

    ## optional:
    # more_baselines/base_match_digit.py

)

# Iterate over the array and run each script sequentially
for script in "${PYTHON_SCRIPTS[@]}"; do
    run_sequential "$script"
done

echo "All base_* scripts have been executed sequentially. Check $LOG_FILE for details."
