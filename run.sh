#!/bin/bash

# Define log file
LOG_FILE="output.txt"

# Define working directory
WORKING_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Change to the working directory
cd "$WORKING_DIR" || { echo "Error: Cannot change to directory $WORKING_DIR"; exit 1; }

# Function to run a script sequentially
run_sequential() {
    local script_name=$1
    echo "Starting $script_name..." | tee -a "$LOG_FILE"
    # Run Python script in unbuffered mode
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
    "noise.py"
    "adversarial.py"
    "imb.py"
    "mismatch.py"
    "noise_test100.py"
    "adversarial_test100.py"
    "imb_test100.py"
    "mismatch_test100.py"

    ## stress test with Transformer on CIFAR-10 and CIFAR-100:
    # "noise_tf.py"
    # "mismatch_tf.py"
    # "imb_tf.py"
    # "adversarial_tf.py"
    # "mismatch_tf_test100.py"
    # "imb_tf_test100.py"
    # "adversarial_tf_test100.py"
    # "noise_tf_test100.py"

    ## multi-source scaling experiments:
    # "scaling_hard.py"
    # "scaling_oracle.py"
    # "scaling_low_lr.py"
    # "scaling.py"

    ## cross domain transfer tasks
    # "match_stl.py"
    # "match_digit.py"
    # "match_domainnet.py"
    # "match_text.py"
    # "match_text_elec.py"

    ## ablation of parameter count in adapter implementation
    # "ablate_text.py"
)

# Iterate over the array and run each script sequentially
for script in "${PYTHON_SCRIPTS[@]}"; do
    run_sequential "$script"
done

echo "All scripts have been executed sequentially. Check $LOG_FILE for details."
