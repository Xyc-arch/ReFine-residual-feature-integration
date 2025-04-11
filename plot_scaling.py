import os
import json
import matplotlib.pyplot as plt
import numpy as np

# Prepare the plots directory.
os.makedirs("./plots/scaling", exist_ok=True)

# Load the scaling results from the JSON file.
with open("./results_test10/scaling.json", "r") as f:
    data = json.load(f)

# Sort the keys (enhanced models) in order.
def sort_key(key):
    # Extract the integer index from keys like "enhanced0", "enhanced1", etc.
    return int(key.replace("enhanced", ""))

sorted_keys = sorted(data.keys(), key=sort_key)

# X-axis: external model numbers.
x = [int(key.replace("enhanced", "")) for key in sorted_keys]

# Extract metrics in sorted order.
accuracy_mean = [data[key]["accuracy_mean"] for key in sorted_keys]
accuracy_std  = [data[key]["accuracy_std"] for key in sorted_keys]
auc_mean      = [data[key]["auc_mean"] for key in sorted_keys]
auc_std       = [data[key]["auc_std"] for key in sorted_keys]
f1_mean       = [data[key]["f1_mean"] for key in sorted_keys]
f1_std        = [data[key]["f1_std"] for key in sorted_keys]
inference_time_mean = [data[key]["inference_time_mean"] for key in sorted_keys]
inference_time_std  = [data[key]["inference_time_std"] for key in sorted_keys]

# Plot function: creates an errorbar plot and saves it.
def plot_metric(x, y_mean, y_std, title, ylabel, filename):
    plt.figure(figsize=(8, 5))
    plt.errorbar(x, y_mean, yerr=y_std, fmt='o-', capsize=5)
    plt.title(title)
    plt.xlabel("External Model Number")
    plt.ylabel(ylabel)
    plt.xticks(x)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"plots/scaling/{filename}")
    plt.close()

# Create and save plots.
plot_metric(x, accuracy_mean, accuracy_std,
            "Model Accuracy vs. External Model Number", "Accuracy", "scaling_accuracy.png")

plot_metric(x, auc_mean, auc_std,
            "Model AUC vs. External Model Number", "AUC", "scaling_auc.png")

plot_metric(x, f1_mean, f1_std,
            "Model F1 Score vs. External Model Number", "F1 Score", "scaling_f1.png")

plot_metric(x, inference_time_mean, inference_time_std,
            "Inference Time vs. External Model Number", "Time (s)", "scaling_inference_time.png")

print("Plots saved in the 'plots' directory.")
