import os
import json
import matplotlib.pyplot as plt

# Define paths
plots_folder = "./plots/scaling_ablate"
json_file = "./results_ablate/scaling_ablate.json"

# Check if the directory exists; if not, create it.
if not os.path.exists(plots_folder):
    os.makedirs(plots_folder)

# Load the JSON data.
with open(json_file, 'r') as f:
    data = json.load(f)

# Extract base model metrics (only using the mean for the horizontal line)
base_metrics = data["base_model"]
base_accuracy_mean = base_metrics["accuracy"]["mean"]
base_auc_mean = base_metrics["auc"]["mean"]
base_f1_mean = base_metrics["f1"]["mean"]

# Define model numbers 1 through 8
model_nums = list(range(1, 9))

# Prepare lists for mean and std for each metric for naive and ReFine models.
naive_accuracy_mean, naive_accuracy_std = [], []
naive_auc_mean, naive_auc_std = [], []
naive_f1_mean, naive_f1_std = [], []

enhanced_accuracy_mean, enhanced_accuracy_std = [], []
enhanced_auc_mean, enhanced_auc_std = [], []
enhanced_f1_mean, enhanced_f1_std = [], []

# Loop over model numbers (assuming keys are like 'naive1', 'naive2', ..., 'naive8')
for i in model_nums:
    naive_model = data["naive_models"].get(f"naive{i}", None)
    enhanced_model = data["enhanced_models"].get(f"enhanced{i}", None)
    
    if naive_model:
        # Extract mean and std for each metric from the naive model.
        naive_accuracy_mean.append(naive_model["accuracy"]["mean"])
        naive_accuracy_std.append(naive_model["accuracy"]["std"])
        naive_auc_mean.append(naive_model["auc"]["mean"])
        naive_auc_std.append(naive_model["auc"]["std"])
        naive_f1_mean.append(naive_model["f1"]["mean"])
        naive_f1_std.append(naive_model["f1"]["std"])
    else:
        naive_accuracy_mean.append(None)
        naive_accuracy_std.append(None)
        naive_auc_mean.append(None)
        naive_auc_std.append(None)
        naive_f1_mean.append(None)
        naive_f1_std.append(None)
    
    if enhanced_model:
        # Extract mean and std for each metric from the enhanced model.
        enhanced_accuracy_mean.append(enhanced_model["accuracy"]["mean"])
        enhanced_accuracy_std.append(enhanced_model["accuracy"]["std"])
        enhanced_auc_mean.append(enhanced_model["auc"]["mean"])
        enhanced_auc_std.append(enhanced_model["auc"]["std"])
        enhanced_f1_mean.append(enhanced_model["f1"]["mean"])
        enhanced_f1_std.append(enhanced_model["f1"]["std"])
    else:
        enhanced_accuracy_mean.append(None)
        enhanced_accuracy_std.append(None)
        enhanced_auc_mean.append(None)
        enhanced_auc_std.append(None)
        enhanced_f1_mean.append(None)
        enhanced_f1_std.append(None)

def plot_metric(metric_name, naive_means, naive_stds, enhanced_means, enhanced_stds, base_mean, filename):
    plt.figure(figsize=(8, 6))

    # Plot error bars
    plt.errorbar(model_nums, naive_means, yerr=naive_stds, marker='o', linestyle='-', capsize=5, label='Naive')
    plt.errorbar(model_nums, enhanced_means, yerr=enhanced_stds, marker='s', linestyle='-', capsize=5, label='ReFine')

    # Baseline
    plt.axhline(y=base_mean, color='r', linestyle='--', label='NoTrans')

    # Axis labels
    plt.xlabel("Model Number (1-8)", fontsize=14)
    plt.ylabel(metric_name.capitalize(), fontsize=14)

    # Tick label font sizes
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # Legend
    plt.legend(fontsize=13)
    plt.grid(True)

    # Save the plot
    plt.savefig(os.path.join(plots_folder, filename))
    plt.close()


# Plot and save for each metric.
plot_metric("accuracy", naive_accuracy_mean, naive_accuracy_std, enhanced_accuracy_mean, enhanced_accuracy_std, base_accuracy_mean, "scaling_ablate_accuracy.png")
plot_metric("auc", naive_auc_mean, naive_auc_std, enhanced_auc_mean, enhanced_auc_std, base_auc_mean, "scaling_ablate_auc.png")
plot_metric("f1", naive_f1_mean, naive_f1_std, enhanced_f1_mean, enhanced_f1_std, base_f1_mean, "scaling_ablate_f1.png")

print("Plots generated and saved in", plots_folder)
