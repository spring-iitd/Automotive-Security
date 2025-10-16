

import pandas as pd
import matplotlib.pyplot as plt

# Read your CSV
df = pd.read_csv("blackbox_spoof_mod_then_inject/perturbation_summary.csv")
df.columns = df.columns.str.strip()

# Remove entries where model_feedback == 1
df = df[df["model_feedback"] != 1]

# --- Adjust feedback (subtract 1) ---
df["adjusted_feedback"] = df["model_feedback"] - 1

# Count frequency of adjusted feedback
feedback_counts = df["adjusted_feedback"].value_counts().sort_index()

# --- Compute statistics ---
mean_val = df["adjusted_feedback"].mean()
median_val = df["adjusted_feedback"].median()
mode_val = df["adjusted_feedback"].mode()[0]   # mode() can return multiple values, take first

# --- Plot histogram for adjusted feedback ---
plt.figure(figsize=(12,5))
plt.bar(feedback_counts.index, feedback_counts.values, color='skyblue', edgecolor='black')

# Labels & title
plt.xlabel("Model Feedback (Feedback - 1)", fontsize=12)
plt.ylabel("Number of Images", fontsize=12)
plt.title("Bitflip Attack on spoofed images only, k=12)", fontsize=14)

# Show values on top of bars
for x, y in zip(feedback_counts.index, feedback_counts.values):
    plt.text(x, y+0.2, str(y), ha='center', fontsize=10)

# Add mean, median, mode annotations
plt.text(mean_val+3, plt.ylim()[1]*0.9, f"Mean = {mean_val:.2f}", color='red')
plt.text(mean_val+3, plt.ylim()[1]*1.0, f"Median = {median_val}", color='green')
plt.text(mean_val+3, plt.ylim()[1]*1.1, f"Mode = {mode_val}", color='purple')

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xlim(0, 10)    # adjust as needed
plt.ylim(0, 3000)  # adjust as needed
# plt.show()
plt.savefig("performance_plot_spoof.png", dpi=300, bbox_inches='tight')
# --- Second Graph: Total Perturbations for a range of images ---

# Extract numeric part from image name (assuming column is "image_name")
df["image_num"] = df["image_name"].str.extract(r'(\d+)').astype(int)

# Compute total perturbations
df["total_perturbations"] = df["injection_count"] + df["modification_count"]

# Filter only images between 52704 and 70500
df_range = df[(df["image_num"] >= 52704) & (df["image_num"] <= 70500)]

# --- Compute averages ---
avg_injections = df_range["injection_count"].mean()
avg_modifications = df_range["modification_count"].mean()
avg_total = df_range["total_perturbations"].mean()

print(f"Average Injections: {avg_injections:.2f}")
print(f"Average Modifications: {avg_modifications:.2f}")
print(f"Average Total Perturbations: {avg_total:.2f}")

# plt.figure(figsize=(14,6))
# plt.bar(df_range["image_num"], df_range["total_perturbations"], 
#         color='orange', edgecolor='black')

# plt.xlabel("Image No.", fontsize=12)
# plt.ylabel("Total Perturbations (Injections + Modifications)", fontsize=12)
# plt.title("Total Perturbations per Image (52704–70500)", fontsize=14)

# plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.tight_layout()
# plt.show()

# plt.savefig("Total_perturbations_rgb_k=17.png", dpi=300, bbox_inches='tight')