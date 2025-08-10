import pandas as pd
import matplotlib.pyplot as plt

# Read your CSV
df = pd.read_csv("blackbox_ch_final_rgb_gradient/perturbation_summary.csv")
df.columns = df.columns.str.strip()

# Remove entries where model_feedback == 1
df = df[df["model_feedback"] != 1]

# Count frequency of each model_feedback value
feedback_counts = df["model_feedback"].value_counts().sort_index()

# Plot histogram
plt.figure(figsize=(12,5))
plt.bar(feedback_counts.index, feedback_counts.values, color='skyblue', edgecolor='black')

# Labels & title
plt.xlabel("Model Feedback", fontsize=12)
plt.ylabel("Number of Images", fontsize=12)
plt.title("Histogram of Model Feedback (Excluding Feedback = 1)", fontsize=14)

# Show values on top of bars
for x, y in zip(feedback_counts.index, feedback_counts.values):
    plt.text(x, y+0.2, str(y), ha='center', fontsize=10)

plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.show()
plt.xlim(0, 75)   # X-axis: feedback values range
plt.ylim(0, 1800)  # Y-axis: number of images range

plt.savefig("model_feedback_histogram_rgb_gradient.png", dpi=300, bbox_inches='tight')