



# import pandas as pd
# import matplotlib.pyplot as plt

# # Read your CSV
# df = pd.read_csv("blackbox_gear/perturbation_summary.csv")
# df.columns = df.columns.str.strip()

# # Remove entries where model_feedback == 1
# df = df[df["model_feedback"] != 1]

# # --- Adjust feedback (subtract 1) ---
# df["feedback"] = df["model_feedback"] - 1

# # Count frequency of adjusted feedback
# feedback_counts = df["feedback"].value_counts().sort_index()

# # --- Compute statistics ---
# mean_val = df["feedback"].mean()
# median_val = df["feedback"].median()
# mode_val = df["feedback"].mode()[0]   # mode() can return multiple values, take first

# # --- Plot histogram for adjusted feedback ---
# plt.figure(figsize=(12,5))
# plt.bar(feedback_counts.index, feedback_counts.values, color='skyblue', edgecolor='black')

# # Labels & title
# plt.xlabel("Model Feedback", fontsize=12)
# plt.ylabel("Number of Images", fontsize=12)
# plt.title("Bitflip Attack on dos images, k=12)", fontsize=14)

# # Show values on top of bars
# for x, y in zip(feedback_counts.index, feedback_counts.values):
#     plt.text(x, y+0.2, str(y), ha='center', fontsize=10)

# # Add mean, median, mode annotations
# plt.text(mean_val+3, plt.ylim()[1]*0.9, f"Mean = {mean_val:.2f}", color='red')
# plt.text(mean_val+3, plt.ylim()[1]*1.0, f"Median = {median_val}", color='green')
# plt.text(mean_val+3, plt.ylim()[1]*1.1, f"Mode = {mode_val}", color='purple')

# plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.xlim(0, 10)    # adjust as needed
# plt.ylim(0, 9500)  # adjust as needed
# # plt.show()
# plt.savefig("performance_plot_gear.png", dpi=300, bbox_inches='tight')
# # --- Second Graph: Total Perturbations for a range of images ---

# # Extract numeric part from image name (assuming column is "image_name")
# df["image_num"] = df["image_name"].str.extract(r'(\d+)').astype(int)

# # Compute total perturbations
# df["total_perturbations"] = df["injection_count"] + df["modification_count"]

# # Filter only images between 52704 and 70500
# df_range = df[(df["image_num"] >= 52704) & (df["image_num"] <= 70500)]

# # --- Compute averages ---
# avg_injections = df_range["injection_count"].mean()
# avg_modifications = df_range["modification_count"].mean()
# avg_total = df_range["total_perturbations"].mean()

# print(f"Average Injections: {avg_injections:.2f}")
# print(f"Average Modifications: {avg_modifications:.2f}")
# print(f"Average Total Perturbations: {avg_total:.2f}")

# plot_histogram.py
# plot_histogram.py
import pandas as pd
import matplotlib.pyplot as plt

def plot_feedback_histogram(csv_path="blackbox_gear_k20/perturbation_summary.csv",
                            out_plot="performance_plot_gear_k20.png",
                            out_stats="blackbox_gear_k20/stats.txt",
                            k_label="k=20",
                            show=False):
    # --- Load and clean data ---
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    # Remove entries where model_feedback == 1
    df = df[df["model_feedback"] != 1].copy()

    # Adjust feedback (subtract 1)
    df["feedback"] = df["model_feedback"] - 1

    # --- Compute feedback statistics ---
    mean_val = df["feedback"].mean()
    median_val = df["feedback"].median()
    mode_val = df["feedback"].mode()[0]

    # --- Compute total perturbations ---
    df["total_perturbations"] = df["injection_count"] + df["modification_count"]
    avg_injections = df["injection_count"].mean()
    avg_modifications = df["modification_count"].mean()
    avg_total = df["total_perturbations"].mean()

    # --- Save statistics to a text file ---
    with open(out_stats, "a") as f:
        f.write("=== Feedback Statistics ===\n")
        f.write(f"Mean Feedback: {mean_val:.2f}\n")
        f.write(f"Median Feedback: {median_val}\n")
        f.write(f"Mode Feedback: {mode_val}\n\n")

        f.write("=== Perturbation Statistics ===\n")
        f.write(f"Average Injections: {avg_injections:.2f}\n")
        f.write(f"Average Modifications: {avg_modifications:.2f}\n")
        f.write(f"Average Total Perturbations: {avg_total:.2f}\n")

    # print(f"Statistics written to {out_stats}")

    # --- Plot histogram ---
    feedback_counts = df["feedback"].value_counts().sort_index()
    plt.figure(figsize=(12, 5))
    plt.bar(feedback_counts.index, feedback_counts.values, color='skyblue', edgecolor='black')

    plt.xlabel("Model Feedback", fontsize=12)
    plt.ylabel("Number of Images", fontsize=12)
    plt.title(f"Bitflip Attack on DOS Images ({k_label})", fontsize=14)

    # Values on top of bars
    for x, y in zip(feedback_counts.index, feedback_counts.values):
        plt.text(x, y + 0.2, str(y), ha='center', fontsize=10)

    # Annotations
    ymax = plt.ylim()[1]
    plt.text(mean_val + 5.6, ymax * 0.9, f"Mean = {mean_val:.2f}", color='red')
    plt.text(mean_val + 5.6, ymax * 0.83, f"Median = {median_val}", color='green')
    plt.text(mean_val + 5.6, ymax * 0.76, f"Mode = {mode_val}", color='purple')

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xlim(0, 20)
    plt.ylim(0, ymax * 1.1)

    if show:
        plt.show()

    plt.savefig(out_plot, dpi=300, bbox_inches='tight')
    plt.close()
    # print(f"Histogram saved to {out_plot}")


if __name__ == "__main__":
    plot_feedback_histogram()

# plt.tight_layout()
# plt.show()

# plt.savefig("Total_perturbations_rgb_k=17.png", dpi=300, bbox_inches='tight')
