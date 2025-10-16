import matplotlib.pyplot as plt

# Sample data
k_values = [1,5, 8, 10, 12, 15, 17,20]

# Average feedback for each k
avg_feedbacks = [22.86, 4.98, 3.29, 2.75, 2.32, 2.02,1.78,1.59]

# Average injections, modifications, total perturbations for each k
avg_injections = [19.86, 21.07,21.83,22.33,22.55,23.63,23.8,24.5]
avg_modifications = [3.0,3.86,4.49,5.13,5.24,6.53,6.31,7.14]
total_perturbations = [i + m for i, m in zip(avg_injections, avg_modifications)]

# Plot 1: Average Feedback vs k
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(k_values, avg_feedbacks, marker='o', linestyle='-', color='blue')
plt.title('Average Feedback vs k')
plt.xlabel('k')
plt.ylabel('Average Feedback')
plt.grid(True)

# Plot 2: Avg injections, modifications, and total perturbations vs k
plt.subplot(1, 2, 2)
plt.plot(k_values, avg_injections, marker='o', linestyle='-', label='Avg Injections')
plt.plot(k_values, avg_modifications, marker='s', linestyle='--', label='Avg Modifications')
plt.plot(k_values, total_perturbations, marker='^', linestyle='-.', label='Total Perturbations')

plt.title('Perturbations vs k')
plt.xlabel('k')
plt.ylabel('Average Count')
plt.legend()
plt.grid(True)

plt.tight_layout()
# plt.show()
plt.savefig("top_k Vs Feedbacks and PErturbations.png", dpi=300, bbox_inches='tight')
