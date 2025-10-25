import matplotlib.pyplot as plt

# Sample data
k_values = [1, 5, 8, 12, 15,20]

avg_feedbacks_adv_spoof = [17.78, 4.10, 2.85, 2.21, 1.90, 1.55]
asr = [0.96,0.94,0.92,0.89,0.87,0.84]
# Average injections, modifications for Adv-Spoof
avg_modifications    = [16.17, 17.58, 18.14, 18.40, 18.94, 18.64]
avg_injections = [1.61,  2.78,  4.32,  7.51,  8.50, 10.87]
total_perturbations = [i + m for i, m in zip(avg_injections, avg_modifications)]

# Create figure
plt.figure(figsize=(15, 4))

# 1️⃣ Average Feedback vs k
plt.subplot(1, 3, 1)
plt.plot(k_values, avg_feedbacks_adv_spoof, marker='o', color='blue')
plt.title('Average Feedback vs k')
plt.xlabel('k')
plt.ylabel('Average Feedback')
plt.grid(True)

# 2️⃣ ASR vs k
plt.subplot(1, 3, 2)
plt.plot(k_values, asr, marker='s', color='green')
plt.title('Attack Success Rate (ASR) vs k')
plt.xlabel('k')
plt.ylabel('ASR')
plt.ylim(0.8, 1.0)
plt.grid(True)

# 3️⃣ Perturbations vs k
plt.subplot(1, 3, 3)
plt.plot(k_values, avg_injections, marker='o', linestyle='-', label='Avg Injections')
plt.plot(k_values, avg_modifications, marker='s', linestyle='--', label='Avg Modifications')
plt.plot(k_values, total_perturbations, marker='^', linestyle='-.', label='Total Perturbations')
plt.title('Perturbations vs k')
plt.xlabel('k')
plt.ylabel('Average Count')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("top_k_vs_feedbacks_asr_perturbations.png", dpi=300, bbox_inches='tight')
# plt.show()
