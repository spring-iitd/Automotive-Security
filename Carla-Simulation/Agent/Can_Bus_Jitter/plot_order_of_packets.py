import matplotlib.pyplot as plt

# Read the CAN log
log_file = './Logs/arbitrated_can_logs.log'

# Set 8 distinct colors
colors = [
    '#FF0000',  # Red
    '#33FF57',  # Lime-Green
    '#3357FF',  # Blue
    '#F333FF',  # Pink
    '#33FFF3',  # Cyan
    '#F3FF33',  # Yellow
    '#A833FF',  # Purple
    '#964B00'   # Brown
]

id_color_map = {}
unique_ids = []
timestamps = []
ids_sequence = []

with open(log_file, 'r') as f:
    for line in f:
        if line.strip():
            parts = line.strip().split()
            timestamp = float(parts[0].strip('()'))
            
            # Only include data between 0 and 1 seconds
            if 0 <= timestamp <= 3:
                can_id = parts[2]

                if can_id not in id_color_map:
                    unique_ids.append(can_id)
                    id_color_map[can_id] = colors[len(unique_ids) - 1]

                timestamps.append(timestamp)
                ids_sequence.append(can_id)

# Group the sequence into chunks of 8
chunks = [ids_sequence[i:i + 8] for i in range(0, len(ids_sequence), 8)]
chunk_timestamps = [timestamps[i] for i in range(0, len(timestamps), 8)]

fig, ax = plt.subplots(figsize=(15, 40))

for idx, (chunk, y) in enumerate(zip(chunks, chunk_timestamps)):
    for x, can_id in enumerate(chunk):
        color = id_color_map.get(can_id, 'black')
        ax.add_patch(plt.Rectangle((x, y), 1, 0.001, color=color))

# Adjust the plot
ax.set_xlim(0, 8)
ax.set_xlabel('Packet Position (0 to 7)')
ax.set_ylabel('Timestamp (s)')
ax.set_title('CAN Packets Timeline by IDs')
ax.set_yticks(chunk_timestamps)
ax.invert_yaxis()
ax.grid(True, axis='x')

# Create a custom legend
legend_handles = [plt.Rectangle((0,0),1,1, color=color) for color in colors]
ax.legend(legend_handles, unique_ids, title='CAN IDs', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig('./Graphs/plot_can_packets_timeline.png', dpi=300)
