# import pandas as pd

# # Load your CSV
# df = pd.read_csv("DoS_car_hacking_wo_stuff.csv")

# # Count how many frames each image has
# frame_counts = df.groupby(" image_no").size()

# # Compute statistics
# min_frames = frame_counts.min()
# max_frames = frame_counts.max()
# mean_frames = frame_counts.mean()

# print(f"Minimum frames in an image: {min_frames}")
# print(f"Maximum frames in an image: {max_frames}")


# only for attack imagse
import pandas as pd

# Load CSV
df = pd.read_csv("DoS_car_hacking_wo_stuff.csv")

# Clean column names (strip spaces)
df.columns = df.columns.str.strip()

# Make sure image_no is integer
df["image_no"] = df["image_no"].astype(int)

# Filter only attack images (label == 1)
df_attack = df[df["label"] == 1]

# Count frames per attack image
frame_counts = df_attack.groupby("image_no").size()

# Compute stats
min_frames = frame_counts.min()
max_frames = frame_counts.max()
mean_frames = frame_counts.mean()

print(f"Minimum frames in an attack image: {min_frames}")
print(f"Maximum frames in an attack image: {max_frames}")
print(f"Mean frames in an attack image: {mean_frames:.2f}")

