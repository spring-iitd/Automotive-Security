
import re
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix

model = joblib.load("carla_sgd_model.joblib")
scaler = joblib.load("train_scaler_carla.joblib")

def parse_log_line(line):
    match = re.match(r"\(([\d\.]+)\)\s+\S+\s+([0-9A-Fa-f]+)\s+\[(\d+)\]\s+((?:[0-9A-Fa-f]{2} ?)+)", line.strip())
    if not match:
        return None
    timestamp = float(match.group(1))
    can_id = match.group(2).lower().zfill(4)
    dlc = int(match.group(3))
    data_bytes = [int(byte, 16) for byte in match.group(4).strip().split()]
    data_bytes += [0] * (8 - len(data_bytes))  # pad to 8 bytes
    label = 1 if set(can_id) <= {'0'} else 0
    return timestamp, [int(can_id, 16), dlc] + data_bytes[:8], label

Y_predicted = []
Y_true = []

prev_attack_index = 0

with open("./Logs/can_data_logs.log", 'r') as file:
    sample_num = 1
    prev_timestamp = None

    while sample_num <= 16185:
        line = file.readline()
        parsed = parse_log_line(line)
        if not parsed:
            continue

        timestamp, features, label = parsed

        if prev_timestamp is None:
            iat = 0.0
        else:
            iat = timestamp - prev_timestamp
        prev_timestamp = timestamp

        features.append(iat)

        X_single = np.array(features).reshape(1, -1)
        Y_single = np.array([label])
        X_scaled = scaler.transform(X_single)

        Y_pred = model.predict(X_scaled)[0]

        Y_true.append(label)
        Y_predicted.append(Y_pred)

        # acc = accuracy_score(Y_single, Y_pred)

        if Y_pred == 1:
            if prev_attack_index + 1 != sample_num:
                print(f"From {prev_attack_index+1} to {sample_num-1} - No Attack detected !!")
            print(f"Sample {sample_num} - Attack detected !!")
            prev_attack_index = sample_num
        # elif Y_pred == 0:
        #     print(f"Sample {sample_num} - No Attack detected")

        # print(f"Sample {sample_num} - Prediction: {Y_pred}, True: {Y_single[0]}")
        sample_num += 1

# === Print Confusion Matrix and Accuracy ===
print("\n=== Evaluation Summary ===")
print(f"Accuracy: {accuracy_score(Y_true, Y_predicted):.4f}")
print("Confusion Matrix:")
print(confusion_matrix(Y_true, Y_predicted))  