import re
import numpy as np
import joblib
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

def parse_log_line(line):
    match = re.match(r"\(([\d\.]+)\)\s+\S+\s+([0-9A-Fa-f]+)\s+\[(\d+)\]\s+((?:[0-9A-Fa-f]{2} ?)+)", line.strip())
    if not match:
        return None
    timestamp = float(match.group(1))
    can_id = match.group(2).lower().zfill(4)
    dlc = int(match.group(3))
    data_bytes = [int(byte, 16) for byte in match.group(4).strip().split()]
    data_bytes += [0] * (8 - len(data_bytes))  # pad to 8 bytes
    label = 1 if set(can_id) <= {'0'} else 0   # 1 = attack, 0 = benign
    return timestamp, [int(can_id, 16), dlc] + data_bytes[:8], label

def load_data_from_file(path):
    X, Y = [], []
    prev_timestamp = None
    with open(path, 'r') as f:
        for line in f:
            parsed = parse_log_line(line)
            if parsed:
                timestamp, features, label = parsed

                
                if prev_timestamp is None:
                    iat = 0.0
                else:
                    iat = timestamp - prev_timestamp
                prev_timestamp = timestamp

                
                features.append(iat)
                X.append(features)
                Y.append(label)
    return np.array(X), np.array(Y)


X_train, Y_train = load_data_from_file("train_data.log")


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)


model = SGDClassifier(loss='log_loss', random_state=42)
model.partial_fit(X_train_scaled, Y_train, classes=np.array([0, 1]))


joblib.dump(model, "carla_sgd_model.joblib")
joblib.dump(scaler, "train_scaler_carla.joblib")
print("Model and scaler saved.")
