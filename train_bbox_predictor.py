# train_bbox_predictor.py
import os
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

CSV_PATH = "data/challenge_data.csv"  # Đổi path đúng file CSV
SEQUENCE_LENGTH = 7
INPUT_SIZE = 9  # 9 features: [x1,y1,x2,y2,vx1,vy1,vx2,vy2,speed_kmh]

df = pd.read_csv(CSV_PATH)
# Không cần split bbox nữa, vì đã có sẵn các cột x1, y1, ..., speed_kmh
def create_sequences(df, sequence_length=7, min_len=10):
    X, y = [], []
    grouped = df.groupby("id")
    for obj_id, group in grouped:
        group_sorted = group.sort_values("frame")
        if len(group_sorted) < sequence_length + 1 or len(group_sorted) < min_len:
            continue  # Bỏ object quá ngắn
        data = group_sorted[["x1", "y1", "x2", "y2", "vx1", "vy1", "vx2", "vy2", "speed_kmh"]].values
        for i in range(len(data) - sequence_length):
            seq_input = data[i:i+sequence_length]
            target = data[i+sequence_length]
            X.append(seq_input)
            y.append(target)
    return np.array(X), np.array(y)

X, y = create_sequences(df, sequence_length=SEQUENCE_LENGTH)
print(f"Đã tạo {X.shape[0]} mẫu. X shape: {X.shape}, y shape: {y.shape}")

# Chuẩn hóa scaler trên y, apply cho X
scaler = StandardScaler()
y_scaled = scaler.fit_transform(y)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
X_flat = X.reshape(-1, INPUT_SIZE)
X_scaled = scaler.transform(X_flat).reshape(X.shape)

# Lưu X_scaled, y_scaled phòng train lại sau này cho nhanh
np.save("X_train.npy", X_scaled)
np.save("y_train.npy", y_scaled)
print("Đã lưu scaler và dữ liệu train.")
