import os
import sys
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Ensure repository root on path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

DATA_PATH = os.path.join(ROOT_DIR, "data", "processed", "train_merged_clean.csv")  # Dùng file đã merge & clean
SEQUENCE_LENGTH = 7
PRED_LENGTH = 10
INPUT_SIZE = 9

ARRAYS_DIR = os.path.join(ROOT_DIR, "artifacts", "arrays")
os.makedirs(ARRAYS_DIR, exist_ok=True)

# Đọc 1 file đã được làm sạch
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Không tìm thấy file dữ liệu đã làm sạch: {DATA_PATH}")

df_all = pd.read_csv(DATA_PATH)
print(
    f"Đã đọc dữ liệu sạch: {len(df_all)} dòng, videos={df_all['video'].nunique()}, tracks={df_all.groupby(['video','id']).ngroups}"
)


def create_sequences(df, sequence_length=7, pred_length=10, min_len=20, require_consecutive=False):
    X, y = [], []
    # Group theo (video, id) để không trộn đối tượng giữa các video
    grouped = df.groupby(["video", "id"])
    for (video_id, obj_id), group in grouped:
        group_sorted = group.sort_values(["frame", "second"])  # đảm bảo theo thời gian
        data = group_sorted[[
            "x1", "y1", "x2", "y2", "vx1", "vy1", "vx2", "vy2", "speed_kmh"
        ]].values
        frames = group_sorted["frame"].values
        n = len(data)
        if n < sequence_length + pred_length or n < min_len:
            continue
        for i in range(n - sequence_length - pred_length + 1):
            j = i + sequence_length + pred_length
            if require_consecutive:
                # Yêu cầu các frame là liên tiếp để timestep đồng đều
                if not np.all(np.diff(frames[i:j]) == 1):
                    continue
            seq_input = data[i:i+sequence_length]
            seq_output = data[i+sequence_length:j]
            X.append(seq_input)
            y.append(seq_output)
    return np.array(X), np.array(y)


X, y = create_sequences(
    df_all,
    sequence_length=SEQUENCE_LENGTH,
    pred_length=PRED_LENGTH,
    require_consecutive=True
)
print(f"Đã tạo {X.shape[0]} mẫu. X shape: {X.shape}, y shape: {y.shape}")

# Fit scaler trên X rồi transform cả X và y để nhất quán phân phối đặc trưng
scaler = StandardScaler()
X_flat = X.reshape(-1, INPUT_SIZE)
X_scaled = scaler.fit_transform(X_flat).reshape(X.shape)
y_flat = y.reshape(-1, INPUT_SIZE)
y_scaled = scaler.transform(y_flat).reshape(y.shape)

np.save(os.path.join(ARRAYS_DIR, "X_train.npy"), X_scaled)
np.save(os.path.join(ARRAYS_DIR, "y_train.npy"), y_scaled)
with open(os.path.join(ARRAYS_DIR, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)

print("Đã lưu scaler và dữ liệu train vào artifacts/arrays.")
