import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Ensure repository root on path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

DATA_PATH = os.path.join(ROOT_DIR, "data", "processed", "finetune_merged_clean.csv")
SEQUENCE_LENGTH = 7
PRED_LENGTH = 10
INPUT_SIZE = 9
ARRAYS_DIR = os.path.join(ROOT_DIR, "artifacts", "arrays")
SCALER_PATH = os.path.join(ARRAYS_DIR, "scaler.pkl")  # dùng scaler đã fit trên train

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Không tìm thấy file: {DATA_PATH}. Hãy chạy tools/merge_and_clean_finetune_csv.py trước.")
if not os.path.exists(SCALER_PATH):
    raise FileNotFoundError(f"Không tìm thấy scaler: {SCALER_PATH}. Hãy tạo bằng train_bbox_predictor.py trước.")

print(f"Đọc dữ liệu fine-tune từ: {DATA_PATH}")
df_all = pd.read_csv(DATA_PATH)
print(
    f"Đã đọc: {len(df_all)} dòng, videos={df_all['video'].nunique()}, tracks={df_all.groupby(['video','id']).ngroups}"
)


def create_sequences(df, sequence_length=7, pred_length=10, min_len=15, require_consecutive=True):
    X, y = [], []
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
            if require_consecutive and not np.all(np.diff(frames[i:j]) == 1):
                continue
            X.append(data[i:i+sequence_length])
            y.append(data[i+sequence_length:j])
    return np.array(X), np.array(y)


X_ft, y_ft = create_sequences(
    df_all,
    sequence_length=SEQUENCE_LENGTH,
    pred_length=PRED_LENGTH,
    require_consecutive=True,
)
print(f"Tạo được {X_ft.shape[0]} mẫu fine-tune. X: {X_ft.shape}, y: {y_ft.shape}")

# Chuẩn hóa bằng scaler đã fit từ train để giữ nhất quán phân phối
with open(SCALER_PATH, "rb") as f:
    scaler: StandardScaler = pickle.load(f)

X_flat = X_ft.reshape(-1, INPUT_SIZE)
X_scaled = scaler.transform(X_flat).reshape(X_ft.shape)
y_flat = y_ft.reshape(-1, INPUT_SIZE)
y_scaled = scaler.transform(y_flat).reshape(y_ft.shape)

np.save(os.path.join(ARRAYS_DIR, "X_finetune.npy"), X_scaled)
np.save(os.path.join(ARRAYS_DIR, "y_finetune.npy"), y_scaled)
print("Đã lưu X_finetune.npy và y_finetune.npy (đã chuẩn hóa bằng scaler train) vào artifacts/arrays.")
