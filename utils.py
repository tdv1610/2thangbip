import numpy as np
import cv2

def multi_level_lag_scheme(seq, lags=[1, 2, 3]):
    """
    Áp dụng multi-level lag scheme:
    Nếu seq có shape (n, features) và n >= max(lags),
    lấy các mẫu cuối theo các lag rồi nối lại theo chiều đầu ra.
    """
    n = seq.shape[0]
    if n < max(lags):
        return seq
    lagged = [seq[n - lag - 1:n - lag] for lag in lags]
    return np.concatenate(lagged, axis=0)

def normalize_input(seq, scaler):
    """
    Normalize dữ liệu đầu vào dùng scaler (ví dụ, StandardScaler).
    scaler cần có thuộc tính mean_ và scale_.
    """
    num_features = scaler.mean_.shape[0]
    flat_seq = seq.reshape(-1, num_features)
    normalized = scaler.transform(flat_seq)
    return normalized.reshape(seq.shape)

def denormalize_output(pred, scaler):
    """
    Denormalize vector dự đoán (lấy 4 giá trị đầu tiên: [x1, y1, x2, y2])
    """
    indices = [0, 1, 2, 3]
    pred_norm = pred[indices]
    mean = scaler.mean_[indices]
    scale = scaler.scale_[indices]
    return pred_norm * scale + mean

def draw_boxes(frame, boxes):
    """
    Vẽ danh sách bounding box lên frame.
    boxes: list các tuple ((x1, y1, x2, y2), color)
    """
    for (x1, y1, x2, y2), color in boxes:
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
