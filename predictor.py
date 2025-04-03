import torch
import numpy as np
from utils import multi_level_lag_scheme, normalize_input, denormalize_output

class LSTMPredictor(torch.nn.Module):
    def __init__(self, input_size=5, hidden_size=64, num_layers=1, output_size=5):
        """
        Mô hình LSTM dự đoán bounding box.
        Input: [x1, y1, x2, y2, speed_kmh] cho mỗi frame.
        Output: Dự đoán bounding box gồm 5 giá trị.
        """
        super(LSTMPredictor, self).__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # x: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]  # Lấy output cuối chuỗi
        out = self.fc(out)         # (batch, output_size)
        return out

class BoundingBoxPredictor:
    def __init__(self, model, scaler):
        """
        model: instance của LSTMPredictor.
        scaler: đối tượng scaler đã huấn luyện dùng để normalize/denormalize.
        """
        self.model = model
        self.scaler = scaler

    def predict(self, df):
        """
        df: pandas DataFrame chứa lịch sử bounding box của 1 đối tượng.
        Các cột cần có: "frame", "x1", "y1", "x2", "y2", "speed_kmh"
        Trả về danh sách dự đoán cho từng frame (None nếu không đủ dữ liệu)
        """
        df_sorted = df.sort_values("frame")
        predictions = []
        # Sử dụng 7 frame cuối (6 frame lịch sử + frame hiện tại) để dự đoán bounding box hiện tại
        for idx in range(len(df_sorted)):
            if idx < 6:
                predictions.append(None)
                continue
            seq = df_sorted.iloc[idx-6: idx+1][["x1", "y1", "x2", "y2", "speed_kmh"]].values  # (7, 5)
            input_seq = multi_level_lag_scheme(seq)
            input_seq = normalize_input(input_seq, self.scaler)
            input_tensor = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0)  # (1, seq_len, 5)
            with torch.no_grad():
                pred = self.model(input_tensor).squeeze(0).numpy()  # (5,)
            pred_box = denormalize_output(pred, self.scaler)
            predictions.append(pred_box)
        return predictions
