# predictor.py
import torch
import torch.nn as nn
from utils import multi_level_lag_scheme, normalize_input, denormalize_output

class LSTMPredictor(nn.Module):
    def __init__(self, input_size=9, hidden_size=64, num_layers=1, output_size=9):
        super(LSTMPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.attn = nn.Linear(hidden_size, 1)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_weights = torch.softmax(self.attn(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        out = self.fc(context)
        return out

class BoundingBoxPredictor:
    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler

    def predict(self, df):
        df_sorted = df.sort_values("frame")
        predictions = []
        for idx in range(len(df_sorted)):
            if idx < 6:
                predictions.append(None)
                continue
            seq = df_sorted.iloc[idx-6: idx+1][["x1", "y1", "x2", "y2", "vx1", "vy1", "vx2", "vy2", "speed_kmh"]].values
            input_seq = multi_level_lag_scheme(seq)
            input_seq = normalize_input(input_seq, self.scaler)
            input_tensor = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                pred = self.model(input_tensor).squeeze(0).numpy()
            pred_box = denormalize_output(pred, self.scaler)
            predictions.append(pred_box)
        return predictions
    
    def predict_single(self, seq_df):
        if len(seq_df) < 7:
            return None
        seq = seq_df[["x1", "y1", "x2", "y2", "vx1", "vy1", "vx2", "vy2", "speed_kmh"]].values[-7:]
        input_seq = multi_level_lag_scheme(seq)
        input_seq = normalize_input(input_seq, self.scaler)
        input_tensor = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            pred = self.model(input_tensor).squeeze(0).numpy()
        return denormalize_output(pred, self.scaler)
