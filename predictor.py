import torch
import torch.nn as nn
from utils import multi_level_lag_scheme, normalize_input, denormalize_output

class LSTMSeq2SeqPredictor(nn.Module):
    def __init__(self, input_size=9, hidden_size=64, num_layers=1, output_size=9, pred_length=10):
        super().__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.decoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_size, output_size)
        self.pred_length = pred_length

    def forward(self, src, trg=None, teacher_forcing_ratio=0.5):
        """
        src: (batch, 7, 9)
        trg: (batch, 10, 9) ground truth (chỉ dùng khi train)
        teacher_forcing_ratio: xác suất dùng GT ở mỗi step decoder

        return: (batch, 10, 9) dự đoán 10 bước tiếp theo
        """
        batch_size = src.size(0)
        device = src.device

        # 1. Encoder
        encoder_outputs, (hidden, cell) = self.encoder(src)  # hidden: (num_layers, batch, hidden)

        # 2. Khởi tạo input đầu tiên cho decoder
        # Thường dùng bước cuối cùng của src (last input)
        input_decoder = src[:, -1, :].unsqueeze(1)  # (batch, 1, 9)

        outputs = []
        for t in range(self.pred_length):
            output, (hidden, cell) = self.decoder(input_decoder, (hidden, cell))  # output: (batch, 1, hidden)
            pred = self.fc_out(output.squeeze(1))  # (batch, 9)
            outputs.append(pred.unsqueeze(1))     # giữ shape (batch, 1, 9)
            # Quyết định input tiếp theo: GT hay pred
            if trg is not None and torch.rand(1).item() < teacher_forcing_ratio:
                # Sử dụng GT step t làm input step t+1
                input_decoder = trg[:, t, :].unsqueeze(1)  # (batch, 1, 9)
            else:
                # Sử dụng output model vừa dự đoán làm input step t+1
                input_decoder = pred.unsqueeze(1)  # (batch, 1, 9)
        outputs = torch.cat(outputs, dim=1)  # (batch, pred_length, 9)
        return outputs

class BoundingBoxPredictor:
    def __init__(self, model, scaler, pred_length=10):
        self.model = model
        self.scaler = scaler
        self.pred_length = pred_length

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
                preds = self.model(input_tensor, teacher_forcing_ratio=0)  # inference: không dùng teacher forcing
                preds = preds.squeeze(0).numpy()  # shape (pred_length, 9)
            # Đưa từng bước về bbox thực
            pred_boxes = [denormalize_output(pred, self.scaler) for pred in preds]
            predictions.append(pred_boxes)
        return predictions  # List, mỗi phần tử là list 10 bbox dự đoán tiếp theo

    def predict_single(self, seq_df):
        if len(seq_df) < 7:
            return None
        seq = seq_df[["x1", "y1", "x2", "y2", "vx1", "vy1", "vx2", "vy2", "speed_kmh"]].values[-7:]
        input_seq = multi_level_lag_scheme(seq)
        input_seq = normalize_input(input_seq, self.scaler)
        input_tensor = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            preds = self.model(input_tensor, teacher_forcing_ratio=0)
            preds = preds.squeeze(0).numpy()  # (pred_length, 9)
        denorm_preds = [denormalize_output(pred, self.scaler) for pred in preds]
        return denorm_preds  # List 10 bbox t+1...t+10
