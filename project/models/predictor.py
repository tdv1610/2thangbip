import torch
import torch.nn as nn
from project.utils.utils import multi_level_lag_scheme, normalize_input, denormalize_output

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
        # 1. Encoder
        encoder_outputs, (hidden, cell) = self.encoder(src)  # hidden: (num_layers, batch, hidden)

        # 2. Input đầu tiên cho decoder: bước cuối src
        input_decoder = src[:, -1, :].unsqueeze(1)  # (batch, 1, 9)

        outputs = []
        for _ in range(self.pred_length):
            output, (hidden, cell) = self.decoder(input_decoder, (hidden, cell))  # (batch, 1, hidden)
            pred = self.fc_out(output.squeeze(1))  # (batch, 9)
            outputs.append(pred.unsqueeze(1))
            if trg is not None and torch.rand(1).item() < teacher_forcing_ratio:
                input_decoder = trg[:, outputs.__len__()-1, :].unsqueeze(1)
            else:
                input_decoder = pred.unsqueeze(1)
        outputs = torch.cat(outputs, dim=1)  # (batch, pred_length, 9)
        return outputs


class BoundingBoxPredictor:
    """Wrapper suy luận cho LSTM, xử lý normalize/denormalize và tuỳ chọn MLS."""
    def __init__(self, model: nn.Module, scaler, pred_length=10):
        self.model = model
        self.scaler = scaler
        self.pred_length = pred_length

    def predict_from_array(self, seq, use_mls: bool = False):
        """
        seq: np.ndarray shape (7, 9) theo thời gian. Nếu use_mls=True sẽ áp dụng MLS trước khi chuẩn hoá.
        Trả về list length=pred_length các bbox [x1,y1,x2,y2] đã denormalize.
        """
        import numpy as np
        self.model.eval()
        device = next(self.model.parameters()).device
        seq = np.asarray(seq)
        if use_mls:
            seq = multi_level_lag_scheme(seq)
        norm_input = normalize_input(seq, self.scaler)
        input_tensor = torch.tensor(norm_input, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            preds = self.model(input_tensor, teacher_forcing_ratio=0).squeeze(0).cpu().numpy()
        denorm_preds = [denormalize_output(p, self.scaler) for p in preds]
        return denorm_preds

    def predict(self, df):
        """Giữ API cũ: dự đoán cho toàn bộ DataFrame đã sort theo frame."""
        import numpy as np
        df_sorted = df.sort_values("frame")
        predictions = []
        self.model.eval()
        device = next(self.model.parameters()).device
        for idx in range(len(df_sorted)):
            if idx < 6:
                predictions.append(None)
                continue
            seq = df_sorted.iloc[idx-6: idx+1][["x1", "y1", "x2", "y2", "vx1", "vy1", "vx2", "vy2", "speed_kmh"]].values
            norm_input = normalize_input(seq, self.scaler)
            input_tensor = torch.tensor(norm_input, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                preds = self.model(input_tensor, teacher_forcing_ratio=0).squeeze(0).cpu().numpy()
            pred_boxes = [denormalize_output(pred, self.scaler) for pred in preds]
            predictions.append(pred_boxes)
        return predictions

    def predict_single(self, seq_df):
        if len(seq_df) < 7:
            return None
        import numpy as np
        self.model.eval()
        device = next(self.model.parameters()).device
        seq = seq_df[["x1", "y1", "x2", "y2", "vx1", "vy1", "vx2", "vy2", "speed_kmh"]].values[-7:]
        norm_input = normalize_input(seq, self.scaler)
        input_tensor = torch.tensor(norm_input, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            preds = self.model(input_tensor, teacher_forcing_ratio=0).squeeze(0).cpu().numpy()
        denorm_preds = [denormalize_output(pred, self.scaler) for pred in preds]
        return denorm_preds
