import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import pickle

# --- Cấu hình ---
CSV_PATH = "/Users/nhxtrxng/2thangbip/data/challenge_data.csv"       # File CSV chứa dữ liệu huấn luyện
MODEL_SAVE_PATH = "bbox_predictor.pth"
SCALER_SAVE_PATH = "scaler.pkl"
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
SEQUENCE_LENGTH = 7   # Sử dụng 7 frame làm input, target là frame tiếp theo
INPUT_SIZE = 5        # [x1, y1, x2, y2, speed_kmh]
HIDDEN_SIZE = 64
NUM_LAYERS = 1
OUTPUT_SIZE = 5       # Dự đoán 1 bounding box gồm 5 giá trị

# --- Định nghĩa DEVICE ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Định nghĩa model LSTM ---
class LSTMPredictor(nn.Module):
    def __init__(self, input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, output_size=OUTPUT_SIZE):
        super(LSTMPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # x: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]  # Lấy output của phần tử cuối chuỗi
        out = self.fc(out)         # (batch, output_size)
        return out

# --- Hàm tạo chuỗi dữ liệu từ DataFrame ---
def create_sequences(df, sequence_length=SEQUENCE_LENGTH):
    X, y = [], []
    # Group theo 'id'
    grouped = df.groupby("id")
    for obj_id, group in grouped:
        group_sorted = group.sort_values("frame")
        # Sử dụng các cột x1,y1,x2,y2,speed_kmh làm input
        data = group_sorted[["x1", "y1", "x2", "y2", "speed_kmh"]].values  # shape: (n, 5)
        # Tạo sequence: với mỗi cửa sổ gồm sequence_length frame, target là frame tiếp theo
        for i in range(len(data) - sequence_length):
            seq_input = data[i:i+sequence_length]
            target = data[i+sequence_length]  # target là frame tiếp theo
            X.append(seq_input)
            y.append(target)
    return np.array(X), np.array(y)

# --- Đọc và xử lý dữ liệu CSV ---
df = pd.read_csv(CSV_PATH)

# Tách cột 'bbox' thành 4 cột riêng: x1, y1, x2, y2
df[['x1', 'y1', 'x2', 'y2']] = df['bbox'].str.split(',', expand=True).astype(float)

# Tạo các sequence huấn luyện
X, y = create_sequences(df, sequence_length=SEQUENCE_LENGTH)
print(f"Đã tạo {X.shape[0]} mẫu. X shape: {X.shape}, y shape: {y.shape}")

# --- Chuẩn hóa dữ liệu ---
# Huấn luyện scaler trên dữ liệu target (y)
scaler = StandardScaler()
y_scaled = scaler.fit_transform(y)

# Lưu scaler vào file
with open(SCALER_SAVE_PATH, "wb") as f:
    pickle.dump(scaler, f)
print(f"Đã lưu scaler vào {SCALER_SAVE_PATH}")

# Cũng chuẩn hóa X
X_flat = X.reshape(-1, INPUT_SIZE)
X_scaled = scaler.fit_transform(X_flat).reshape(X.shape)

# --- Tạo DataLoader ---
import torch.utils.data as data

class BBoxDataset(data.Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)  # shape: (num_samples, SEQUENCE_LENGTH, INPUT_SIZE)
        self.y = torch.tensor(y, dtype=torch.float32)  # shape: (num_samples, OUTPUT_SIZE)
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

dataset = BBoxDataset(X_scaled, y_scaled)
dataloader = data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# --- Khởi tạo model, loss, optimizer ---
model = LSTMPredictor(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, output_size=OUTPUT_SIZE)
model.to(DEVICE)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- Huấn luyện model ---
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for inputs, targets in dataloader:
        inputs = inputs.to(DEVICE)
        targets = targets.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(dataset)
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss:.6f}")

# --- Lưu model ---
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"Đã lưu state_dict của model vào {MODEL_SAVE_PATH}")
