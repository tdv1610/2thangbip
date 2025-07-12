import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from predictor import LSTMPredictor
from dataset import BBoxDataset
from torch.utils.data import DataLoader

# ==== Config ====
LEARNING_RATE = 1e-3
DROPOUT = 0.2
BATCH_SIZE = 32
MAX_EPOCHS = 10000    # Cực lớn, thực ra sẽ dừng sớm nếu loss không giảm
PATIENCE = 50        # Sau 30 epoch liên tiếp loss không giảm thì dừng

# ==== Load data ====
X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")
dataset = BBoxDataset(X_train, y_train)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== Model có dropout ====
class LSTMPredictorDropout(nn.Module):
    def __init__(self, input_size=9, hidden_size=64, num_layers=1, output_size=9, dropout=0.2):
        super(LSTMPredictorDropout, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.attn = nn.Linear(hidden_size, 1)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_weights = torch.softmax(self.attn(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        context = self.dropout(context)
        out = self.fc(context)
        return out

model = LSTMPredictorDropout(input_size=X_train.shape[2], hidden_size=64, num_layers=1, output_size=y_train.shape[1], dropout=DROPOUT).to(DEVICE)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ==== Early Stopping ====
best_loss = float('inf')
patience_counter = 0
loss_history = []

for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    running_loss = 0.0
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(dataset)
    loss_history.append(epoch_loss)

    if epoch_loss < best_loss:
        best_loss = epoch_loss
        patience_counter = 0
        torch.save(model.state_dict(), "bbox_predictor.pth")
    else:
        patience_counter += 1

    print(f"Epoch {epoch}/{MAX_EPOCHS}, Loss: {epoch_loss:.6f} (Best: {best_loss:.6f}), Patience: {patience_counter}/{PATIENCE}")

    if patience_counter >= PATIENCE:
        print(f"Early stopping at epoch {epoch}. Loss không giảm thêm {PATIENCE} epoch liên tiếp.")
        break

print("Đã lưu model tốt nhất (bbox_predictor.pth)")

np.save("train_loss_history.npy", np.array(loss_history))
