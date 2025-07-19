import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from predictor import LSTMSeq2SeqPredictor  # Sử dụng model mới seq2seq
from dataset import BBoxDataset
from torch.utils.data import DataLoader

# ==== Config ====
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
MAX_EPOCHS = 10000    # Cực lớn, thực ra sẽ dừng sớm nếu loss không giảm
PATIENCE = 50         # Sau PATIENCE epoch liên tiếp loss không giảm thì dừng
PRED_LENGTH = 10      # Dự đoán 10 bước tiếp theo

# ==== Load data ====
X_train = np.load("X_train.npy")    # (num_samples, 7, 9)
y_train = np.load("y_train.npy")    # (num_samples, 10, 9)
dataset = BBoxDataset(X_train, y_train)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== Model Seq2Seq ====
model = LSTMSeq2SeqPredictor(
    input_size=X_train.shape[2],   # 9
    hidden_size=64,
    num_layers=1,
    output_size=y_train.shape[2],  # 9
    pred_length=PRED_LENGTH
).to(DEVICE)

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
        # Teacher forcing: tỉ lệ 0.5 khi train (giúp model học nhanh và ổn định hơn)
        outputs = model(inputs, targets, teacher_forcing_ratio=0.5)
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
