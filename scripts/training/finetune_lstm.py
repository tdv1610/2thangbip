import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Ensure repository root on path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from project.models.predictor import LSTMSeq2SeqPredictor
from project.data.dataset import BBoxDataset
from torch.utils.data import DataLoader

LEARNING_RATE = 1e-4      # Thường giảm lr khi fine-tune
BATCH_SIZE = 32
MAX_EPOCHS = 2000
PATIENCE = 20
PRED_LENGTH = 10

ARRAYS_DIR = os.path.join(ROOT_DIR, "artifacts", "arrays")
WEIGHTS_DIR = os.path.join(ROOT_DIR, "artifacts", "weights")

# Load data mới (vd: 1 video/1 domain khác)
X_ft = np.load(os.path.join(ARRAYS_DIR, "X_finetune.npy"))     # Chuẩn bị như X_train.npy (7, 9)
y_ft = np.load(os.path.join(ARRAYS_DIR, "y_finetune.npy"))     # (10, 9)
dataset = BBoxDataset(X_ft, y_ft)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LSTMSeq2SeqPredictor(
    input_size=X_ft.shape[2], hidden_size=64, num_layers=1,
    output_size=y_ft.shape[2], pred_length=PRED_LENGTH
).to(DEVICE)

# 1. Load lại trọng số pretrained
model.load_state_dict(torch.load(os.path.join(WEIGHTS_DIR, "pretrain.pth"), map_location=DEVICE))

# 2. (Tuỳ chọn) Đóng băng encoder, chỉ train decoder + fc_out
# for param in model.encoder.parameters():
#     param.requires_grad = False

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

best_loss = float('inf')
patience_counter = 0
loss_history = []

for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    running_loss = 0.0
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()
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
        torch.save(model.state_dict(), os.path.join(WEIGHTS_DIR, "finetune.pth"))
    else:
        patience_counter += 1

    print(f"Epoch {epoch}, Loss: {epoch_loss:.6f} (Best: {best_loss:.6f}), Patience: {patience_counter}/{PATIENCE}")
    if patience_counter >= PATIENCE:
        print(f"Early stopping tại epoch {epoch}.")
        break

print("Đã lưu model finetuned (finetune.pth)")
np.save(os.path.join(ARRAYS_DIR, "finetune_loss_history.npy"), np.array(loss_history))
