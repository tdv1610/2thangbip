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
from torch.utils.data import DataLoader, random_split

# ==== Reproducibility ====
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ==== Config ====
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
MAX_EPOCHS = 1000
PATIENCE = 20
PRED_LENGTH = 10
VAL_SPLIT = 0.1
GRAD_CLIP_NORM = 1.0

# ==== Load data ====
ARRAYS_DIR = os.path.join(ROOT_DIR, "artifacts", "arrays")
X_train = np.load(os.path.join(ARRAYS_DIR, "X_train.npy"))
y_train = np.load(os.path.join(ARRAYS_DIR, "y_train.npy"))
full_dataset = BBoxDataset(X_train, y_train)

# Train/Val split
val_size = max(1, int(len(full_dataset) * VAL_SPLIT))
train_size = len(full_dataset) - val_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(SEED))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pin_mem = True if DEVICE.type == "cuda" else False
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=pin_mem, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=pin_mem, num_workers=0)

# ==== Model ====
model = LSTMSeq2SeqPredictor(
    input_size=X_train.shape[2], hidden_size=64, num_layers=1,
    output_size=y_train.shape[2], pred_length=PRED_LENGTH
).to(DEVICE)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

best_val = float('inf')
patience_counter = 0
train_loss_history, val_loss_history = [], []

for epoch in range(1, MAX_EPOCHS + 1):
    # Train
    model.train()
    running_loss = 0.0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs, targets, teacher_forcing_ratio=0.5)
        loss = criterion(outputs, targets)
        loss.backward()
        if GRAD_CLIP_NORM is not None:
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    epoch_train = running_loss / train_size
    train_loss_history.append(epoch_train)

    # Validate
    model.eval()
    val_running = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs, targets, teacher_forcing_ratio=0.0)
            loss = criterion(outputs, targets)
            val_running += loss.item() * inputs.size(0)
    epoch_val = val_running / val_size
    val_loss_history.append(epoch_val)

    scheduler.step(epoch_val)

    improved = epoch_val < best_val
    if improved:
        best_val = epoch_val
        patience_counter = 0
        torch.save(model.state_dict(), os.path.join(ROOT_DIR, "artifacts", "weights", "pretrain.pth"))
    else:
        patience_counter += 1

    print(f"Epoch {epoch}, Train: {epoch_train:.6f}, Val: {epoch_val:.6f} (Best: {best_val:.6f}), Patience: {patience_counter}/{PATIENCE}")
    if patience_counter >= PATIENCE:
        print(f"Early stopping tại epoch {epoch}.")
        break

print("Đã lưu model pretrained (pretrain.pth)")
np.save(os.path.join(ARRAYS_DIR, "pretrain_loss_history.npy"), np.array(train_loss_history))
np.save(os.path.join(ARRAYS_DIR, "pretrain_val_loss_history.npy"), np.array(val_loss_history))
