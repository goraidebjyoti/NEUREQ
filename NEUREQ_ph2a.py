import os

# ====================== FORCE GPU 0 ======================
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import sys

# ====================== CONFIG ======================
CSV_PATH = "data/train/neureq_syn_training_data_1196.csv"
MODEL_DIR = "models_new/LSTM_1196"
os.makedirs(MODEL_DIR, exist_ok=True)

EPOCHS = 30
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
SEED = 42
INPUT_DIM = 1
SEQUENCE_LENGTH = 10
HIDDEN_SIZE = 64

# ====================== DEVICE SETUP ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"💻 Using device: {device}")
torch.manual_seed(SEED)
np.random.seed(SEED)

# ====================== DATASET ======================
class NEUREQDataset(Dataset):
    def __init__(self, df):
        self.X = df[[f"q{i}" for i in range(1, 11)]].values.astype(np.float32)
        self.X = np.expand_dims(self.X, axis=2)  # shape: (N, 10, 1)
        self.y = df["label"].astype(np.float32).values

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])

# ====================== MODEL ======================
class LSTMModel(nn.Module):
    def __init__(self, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=INPUT_DIM,
            hidden_size=HIDDEN_SIZE,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0.0
        )
        self.fc = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# ====================== LOAD DATA ======================
df = pd.read_csv(CSV_PATH)
dataset = NEUREQDataset(df)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(SEED))

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

# ====================== TRAIN FUNCTION ======================
def train_lstm_model(num_layers, model_prefix):
    log_path = os.path.join(MODEL_DIR, f"{model_prefix}_train_log.txt")
    sys.stdout = open(log_path, "w")

    print(f"\n🚀 Training LSTM-{num_layers}L (NEUREQ)")
    model = LSTMModel(num_layers).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        loop = tqdm(train_loader, desc=f"[Train] Epoch {epoch}/{EPOCHS} (LSTM-{num_layers}L)")
        for X, y in loop:
            X, y = X.to(device), y.to(device).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        # === Validation ===
        model.eval()
        val_loss = 0
        correct, total = 0, 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device).unsqueeze(1)
                outputs = model(X)
                val_loss += criterion(outputs, y).item()
                correct += (outputs > 0.5).eq(y).sum().item()
                total += y.size(0)

        val_acc = correct / total
        print(f"✅ Epoch {epoch} — Train Loss: {total_loss:.4f} — Val Loss: {val_loss:.4f} — Val Acc: {val_acc:.4f}")

        # === Save checkpoint for this epoch ===
        ckpt_path = os.path.join(MODEL_DIR, f"{model_prefix}_epoch{epoch}.pt")
        torch.save(model.state_dict(), ckpt_path)
        print(f"💾 Saved checkpoint: {ckpt_path}")

    sys.stdout.close()
    sys.stdout = sys.__stdout__
    print(f"📄 Training log saved: {log_path}")

# ====================== START TRAINING ======================
# 🚨 Only training 1-layer LSTM (our NEUREQ model)
train_lstm_model(1, "lstm_1layer")
