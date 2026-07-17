"""
Baseline: Self-Supervised MonoBERT Training (Step 1)
This script fine-tunes a pre-trained SciBERT model on a synthetic dataset of 
clinical trial field pairs (Brief Summary vs. Eligibility Criteria). 
By training on these pairs, the model learns the semantic relationship between 
patient-like narratives (summaries) and strict medical requirements (criteria) 
without relying on human-annotated relevance judgments.
"""

import os
import json
import random
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_scheduler
)
from torch.optim import AdamW
from tqdm import tqdm

# =============================
# Configuration & Hardware Setup
# =============================
# Restrict execution to a single GPU to prevent multi-device memory fragmentation
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

# Model & Data Paths
MODEL_NAME = "allenai/scibert_scivocab_uncased" # Base model specialized for scientific text
DATA_FILE = "data/clinicaltrials/train_pairs.jsonl" # Pre-generated contrastive pairs from SCT_BERT dataset creation 
OUTPUT_DIR = "models/self_supervised_monobert" # Directory for saved checkpoints

# Hyperparameters
BATCH_SIZE = 16
EPOCHS = 5
LR = 2e-5 # Standard learning rate for fine-tuning BERT-based architectures
MAX_LEN = 512 # Maximum context window for SciBERT
SEED = 42
SAVE_EVERY_EPOCH = True # Retain intermediate weights for early-stopping analysis

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================
# Logging Configuration
# =============================
# Stream logs to both the console and a persistent text file for auditing
logging.basicConfig(
    filename=os.path.join(OUTPUT_DIR, "train_log.txt"),
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger()
logger.addHandler(logging.StreamHandler())

# =============================
# Reproducibility Engine
# =============================
def set_seed(seed: int):
    """Locks all random number generators to ensure deterministic training runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)

# =============================
# Data Management
# =============================
class TrialsPairsDataset(Dataset):
    """
    Custom PyTorch Dataset that loads JSONL files containing text pairs.
    Extracts the 'summary' (Sentence A), 'criteria' (Sentence B), and 'label' (0 or 1).
    """
    def __init__(self, file_path: str):
        self.samples = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                # Ensure all necessary fields exist before appending to memory
                if "summary" in obj and "criteria" in obj and "label" in obj:
                    self.samples.append({
                        "summary": obj["summary"],
                        "criteria": obj["criteria"],
                        "label": int(obj["label"])
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def collate_fn(batch, tokenizer, max_len):
    """
    Dynamically pads sequences in a batch to the length of the longest sequence in THAT batch.
    This is significantly faster than padding all sequences to MAX_LEN statically.
    """
    texts_a = [b["summary"] for b in batch]
    texts_b = [b["criteria"] for b in batch]
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)

    # Tokenize Sentence A and Sentence B together, inserting the [SEP] token automatically
    encoding = tokenizer(
        texts_a,
        texts_b,
        truncation=True,
        padding=True,          
        max_length=max_len,
        return_tensors="pt",
    )
    encoding["labels"] = labels
    return encoding

# =============================
# Initialization
# =============================
# Load the pre-trained fast tokenizer
tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased", use_fast=True)
dataset = TrialsPairsDataset(DATA_FILE)
print(f"Loaded {len(dataset)} pairs from {DATA_FILE}")
logger.info(f"Loaded {len(dataset)} pairs from {DATA_FILE}")

# Create a robust 90/10 Train/Validation split
indices = list(range(len(dataset)))
random.shuffle(indices)
split = int(0.9 * len(indices))
train_idx, val_idx = indices[:split], indices[split:]

train_subset = torch.utils.data.Subset(dataset, train_idx)
val_subset = torch.utils.data.Subset(dataset, val_idx)

# Initialize DataLoaders with the dynamic collate function
train_loader = DataLoader(
    train_subset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=lambda b: collate_fn(b, tokenizer, MAX_LEN),
)
val_loader = DataLoader(
    val_subset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=lambda b: collate_fn(b, tokenizer, MAX_LEN),
)

# =============================
# Model & Optimizer Setup
# =============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize SciBERT with a binary classification head (num_labels=2)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
model.to(device)

optimizer = AdamW(model.parameters(), lr=LR)
num_training_steps = len(train_loader) * EPOCHS

# Linear learning rate scheduler (no warmup steps explicitly configured)
lr_scheduler = get_scheduler("linear", optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

# =============================
# Evaluation Utility
# =============================
def compute_accuracy(logits, labels):
    """Calculates batch accuracy by taking the argmax of the raw logits."""
    preds = torch.argmax(logits, dim=1)
    return (preds == labels).float().mean().item()

# =============================
# Primary Training Loop
# =============================
for epoch in range(1, EPOCHS + 1):
    # ---- Training Phase ----
    model.train()
    total_loss, total_acc, steps = 0.0, 0.0, 0
    
    # Wrap train_loader in tqdm for a real-time progress bar
    pbar = tqdm(train_loader, desc=f"Train Epoch {epoch}/{EPOCHS}", leave=False)

    for batch in pbar:
        # Move inputs and labels to target device (GPU)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss, logits = outputs.loss, outputs.logits

        # Backward pass & optimization
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad() # Flush gradients for the next step

        # Accumulate metrics
        acc = compute_accuracy(logits.detach().cpu(), labels.detach().cpu())
        total_loss += loss.item()
        total_acc += acc
        steps += 1
        
        # Update progress bar dynamically
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{acc:.4f}"})

    avg_train_loss = total_loss / steps
    avg_train_acc = total_acc / steps

    # ---- Validation Phase ----
    model.eval()
    val_loss, val_acc, vsteps = 0.0, 0.0, 0
    
    # Disable gradient tracking to save memory and compute during validation
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss, logits = outputs.loss, outputs.logits

            acc = compute_accuracy(logits.detach().cpu(), labels.detach().cpu())
            val_loss += loss.item()
            val_acc += acc
            vsteps += 1

    avg_val_loss = val_loss / vsteps
    avg_val_acc = val_acc / vsteps

    # Log epoch summary
    msg = (f"Epoch {epoch}/{EPOCHS} | "
           f"Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f} | "
           f"Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}")
    print(msg)
    logger.info(msg)

    # Save intermediate checkpoint
    if SAVE_EVERY_EPOCH:
        out_dir = Path(OUTPUT_DIR) / f"epoch-{epoch}"
        out_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)

# =============================
# Final Serialization
# =============================
final_dir = Path(OUTPUT_DIR) / "final"
final_dir.mkdir(parents=True, exist_ok=True)
model.save_pretrained(final_dir)
tokenizer.save_pretrained(final_dir)

print("Training finished. Final model saved to:", final_dir)
logger.info("Training finished.")