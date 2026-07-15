import json
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW   # use torch version
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from tqdm import tqdm

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# -----------------
# Config
# -----------------
MODEL_NAME = "allenai/scibert_scivocab_uncased"   # SCT-BERT paper
TRAIN_FILE = "data/clinicaltrials/train_pairs.jsonl"
OUTPUT_DIR = "models/sct_bert"

BATCH_SIZE = 32
LR = 2e-5
EPOCHS = 1
WARMUP_RATIO = 0.1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------
# Load Dataset
# -----------------
print("Loading dataset...")
dataset = load_dataset("json", data_files=TRAIN_FILE, split="train")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def preprocess(example):
    return tokenizer(
        example["summary"],
        example["criteria"],
        truncation=True,
        max_length=512,
        padding="max_length"
    )

print("Tokenizing dataset...")
dataset = dataset.map(
    preprocess,
    batched=True,
    remove_columns=["summary", "criteria", "trial_id_summary", "trial_id_criteria"]
)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# -----------------
# Model
# -----------------
print("Loading model...")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
model.to(DEVICE)

# -----------------
# Optimizer & Scheduler
# -----------------
optimizer = AdamW(model.parameters(), lr=LR)

total_steps = len(dataloader) * EPOCHS
warmup_steps = int(total_steps * WARMUP_RATIO)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
)

scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

# -----------------
# Training Loop
# -----------------
print("Starting training...")
model.train()
for epoch in range(EPOCHS):
    loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for batch in loop:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["label"]   # ✅ must be 'labels'
            )
            loss = outputs.loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad()

        loop.set_postfix(loss=loss.item())

# -----------------
# Save Model
# -----------------
print("Saving model...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"✅ Training finished. Model saved to {OUTPUT_DIR}")
