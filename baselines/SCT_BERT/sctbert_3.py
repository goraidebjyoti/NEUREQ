# sctbert_3.py

"""
Baseline Model: SCT-BERT (Step 3/4) - Self-Supervised Training

Objective:
    This script trains the SCT-BERT (Self-Supervised MonoBERT) baseline evaluated 
    in Section 4.2 of the NEUREQ paper. 
    
    Using the dataset generated in Steps 1 and 2, it fine-tunes a scientific BERT 
    model to classify pairs of (Brief Summary, Eligibility Criteria). By learning 
    to distinguish genuine trial matches from randomly sampled mismatches, the 
    network implicitly learns the semantic parameters of clinical relevance.
"""

import json
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW   # PyTorch's native Adam optimizer with weight decay
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from tqdm import tqdm
import os

# =============================================================================
# HARDWARE CONFIGURATION
# =============================================================================
# Restrict execution to GPU 1. Training transformers requires significant VRAM, 
# so isolating the process prevents Out-Of-Memory (OOM) collisions.
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# =============================================================================
# CONFIGURATION & HYPERPARAMETERS
# =============================================================================
# --- Model Selection ---
# We initialize from SciBERT, a foundation model pre-trained heavily on 
# biomedical literature, providing a strong prior for clinical vocabulary.
MODEL_NAME = "allenai/scibert_scivocab_uncased"   

# --- File Paths ---
# Input: The balanced (1 Positive : 2 Negatives) dataset from Step 2
TRAIN_FILE = "data/clinicaltrials/train_pairs.jsonl"
# Output: Directory to save the fine-tuned model weights
OUTPUT_DIR = "models/sct_bert"

# --- Training Hyperparameters ---
BATCH_SIZE = 32
LR = 2e-5             # Standard learning rate for fine-tuning BERT architectures
EPOCHS = 1            # 1 Epoch is standard for self-supervised tasks on massive datasets
WARMUP_RATIO = 0.1    # The learning rate scales up linearly for the first 10% of steps

# Automatically route tensors to GPU if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================
# DATASET LOADING & PREPROCESSING
# =============================================================================
print("Loading dataset...")
# HuggingFace 'datasets' library efficiently streams the JSONL file, preventing RAM bloat
dataset = load_dataset("json", data_files=TRAIN_FILE, split="train")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def preprocess(example):
    """
    Cross-Encoder Tokenization:
    Packs the summary and criteria into a single sequence: [CLS] Summary [SEP] Criteria [SEP].
    This allows the transformer's self-attention heads to compare specific terms 
    in the summary directly against specific requirements in the criteria.
    """
    return tokenizer(
        example["summary"],
        example["criteria"],
        truncation=True,        # Truncate overly long text blocks
        max_length=512,         # Hard limit for standard BERT models
        padding="max_length"    # Pad shorter sequences with 0s for uniform batch matrices
    )

print("Tokenizing dataset...")
# .map() applies the preprocessing function across the dataset using an optimized C++ backend
dataset = dataset.map(
    preprocess,
    batched=True,
    # Drop the raw text columns to free up memory; the model only computes on numeric IDs
    remove_columns=["summary", "criteria", "trial_id_summary", "trial_id_criteria"]
)

# Convert the dataset into native PyTorch tensors
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# DataLoader handles batching and shuffling during the training loop
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# =============================================================================
# MODEL INITIALIZATION
# =============================================================================
print("Loading model...")
# num_labels=2 configures the model for Binary Classification. 
# It adds a randomly initialized Linear layer on top of the pooled [CLS] token.
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
model.to(DEVICE)

# =============================================================================
# OPTIMIZER & SCHEDULER
# =============================================================================
optimizer = AdamW(model.parameters(), lr=LR)

# Calculate total optimization steps
total_steps = len(dataloader) * EPOCHS
warmup_steps = int(total_steps * WARMUP_RATIO)

# Linear Scheduler: Prevents catastrophic forgetting by slowly warming up the 
# learning rate, then linearly decaying it to zero.
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
)

# Automatic Mixed Precision (AMP) Scaler:
# Prevents gradient underflow when calculating derivatives in 16-bit precision.
scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

# =============================================================================
# TRAINING LOOP
# =============================================================================
print("Starting training...")
model.train() # Enable Dropout and BatchNorm tracking

# LOOP EXPLANATION (OUTER): Iterate over epochs
for epoch in range(EPOCHS):
    loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    
    # LOOP EXPLANATION (INNER): Iterate over batches of (Summary, Criteria) pairs
    for batch in loop:
        # Move tensor batch to GPU VRAM
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        
        # AMP Context Manager: Dynamically casts operations to FP16 where safe, 
        # drastically reducing VRAM usage and speeding up Tensor Core operations.
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["label"]   # 1 for True Matches, 0 for Random Mismatches
            )
            # AutoModelForSequenceClassification automatically computes CrossEntropyLoss 
            # internally when the 'labels' argument is provided.
            loss = outputs.loss

        # 1. Scale the loss and compute gradients (Backward Pass)
        scaler.scale(loss).backward()
        
        # 2. Unscale gradients and update model weights (Optimizer Step)
        scaler.step(optimizer)
        
        # 3. Update the scale factor for the next iteration
        scaler.update()
        
        # 4. Advance the learning rate schedule
        scheduler.step()
        
        # 5. Flush the gradients so they don't accumulate into the next batch
        optimizer.zero_grad()

        # Update the progress bar GUI with the current batch loss
        loop.set_postfix(loss=loss.item())

# =============================================================================
# SAVE MODEL ARTIFACTS
# =============================================================================
print("Saving model...")
# Securely dump the fine-tuned weights, configuration, and tokenizer vocabulary 
# to disk so it can be loaded for zero-shot inference in Step 4.
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"✅ Training finished. Model saved to {OUTPUT_DIR}")