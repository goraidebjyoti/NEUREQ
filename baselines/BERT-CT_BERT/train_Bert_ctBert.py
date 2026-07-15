# train_Bert_ctBert.py

"""
Baseline Model Training: BERT & CT-BERT Cross-Encoders

Objective:
    Trains the deep learning baselines (BERT and CT-BERT) evaluated in Section 4.2 
    of the NEUREQ paper. These models act as standard cross-encoders: they concatenate 
    the raw patient query and the raw clinical trial text, process them jointly 
    through a transformer, and apply a regression head to the [CLS] token to predict 
    a point-wise relevance score (1.0 for relevant, 0.0 for non-relevant).

Data Source:
    triplet_syn.jsonl (A text-only variant of the 1196 synthetic dataset)
    Contains: topic_id, topic (patient case), positive_trial (text), negative_trial (text).

Usage:
    Toggle the `MODEL_NAME` and `MODEL_SAVE_PATH` variables to switch between 
    training the standard BERT baseline and the domain-specific CT-BERT baseline.
"""

import os
import json
import torch
import random
import warnings
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from torch import nn

# Suppress HuggingFace deprecation warnings for cleaner logging output
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
# --- TOGGLE THESE TO SWITCH BASELINES ---
# Standard BERT Baseline: "google-bert/bert-base-uncased"
# Domain-Specific Clinical Trial Baseline: "ielabgroup/PubMedBERT-CT-MLM"
MODEL_NAME = "ielabgroup/PubMedBERT-CT-MLM" 

# --- File Paths ---
# The text-only 1196 triplet dataset containing patient descriptions and full trial texts
DATA_PATH = "data/train/triplet_syn.jsonl"  
# Intermediate cache for the flattened Point-wise pairs (for auditing data distribution)
CSV_PATH = "data/train/train_dataset.csv"   

# Dynamically adjust the save directory to avoid overwriting baseline weights
if "PubMedBERT" in MODEL_NAME:
    MODEL_SAVE_PATH = "models/CT_MLM_BERT"
else:
    MODEL_SAVE_PATH = "models/SIMPLE_BERT"

# --- Hyperparameters ---
EPOCHS = 5
BATCH_SIZE = 8

# BERT architectures have a strict 512 maximum token limit. 
# We explicitly allocate this budget: 
# 179 (Query) + 330 (Doc) + 3 (Special Tokens: [CLS], [SEP], [SEP]) = 512 tokens.
MAX_QUERY_LEN = 179 
MAX_DOC_LEN = 330   
SEED = 42

# =============================================================================
# REPRODUCIBILITY
# =============================================================================
def set_seed(seed):
    """
    Locks all random number generators across Python, NumPy, and PyTorch.
    Crucial for academic research to ensure that baseline models are initialized 
    with the exact same random weights and train/val splits across different runs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# =============================================================================
# DATA PREPARATION (Triplet -> Point-wise Pairs)
# =============================================================================
print("Loading and formatting data from triplet file...")
data = []

# LOOP EXPLANATION: 
# The input dataset provides data in triplets: (Query, Positive Doc, Negative Doc).
# However, standard Cross-Encoders evaluate one (Query, Doc) pair at a time to output a scalar.
# We must "flatten" the triplets into independent pairs and assign them explicit binary labels.
with open(DATA_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        if not line.strip():
            continue
            
        item = json.loads(line)
        topic = item.get("topic", "")
        pos_doc = item.get("positive_trial", "")
        neg_doc = item.get("negative_trial", "")
        
        # Create a positive training example (Target Label = 1.0)
        if pos_doc:
            data.append({"query": topic, "doc": pos_doc, "label": 1.0})
        # Create a hard-negative training example (Target Label = 0.0)
        if neg_doc:
            data.append({"query": topic, "doc": neg_doc, "label": 0.0})

# Convert to a Pandas DataFrame for easy manipulation
df = pd.DataFrame(data)

# CRITICAL STEP: We must shuffle the DataFrame. If we don't, the DataLoader might 
# pull a batch of purely positive or purely negative examples, causing the model's 
# gradients to oscillate wildly and preventing convergence.
df = shuffle(df, random_state=SEED).reset_index(drop=True)

# Save intermediate preprocessed CSV for auditing
os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
df.to_csv(CSV_PATH, index=False)

# 90/10 Train-Validation split to monitor for overfitting
train_df, test_df = train_test_split(df, test_size=0.1, random_state=SEED)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# =============================================================================
# PYTORCH DATASET (CROSS-ENCODER FORMATTING)
# =============================================================================
class CTDataset(Dataset):
    """
    Formats the raw textual queries and trials into numerical Cross-Encoder tensor inputs.
    """
    def __init__(self, dataframe, tokenizer, max_query_len, max_doc_len):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_query_len = max_query_len
        self.max_doc_len = max_doc_len
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, index):
        row = self.data.iloc[index]
        query = str(row['query'])
        doc = str(row['doc'])
        label = float(row['label'])
        
        # ARCHITECTURE NOTE: Cross-Encoder Tokenization
        # By passing BOTH `query` and `doc` to the tokenizer simultaneously, HuggingFace 
        # automatically constructs the cross-encoder sequence:
        # [CLS] Query Tokens [SEP] Document Tokens [SEP] [PAD] ...
        # This allows every query token to attend to every document token in the transformer's 
        # self-attention layers, making it highly accurate but computationally expensive.
        encoding = self.tokenizer(
            query,
            doc,
            max_length=self.max_query_len + self.max_doc_len, # Enforces the strict 512 limit
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Squeeze removes the redundant batch dimension added by the tokenizer
        input_ids = encoding['input_ids'].squeeze(0)
        attn_mask = encoding['attention_mask'].squeeze(0)
        
        # Token Type IDs (Segment IDs) tell the model which tokens belong to the query (0s) 
        # and which belong to the document (1s). We safely fetch them as some newer 
        # architectures (like RoBERTa) deprecated their use.
        token_type_ids = encoding.get('token_type_ids', torch.zeros_like(input_ids)).squeeze(0)
        
        return input_ids, attn_mask, token_type_ids, torch.tensor(label, dtype=torch.float)

# Instantiate the dataset objects
train_dataset = CTDataset(train_df, tokenizer, MAX_QUERY_LEN, MAX_DOC_LEN)
test_dataset = CTDataset(test_df, tokenizer, MAX_QUERY_LEN, MAX_DOC_LEN)

# DataLoader automatically batches the tensors and handles parallel CPU fetching
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================
class BERTRegression(nn.Module):
    """
    Wraps the HuggingFace base transformer with a linear regression head.
    This architecture outputs a continuous score rather than a categorical class.
    """
    def __init__(self):
        super(BERTRegression, self).__init__()
        # Load the frozen pre-trained weights (BERT or CT-BERT)
        self.bert = AutoModel.from_pretrained(MODEL_NAME)
        
        # Regression layer: Maps the 768-dimensional hidden state space down to 1 dimension
        self.regressor = nn.Linear(self.bert.config.hidden_size, 1)
        
    def forward(self, input_ids, attn_mask, token_type_ids):
        # Forward pass through all 12 transformer layers
        outputs = self.bert(input_ids=input_ids, attention_mask=attn_mask, token_type_ids=token_type_ids)
        
        # hidden_state shape: (Batch_Size, Sequence_Length=512, Hidden_Dim=768)
        hidden_state = outputs.last_hidden_state
        
        # CLS POOLING: We isolate the very first token [CLS] at index 0. 
        # Because of bidirectional self-attention, the [CLS] token aggregates the 
        # interaction signals between the entire query and the entire document.
        # cls_output shape drops to: (Batch_Size, 768)
        cls_output = hidden_state[:, 0, :]  
        
        # Pass through linear layer. Shape becomes (Batch_Size, 1)
        # Squeeze(-1) flattens it to a 1D vector (Batch_Size,) to match label tensor shape.
        return self.regressor(cls_output).squeeze(-1)

# Initialize model to GPU
model = BERTRegression().to(device)

# We use Mean Squared Error (MSE) Loss because the Cross-Encoder is structured 
# as a regression task, predicting a continuous scalar targeting 1.0 or 0.0.
criterion = nn.MSELoss()

# AdamW includes weight decay regularization, standard for fine-tuning transformers
optimizer = AdamW(model.parameters(), lr=2e-5)

# =============================================================================
# TRAINING LOOP
# =============================================================================
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

print(f"Starting training for {MODEL_NAME}...")

# LOOP EXPLANATION (OUTER): Iterate over total Epochs
for epoch in range(EPOCHS):
    # .train() enables Dropout layers and updates Batch Normalization statistics
    model.train()
    total_loss = 0
    
    # LOOP EXPLANATION (INNER): Iterate through the batched training pairs
    for input_ids, attn_mask, token_type_ids, labels in train_loader:
        # Move tensors to GPU VRAM
        input_ids = input_ids.to(device)
        attn_mask = attn_mask.to(device)
        token_type_ids = token_type_ids.to(device)
        labels = labels.to(device)
        
        # Flush the gradients from the previous batch to prevent accumulation
        optimizer.zero_grad()
        
        # 1. Forward pass: Calculate predictions
        outputs = model(input_ids, attn_mask, token_type_ids)
        
        # 2. Loss computation: Compare predictions to true labels
        loss = criterion(outputs, labels)
        
        # 3. Backward pass: Calculate gradients (partial derivatives) for all parameters
        loss.backward()
        
        # 4. Optimizer step: Update the model weights based on the gradients
        optimizer.step()
        
        total_loss += loss.item()
        
    avg_train_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {avg_train_loss:.4f}")

    # =========================================================================
    # VALIDATION PASS
    # =========================================================================
    # .eval() disables Dropout, ensuring deterministic predictions for validation
    model.eval()
    val_loss = 0
    
    # torch.no_grad() temporarily disables the autograd engine. 
    # This prevents the network from storing intermediate activations, 
    # drastically reducing memory usage and speeding up evaluation.
    with torch.no_grad():
        for input_ids, attn_mask, token_type_ids, labels in test_loader:
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)
            token_type_ids = token_type_ids.to(device)
            labels = labels.to(device)
            
            outputs = model(input_ids, attn_mask, token_type_ids)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
    avg_val_loss = val_loss / len(test_loader)
    print(f"Epoch {epoch+1}/{EPOCHS}, Validation Loss: {avg_val_loss:.4f}")

# Save the final model state dictionary to the dynamically assigned directory
model_save_file = os.path.join(MODEL_SAVE_PATH, "bert_regression_model.pt")
torch.save(model.state_dict(), model_save_file)
print(f"Training complete. Model saved to {model_save_file}")