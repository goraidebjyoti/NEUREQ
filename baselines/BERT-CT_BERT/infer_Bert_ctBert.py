# infer_Bert_ctBert.py

"""
Baseline Model Inference: BERT & CT-BERT Cross-Encoders

Objective:
    Applies the trained deep learning baselines (from train_Bert_ctBert.py) 
    to the unseen test sets (TREC 2021 or 2022). It scores the patient-trial 
    candidate pairs retrieved by the first-stage models and outputs standard 
    TREC-formatted run files for statistical evaluation.

Usage:
    Update the `TRAINED_MODEL_CHECKPOINT` to point to the exact epoch weight 
    file that achieved the lowest Validation Loss during training.
"""

import os
import pandas as pd
import torch
import torch.nn as nn
import json
from tqdm import tqdm
from transformers import BertTokenizer, BertModel

# =============================================================================
# CONFIGURATION & HARDCODED VARIABLES
# =============================================================================
YEAR = 2022 # Toggle between 2021 and 2022 test sets

# First-stage baseline retrieval files to be re-ranked
RUN_FILES = [ 
    f"WholeQ_RETRIEVAL_T{YEAR}.txt", 
    f"WholeQ_RM3_RETRIEVAL_T{YEAR}.txt"
]
FIRST_STAGE_RUN_DIR = f"runs/{YEAR}/FIRST_STAGE"

# Toggle this depending on whether you are running standard BERT or CT-BERT
RUN_NAME = "CT_MLM_BERT"
OUTPUT_DIR = f"runs0/{YEAR}/{RUN_NAME}"

# Raw data paths needed to reconstruct the text strings for inference
CORPUS_PATH = "data/clinicaltrials/corpus.jsonl"
QUERY_PATH = f"data/{YEAR}/ct_{YEAR}_queries.tsv"

# The base HuggingFace model used for tokenization
BASE_CHECKPOINT = "ielabgroup/PubMedBERT-CT-MLM" 

# --- CRITICAL: BEST MODEL SELECTION ---
# You must manually review the training logs from `train_Bert_ctBert.py`. 
# Identify the epoch that yielded the lowest Validation Loss before overfitting began,
# and point this variable to that specific `.pt` file.
TRAINED_MODEL_CHECKPOINT = "models0/CT_MLM_BERT/bert_regression_epoch4.pt"

# Fixed truncation limits (Must identically match the training script)
SEED = 42
MAX_QUERY_LEN = 179
MAX_DOC_LEN = 330
# +3 accounts for the mandatory special tokens: [CLS] Query [SEP] Doc [SEP]
MAX_LEN = MAX_QUERY_LEN + MAX_DOC_LEN + 3 

# === Hardware & Reproducibility ===
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Load Tokenizer ===
tokenizer = BertTokenizer.from_pretrained(BASE_CHECKPOINT)

# =============================================================================
# MODEL DEFINITION (Must match training architecture)
# =============================================================================
class BERTRegression(nn.Module):
    """
    Standard Cross-Encoder architecture predicting a continuous relevance score.
    """
    def __init__(self):
        super(BERTRegression, self).__init__()
        self.bert = BertModel.from_pretrained(BASE_CHECKPOINT)
        
        # In this inference script, the regressor is expanded into a multi-layer 
        # perceptron with a Sigmoid activation. This explicitly squashes the 
        # final logit into a normalized probability bound between 0.0 and 1.0.
        self.regressor = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid() 
        )

    def forward(self, input_ids, attention_mask, token_type_ids):
        # Pass the sequence through the transformer
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        
        # Isolate the contextualized [CLS] token representing the whole sequence
        cls_output = outputs.last_hidden_state[:, 0, :]
        
        # Return the squashed relevance scalar
        return self.regressor(cls_output).squeeze()

# === Load Model Weights ===
model = BERTRegression().to(device)
# Inject the fine-tuned parameters from disk into the architecture
model.load_state_dict(torch.load(TRAINED_MODEL_CHECKPOINT, map_location=device))

# .eval() disables dropout to ensure predictions are deterministic
model.eval()
print(f"Model loaded from {TRAINED_MODEL_CHECKPOINT}")


# =============================================================================
# DATA PROCESSING UTILITIES
# =============================================================================
def tokenize_input(query, doc):
    """
    Manually constructs the Cross-Encoder input tensors.
    Unlike training where we used HuggingFace's automatic joint tokenization, 
    this explicitly manages truncation and padding to ensure strict compliance 
    with the 512 token budget.
    """
    # Force strict truncation independently to ensure neither the query 
    # nor the document cannibalizes the other's token budget.
    query_tokens = tokenizer.tokenize(query)[:MAX_QUERY_LEN]
    doc_tokens = tokenizer.tokenize(doc)[:MAX_DOC_LEN]

    # Stitch the sequence together with structural delimiters
    tokens = tokenizer.convert_tokens_to_ids(['[CLS]'] + query_tokens + ['[SEP]'] + doc_tokens + ['[SEP]'])
    
    # segment_ids (token_type_ids): 0 for Query, 1 for Document
    segment_ids = [0] * (len(query_tokens) + 2) + [1] * (len(doc_tokens) + 1)
    
    # attention_mask: 1 for real text, 0 for padding
    attention_mask = [1] * len(tokens)

    # Apply right-padding to ensure the tensor hits exactly MAX_LEN (512)
    padding_length = MAX_LEN - len(tokens)
    tokens += [0] * padding_length
    segment_ids += [0] * padding_length
    attention_mask += [0] * padding_length

    # Package as PyTorch tensors with a batch dimension of 1
    return {
        'input_ids': torch.tensor(tokens).unsqueeze(0).to(device),
        'attention_mask': torch.tensor(attention_mask).unsqueeze(0).to(device),
        'token_type_ids': torch.tensor(segment_ids).unsqueeze(0).to(device)
    }

def load_corpus(path):
    """Loads the massive trial JSONL corpus into memory for instant lookup."""
    corpus_dict = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            trial = json.loads(line.strip())
            corpus_dict[trial["id"]] = trial.get("contents")
    return corpus_dict

corpus = load_corpus(CORPUS_PATH)
print(f"Loaded {len(corpus)} documents from corpus.")

# Load Test Queries
df_queries = pd.read_csv(QUERY_PATH, sep="\t", header=None, names=["id", "text"])
query_dict = dict(zip(df_queries["id"].astype(str), df_queries["text"]))

# === Core Scoring Function ===
def score(query, doc):
    """Transforms text to tensors and predicts relevance via the neural network."""
    model_input = tokenize_input(query, doc)
    
    # Disable autograd to save VRAM and drastically speed up inference
    with torch.no_grad():
        output = model(**model_input)
    
    return output.item()


# =============================================================================
# MAIN INFERENCE LOOP
# =============================================================================

# Iterate through the multiple test environments (e.g., BM25 alone vs BM25+RM3)
for run_file in RUN_FILES:
    print(f"\nProcessing {run_file}...")
    retrieved_trials_file = os.path.join(FIRST_STAGE_RUN_DIR, run_file)
    output_file_name = f"{os.path.splitext(run_file)[0]}.txt"
    output_file_path = os.path.join(OUTPUT_DIR, output_file_name)

    # --- 1. Load First-Stage Candidate List ---
    bm25_results = {}
    with open(retrieved_trials_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            topic_no, trial_id = parts[0], parts[2]
            bm25_results.setdefault(topic_no, []).append(trial_id)

    # --- 2. Predict and Re-rank ---
    text_lines = []
    
    # Iterate over every patient case description
    for topic_no, trials in tqdm(bm25_results.items(), desc=f"Reranking {run_file}"):
        query = query_dict.get(topic_no)
        if not query:
            continue

        scored_trials = []
        
        # Iterate over the top 100 candidate trials retrieved for this specific patient
        for trial_id in trials:
            doc_text = corpus.get(trial_id)
            if not doc_text:
                continue
            
            # Predict cross-encoder relevance
            score_val = score(query, doc_text)
            scored_trials.append((trial_id, score_val))

        # Sort descending based on the neural model's prediction
        scored_trials.sort(key=lambda x: x[1], reverse=True)

        # --- 3. Format Output ---
        # Generate the strict space-delimited format required by trec_eval:
        # `query_id Q0 document_id rank relevance_score run_name`
        for rank, (trial_id, score_val) in enumerate(scored_trials[:100], start=1):
            line = f"{topic_no} Q0 {trial_id} {rank} {score_val:.5f} {RUN_NAME}"
            text_lines.append(line)

    # Write the entire evaluated set to disk
    with open(output_file_path, "w") as fout:
        fout.write("\n".join(text_lines) + "\n")

    print(f"Saved reranked run to {output_file_path}")