# sctbert_4.py

"""
Baseline Model: SCT-BERT (Step 4/4) - Zero-Shot Inference & Re-ranking

Objective:
    This script executes the evaluation phase for the SCT-BERT (Self-Supervised MonoBERT) baseline. 
    It ingests a candidate list of clinical trials retrieved by a first-stage lexical algorithm 
    (like BM25+RM3) and re-ranks them using the fine-tuned neural network.

    Crucial Context: Zero-Shot Transfer
    During training (Step 3), the model learned to map (Trial Summary -> Trial Criteria). 
    In this script, we perform a zero-shot domain shift, swapping the inputs to 
    (Trial Summary -> Patient Query). Because the model learned the underlying "language" 
    of clinical relevance, it can score how well the trial summary matches the patient case.
"""

import json
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import os

# =============================================================================
# CONFIGURATION & FILE PATHS
# =============================================================================
# --- Inputs ---
# The directory containing the weights fine-tuned in Step 3
MODEL_DIR = "models/sct_bert"   

# The official TREC queries for the evaluation year
TOPICS_FILE = "data/2021/ct_2021_queries.tsv"   

# The first-stage candidate lists we need to evaluate (e.g., top 100 BM25 trials per query)
CANDIDATE_RUN = "data/2021/WholeQ_RM3_RETRIEVAL_T2021.txt"   

# The parsed trials generated in Step 1, used here as a fast lookup dictionary for trial summaries
SUMMARY_FILE = "data/clinicaltrials/positive_pairs.jsonl"  

# --- Output ---
# The final TREC-formatted file ready for statistical evaluation via `trec_eval`
OUTPUT_RUN = "runs/2021/sctbert/WholeQ_RM3_T2021.txt"

# --- Hyperparameters ---
BATCH_SIZE = 32
MAX_LEN = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ensure the output directory exists before processing
os.makedirs(os.path.dirname(OUTPUT_RUN), exist_ok=True)

# =============================================================================
# DATA LOADING (In-Memory Lookups)
# =============================================================================
print("Loading topics...")
# Load patient case descriptions into a dictionary { topic_id: query_text }
topics = {}
with open(TOPICS_FILE, "r") as f:
    for line in f:
        tid, query = line.strip().split("\t", 1)
        topics[tid] = query
print(f"Loaded {len(topics)} topics.")

print("Loading trial summaries from positive_pairs.jsonl...")
# Load the trial summaries extracted in Step 1 to avoid parsing the raw corpus again
trial_summaries = {}
with open(SUMMARY_FILE, "r") as f:
    for line in f:
        rec = json.loads(line)
        # Store as { trial_id: brief_summary_text }
        trial_summaries[rec["trial_id_summary"]] = rec["summary"]
print(f"Loaded {len(trial_summaries)} trial summaries.")

print("Loading candidate run...")
# Parse the first-stage retrieval file. 
# We build a dictionary mapping a patient topic to its list of retrieved candidate trials.
candidates = {}  
with open(CANDIDATE_RUN, "r") as f:
    for line in f:
        topic_id, _, trial_id, _, _, _ = line.strip().split()
        candidates.setdefault(topic_id, []).append(trial_id)
print(f"Loaded candidates for {len(candidates)} topics.")


# =============================================================================
# PYTORCH DATASET (Formatting for Inference)
# =============================================================================
class TrialDataset(Dataset):
    """
    Custom Dataset to format the (Summary, Patient Query) pairs into tensor batches.
    """
    def __init__(self, pairs, tokenizer, max_len=512):
        self.pairs = pairs  # Expected format: list of (summary, query, topic_id, trial_id)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        summary, query, topic_id, trial_id = self.pairs[idx]
        
        # DOMAIN SHIFT TOKENIZATION:
        # We pass the trial's 'summary' as sequence A, and the patient's 'query' as sequence B.
        # Format: [CLS] Trial Summary [SEP] Patient Query [SEP]
        encoding = self.tokenizer(
            summary,
            query,     
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Squeeze removes the redundant batch dimension added by the tokenizer
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        
        # Attach IDs so we can map the neural score back to the correct output line later
        item["topic_id"] = topic_id
        item["trial_id"] = trial_id
        return item


# =============================================================================
# BUILD INFERENCE DATASET
# =============================================================================
print("Building dataset for reranking...")
pairs = []
missing = 0

# LOOP EXPLANATION: Iterate through the candidate lists
for topic_id, trial_ids in candidates.items():
    query = topics[topic_id]
    
    # Evaluate every candidate trial retrieved for this patient
    for trial_id in trial_ids:
        # If the trial lacks a summary, we cannot score it.
        if trial_id not in trial_summaries:
            missing += 1
            continue
        # Package the raw data for the Dataset class
        pairs.append((trial_summaries[trial_id], query, topic_id, trial_id))

if missing > 0:
    print(f"⚠️ Warning: {missing} candidate trials not found in summaries. These will be dropped.")

# =============================================================================
# INITIALIZE MODEL
# =============================================================================
# Load the weights we fine-tuned in baseline_sctbert_3.py
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(DEVICE)

# .eval() disables dropout and batch normalization tracking, ensuring deterministic outputs
model.eval()

# DataLoader handles batching the tensors automatically
dataset = TrialDataset(pairs, tokenizer, max_len=MAX_LEN)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)


# =============================================================================
# INFERENCE LOOP
# =============================================================================
print("Scoring candidates...")
# Dictionary to store the neural scores: { topic_id: [(trial_id, score), ...] }
scores = {}

# Disable the autograd engine. We are not calculating derivatives, so this saves 
# massive amounts of VRAM and speeds up prediction significantly.
with torch.no_grad():
    for batch in tqdm(dataloader, desc="Inference"):
        input_ids = batch["input_ids"].to(DEVICE)
        attn_mask = batch["attention_mask"].to(DEVICE)

        # Forward pass through the network
        outputs = model(input_ids=input_ids, attention_mask=attn_mask)
        
        # Extract the raw, unnormalized network outputs (Logits)
        logits = outputs.logits
        
        # Apply the Softmax function along the class dimension to convert logits 
        # into normalized probabilities (0.0 to 1.0).
        # [:, 1] isolates the probability for Class 1 (Relevant/Match).
        probs = torch.softmax(logits, dim=-1)[:, 1]  

        # Map the batch tensor probabilities back to their respective string IDs
        for topic_id, trial_id, score in zip(batch["topic_id"], batch["trial_id"], probs.cpu().tolist()):
            scores.setdefault(topic_id, []).append((trial_id, score))

# =============================================================================
# EXPORT TO TREC FORMAT
# =============================================================================
print("Writing reranked run...")
with open(OUTPUT_RUN, "w") as fout:
    # Iterate over the scored topics
    for topic_id, trial_scores in scores.items():
        # Sort the candidate list purely by the SCT-BERT probability score (Descending)
        trial_scores = sorted(trial_scores, key=lambda x: x[1], reverse=True)
        
        # Write the top trials back out to the official format for evaluation
        for rank, (trial_id, score) in enumerate(trial_scores, start=1):
            # Standard TREC formatting: query_id iteration document_id rank score run_name
            fout.write(f"{topic_id} Q0 {trial_id} {rank} {score:.6f} SCTBERT\n")

print(f"✅ Reranked run written to {OUTPUT_RUN}")