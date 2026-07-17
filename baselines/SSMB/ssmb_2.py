"""
Baseline: Self-Supervised MonoBERT Inference & Score Interpolation (Step 2)
This script applies the fine-tuned SciBERT model from Step 1 to re-rank an initial 
list of candidate trials retrieved by BM25. Unlike other baselines that use purely 
neural scoring, this script implements a score interpolation layout that linearly 
combines normalized first-stage lexical scores (10%) with semantic sequence 
classification scores (90%) to calculate a final relevance ranking.
"""

import os
import json
import torch
import numpy as np
import re
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# =============================================================================
# CONFIGURATION & PATHS
# =============================================================================
# Path to the specific fine-tuned checkpoint folder saved in Step 1
MODEL_WEIGHTS = "models/self_supervised_monobert/epoch-1"  

# Input Files
TOPICS_FILE = "data/clinicaltrials/ct_2021_queries.tsv"  # Structured patient topic statements
CORPUS_FILE = "data/clinicaltrials/corpus.jsonl"         # Global document collection

# First-stage retrieval run (Lexical BM25 ranking list to be re-ranked)
BM25_FILE = "data/2021/WholeQ_RM3_RETRIEVAL_T2021.txt"

# Output Destination
OUTPUT_FILE = "runs/2021/SSMONOBERT_baseline/WHOLEQ_RM3_RETRIEVAL_2021.txt"

# Runtime Environment Hyperparameters
BATCH_SIZE = 16
MAX_LENGTH = 256  # Truncation boundary tailored for compact 'Brief Summary' sections
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ensure the parent directory tree for the final run file is safely instantiated
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# =============================================================================
# MODEL INITIALIZATION
# =============================================================================
print("🔹 Loading MonoBERT model from fine-tuned weights...")

# Pull weights from the designated local checkpoint rather than the global model hub
tokenizer = AutoTokenizer.from_pretrained(MODEL_WEIGHTS)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_WEIGHTS, num_labels=2)
model.to(DEVICE)
model.eval()  # Strictly freeze layers and deactivate dropout parameters

# =============================================================================
# DATA LOADING
# =============================================================================
print("🔹 Loading topics...")
topics = {}
with open(TOPICS_FILE, "r") as f:
    for line in f:
        # Unpack the tab-delimited query identifier and text narrative
        tid, text = line.strip().split("\t", 1)
        topics[tid] = text

print("🔹 Loading corpus and extracting summaries...")
corpus = {}
with open(CORPUS_FILE, "r") as f:
    for line in f:
        doc = json.loads(line)
        # Apply a regular expression to cleanly slice out the 'Brief Summary' subsection
        match = re.search(r"Brief Summary:(.*?)(?:\n[A-Z][a-z]+:|\Z)", doc["contents"], re.S)
        if match:
            corpus[doc["id"]] = match.group(1).strip()
        else:
            # Fallback routine: load the absolute document if specific block boundary is absent
            corpus[doc["id"]] = doc["contents"]  

print("🔹 Loading BM25 rankings...")
bm25_runs = {}
with open(BM25_FILE, "r") as f:
    for line in f:
        # Standard 6-column space-delimited TREC execution row parsing
        qid, _, docid, rank, score, _ = line.strip().split()
        if qid not in bm25_runs:
            bm25_runs[qid] = []
        # Group candidates under their respective Query IDs alongside their float scores
        bm25_runs[qid].append((docid, float(score)))

# =============================================================================
# RERANKING & INTERPOLATION
# =============================================================================
print("🔹 Starting reranking...")
with open(OUTPUT_FILE, "w") as fout:
    for qid in tqdm(bm25_runs.keys(), desc="Queries"):
        query_text = topics[qid]

        # Extract the specific sub-arrays for the targeted query context
        doc_ids, bm25_scores = zip(*bm25_runs[qid])
        bm25_scores = np.array(bm25_scores)

        # Min-Max Normalization: Rescale BM25 raw scores strictly to a [0, 1] distribution
        bm25_min, bm25_max = bm25_scores.min(), bm25_scores.max()
        bm25_norm = (bm25_scores - bm25_min) / (bm25_max - bm25_min + 1e-8)

        bert_scores = []

        # Stream candidate trial blocks through the classification model in mini-batches
        for i in range(0, len(doc_ids), BATCH_SIZE):
            batch_ids = doc_ids[i:i + BATCH_SIZE]
            texts = [corpus[docid] for docid in batch_ids]

            # Cross-Encoder Style input alignment: Sentence A = Summary, Sentence B = Query
            encodings = tokenizer(
                texts,                          
                [query_text] * len(texts),      
                truncation=True,
                padding=True,
                max_length=MAX_LENGTH,
                return_tensors="pt"
            ).to(DEVICE)

            # Prevent gradient calculation graph allocation to lower GPU memory footprints
            with torch.no_grad():
                outputs = model(**encodings)
                # Apply softmax over dim=1 and extract the positive relevance index (Column 1)
                probs = torch.softmax(outputs.logits, dim=1)[:, 1]  
                bert_scores.extend(probs.cpu().numpy())

        bert_scores = np.array(bert_scores)

        # ---------------------------------------------------------------------
        # SCORE INTERPOLATION
        # Linearly interpolate the normalized lexical baseline and neural scores
        # Formula: Final = (0.1 * BM25_normalized) + (0.9 * BERT_probability)
        # ---------------------------------------------------------------------
        final_scores = 0.1 * bm25_norm + 0.9 * bert_scores

        # Rank all processed trials descending based on the combined score valuation
        reranked = sorted(zip(doc_ids, final_scores), key=lambda x: x[1], reverse=True)

        # Export entries into a standard 6-column TREC run template
        for rank, (docid, score) in enumerate(reranked, start=1):
            fout.write(f"{qid} Q0 {docid} {rank} {score:.6f} SSMonoBERT\n")

print(f"✅ Reranking complete. Results saved to {OUTPUT_FILE}")