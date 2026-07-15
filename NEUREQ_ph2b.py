"""
NEUREQ Phase 2b: Neural Inference and TREC Formatting
This script loads the trained BiLSTM re-ranker, scores the sanitized Patient-Trial 
pairs, and outputs the final rankings in standard TREC format for evaluation.
"""

import os
import json
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from collections import defaultdict

# ============================================================
# CONFIGURATION
# ============================================================

# ----- MODEL -----
# Hardcoded path pointing to the selected best checkpoint from Phase 2a
MODEL_DIR = "models_new/BiLSTM_ADD_ATTN_1196"
MODEL_FILE = os.path.join(MODEL_DIR, "model_epoch_12.pt")

# ----- INPUT FILES -----
# The sanitized JSONL files containing the structured 10-question evaluations.
# These encompass both the 2021 and 2022 TREC datasets under two retrieval settings.
INPUT_FILES = [
    "data/2021/WholeQ_RETRIEVAL_T2021_llm_responses_sanitized.jsonl",
    "data/2021/WholeQ_RM3_RETRIEVAL_T2021_llm_responses_sanitized.jsonl",
    "data/2022/WholeQ_RETRIEVAL_T2022_llm_responses_sanitized.jsonl",
    "data/2022/WholeQ_RM3_RETRIEVAL_T2022_llm_responses_sanitized.jsonl",
]

# ----- OUTPUT RUN FILES -----
# Directory to store the final evaluated run files
OUTPUT_RUN_DIR = "models_new/BiLSTM_ADD_ATTN_1196/epoch12"
os.makedirs(OUTPUT_RUN_DIR, exist_ok=True)

RUN_NAME = "BiLSTM_ADD_ATTN_1196" # Identifier used in the TREC evaluation file

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# CLINICALBERT (FROZEN TEXT ENCODER)
# ============================================================

# Load the identical frozen Bio_ClinicalBERT instance used during training
# to ensure the textual justification embeddings perfectly match the learned space.
CLINICAL_BERT_MODEL = "emilyalsentzer/Bio_ClinicalBERT"
tokenizer = AutoTokenizer.from_pretrained(CLINICAL_BERT_MODEL)
bert_model = AutoModel.from_pretrained(CLINICAL_BERT_MODEL).to(DEVICE)
bert_model.eval()

@torch.no_grad()
def encode_justification(text: str):
    """
    Encodes the textual justification into a 768-dimensional semantic vector.
    """
    inputs = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    ).to(DEVICE)

    outputs = bert_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze(0)

# ============================================================
# MODEL DEFINITION (MUST STRICTLY MATCH TRAINING ARCHITECTURE)
# ============================================================

ANSWER_MAP = {"YES": 0, "NO": 1, "NA": 2}

class AdditiveAttention(nn.Module):
    """Additive attention layer to weight question importance."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, H):
        scores = self.v(torch.tanh(self.attn(H))).squeeze(-1)
        alpha = torch.softmax(scores, dim=1)
        context = torch.sum(alpha.unsqueeze(-1) * H, dim=1)
        return context, alpha

class EligibilityBiLSTM(nn.Module):
    """The 1-layer bidirectional LSTM scoring model."""
    def __init__(self):
        super().__init__()
        self.question_embed = nn.Embedding(10, 8)
        self.answer_embed = nn.Embedding(3, 3)

        self.bilstm = nn.LSTM(
            input_size=8 + 3 + 768,
            hidden_size=64,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )

        self.attention = AdditiveAttention(128)

        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, q_ids, a_ids, j_embs):
        q_emb = self.question_embed(q_ids)
        a_emb = self.answer_embed(a_ids)

        x = torch.cat([q_emb, a_emb, j_embs], dim=-1)
        H, _ = self.bilstm(x)

        context, _ = self.attention(H) # We discard attention weights during pure inference
        logit = self.classifier(context).squeeze(-1)
        return logit

# ============================================================
# LOAD TRAINED MODEL
# ============================================================

# Instantiate model architecture and inject the learned weights
model = EligibilityBiLSTM().to(DEVICE)
model.load_state_dict(torch.load(MODEL_FILE, map_location=DEVICE))
model.eval() # Explicitly set to evaluation mode to disable dropout

# ============================================================
# SCORING FUNCTION
# ============================================================

@torch.no_grad()
def score_trial(cleaned_output):
    """
    Parses a single patient-trial eligibility evaluation and returns a relevance score.
    
    Args:
        cleaned_output (dict): The sanitized 10-question evaluation payload.
        
    Returns:
        float: The raw relevance logit predicted by the BiLSTM.
    """
    # FAULT TOLERANCE: If Phase 1b marked the output as None (due to severe LLM 
    # formatting failure), assign an extreme negative score. This safely pushes 
    # the trial to the absolute bottom of the ranked list rather than crashing.
    if cleaned_output is None:
        return -1e6

    question_ids = []
    answer_ids = []
    justification_embs = []

    # Iterate through the expected 10-question schema
    for qid in range(1, 11):
        q = cleaned_output.get(str(qid), None)

        # Handle missing individual questions by defaulting to "NA"
        if q is None:
            response = "NA"
            just_text = ""
        else:
            response = q.get("response", "NA")
            just_text = q.get("justification", "")

        question_ids.append(qid - 1)
        answer_ids.append(ANSWER_MAP.get(response, 2)) # Default to 2 (NA) if unexpected string
        justification_embs.append(encode_justification(just_text))

    # Add batch dimension (unsqueeze) and move tensors to GPU
    q_ids = torch.tensor(question_ids, dtype=torch.long).unsqueeze(0).to(DEVICE)
    a_ids = torch.tensor(answer_ids, dtype=torch.long).unsqueeze(0).to(DEVICE)
    j_embs = torch.stack(justification_embs).unsqueeze(0).to(DEVICE)

    # Compute final score
    logit = model(q_ids, a_ids, j_embs)
    return logit.item()

# ============================================================
# RERANKING AND EXPORT
# ============================================================

def rerank_file(input_path, output_path):
    """
    Scores all candidate trials for every query and exports the results 
    in the strict TREC file format required by trec_eval.
    """
    results = defaultdict(list)

    # 1. Score all query-document pairs in the test set
    with open(input_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc=f"Scoring {os.path.basename(input_path)}"):
            item = json.loads(line)

            qid = item["qid"]
            docid = item["docid"]
            cleaned_output = item.get("cleaned_output", None)

            score = score_trial(cleaned_output)
            results[qid].append((docid, score))

    # 2. WRITE PURE TREC FORMAT (NO HEADERS)
    # The TREC format is strictly: `query_id iter document_id rank score run_id`
    with open(output_path, "w", encoding="utf-8") as out:
        # Iterate over each patient query
        for qid in sorted(results.keys()):
            # Sort the trials for this query by the predicted logit score (descending)
            ranked = sorted(results[qid], key=lambda x: x[1], reverse=True)
            
            # Enumerate generates the rank column directly
            for rank, (docid, score) in enumerate(ranked, start=1):
                # 'Q0' is a legacy required field generally ignored by modern evaluators
                out.write(
                    f"{qid} Q0 {docid} {rank} {score:.6f} {RUN_NAME}\n"
                )

# ============================================================
# MAIN
# ============================================================

def main():
    """
    Iterates through the defined test datasets and generates ranked TREC outputs.
    """
    for input_path in INPUT_FILES:
        output_path = os.path.join(
            OUTPUT_RUN_DIR,
            os.path.basename(input_path).replace(".jsonl", ".reranked.txt")
        )
        rerank_file(input_path, output_path)

if __name__ == "__main__":
    main()