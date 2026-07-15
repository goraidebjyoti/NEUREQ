import json
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

# -----------------
# Config
# -----------------
MODEL_DIR = "models/sct_bert"   # trained SCT-BERT checkpoint
TOPICS_FILE = "data/2021/ct_2021_queries.tsv"   # <topic_id>\t<query>
CANDIDATE_RUN = "data/2021/WholeQ_RM3_RETRIEVAL_T2021.txt"   # TREC format
SUMMARY_FILE = "data/clinicaltrials/positive_pairs.jsonl"  # has trial summaries
OUTPUT_RUN = "runs/2021/sctbert/WholeQ_RM3_T2021.txt"

BATCH_SIZE = 32
MAX_LEN = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------
# Load Topics
# -----------------
print("Loading topics...")
topics = {}
with open(TOPICS_FILE, "r") as f:
    for line in f:
        tid, query = line.strip().split("\t", 1)
        topics[tid] = query

print(f"Loaded {len(topics)} topics.")

# -----------------
# Load Summaries (trial_id → summary)
# -----------------
print("Loading trial summaries from positive_pairs.jsonl...")
trial_summaries = {}
with open(SUMMARY_FILE, "r") as f:
    for line in f:
        rec = json.loads(line)
        trial_summaries[rec["trial_id_summary"]] = rec["summary"]

print(f"Loaded {len(trial_summaries)} trial summaries.")

# -----------------
# Load Candidate Run
# -----------------
print("Loading candidate run...")
candidates = {}  # topic_id -> list of trial_ids
with open(CANDIDATE_RUN, "r") as f:
    for line in f:
        topic_id, _, trial_id, _, _, _ = line.strip().split()
        candidates.setdefault(topic_id, []).append(trial_id)

print(f"Loaded candidates for {len(candidates)} topics.")

# -----------------
# Dataset for Inference
# -----------------
class TrialDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_len=512):
        self.pairs = pairs  # list of (summary, query, topic_id, trial_id)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        summary, query, topic_id, trial_id = self.pairs[idx]
        encoding = self.tokenizer(
            summary,
            query,     # query replaces criteria
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item["topic_id"] = topic_id
        item["trial_id"] = trial_id
        return item

# -----------------
# Build Pair List
# -----------------
print("Building dataset for reranking...")
pairs = []
missing = 0
for topic_id, trial_ids in candidates.items():
    query = topics[topic_id]
    for trial_id in trial_ids:
        if trial_id not in trial_summaries:
            missing += 1
            continue
        pairs.append((trial_summaries[trial_id], query, topic_id, trial_id))

if missing > 0:
    print(f"⚠️ Warning: {missing} candidate trials not found in summaries.")

# -----------------
# Tokenizer & Model
# -----------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(DEVICE)
model.eval()

# -----------------
# DataLoader
# -----------------
dataset = TrialDataset(pairs, tokenizer, max_len=MAX_LEN)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)

# -----------------
# Inference
# -----------------
print("Scoring candidates...")
scores = {}
with torch.no_grad():
    for batch in tqdm(dataloader, desc="Inference"):
        input_ids = batch["input_ids"].to(DEVICE)
        attn_mask = batch["attention_mask"].to(DEVICE)

        outputs = model(input_ids=input_ids, attention_mask=attn_mask)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)[:, 1]  # probability of label=1

        for topic_id, trial_id, score in zip(batch["topic_id"], batch["trial_id"], probs.cpu().tolist()):
            scores.setdefault(topic_id, []).append((trial_id, score))

# -----------------
# Write Reranked Run
# -----------------
print("Writing reranked run...")
with open(OUTPUT_RUN, "w") as fout:
    for topic_id, trial_scores in scores.items():
        trial_scores = sorted(trial_scores, key=lambda x: x[1], reverse=True)
        for rank, (trial_id, score) in enumerate(trial_scores, start=1):
            fout.write(f"{topic_id} Q0 {trial_id} {rank} {score:.6f} SCTBERT\n")

print(f"✅ Reranked run written to {OUTPUT_RUN}")
