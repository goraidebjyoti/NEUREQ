import os
import json
import torch
import torch.nn as nn
from tqdm import tqdm
from collections import defaultdict

# ==================== CONFIG ====================
YEAR = 2022  # 🔧 Change this to 2021 or 2022
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

FILE_NAMES = [
    f"WholeQ_RETRIEVAL_T{YEAR}",
    f"WholeQ_RM3_RETRIEVAL_T{YEAR}"
]

RESULT_FILES = [f"data/{YEAR}/{fname}_llm_responses_sanitized.jsonl" for fname in FILE_NAMES] #LLM Sanitised resposnses jsonl file

OUTPUT_DIR = f"runs/{YEAR}/NEUREQ_1196"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Pick the best checkpoint manually (example: epoch20)
MODEL_PATH = "models_new/LSTM_1196/lstm_1layer_epoch20.pt"

RESPONSE_MAP = {"YES": 1, "NO": 0, "NA": 0.5}
DEFAULT_SCORE = -100.0

INPUT_DIM = 1
HIDDEN_SIZE = 64
SEQUENCE_LENGTH = 10

# ==================== MODEL CLASS ====================
class LSTMModel(nn.Module):
    def __init__(self, num_layers):
        super().__init__()
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
        out, _ = self.lstm(x)  # (B, T, H)
        return self.fc(out[:, -1, :])  # use last hidden state

# ==================== LOAD MODEL ====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"💻 Using device: {device}")

print("📦 Loading NEUREQ (LSTM-1L)...")
model = LSTMModel(num_layers=1).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# ==================== SCORE FUNCTION ====================
def get_score(model, entry):
    try:
        answers = [
            RESPONSE_MAP.get(entry["cleaned_output"].get(str(i), {}).get("response", "NA").upper(), -1)
            for i in range(1, 11)
        ]
        input_tensor = torch.tensor(answers, dtype=torch.float32).view(1, 10, 1).to(device)
        with torch.no_grad():
            return model(input_tensor).item()
    except Exception:
        return DEFAULT_SCORE

# ==================== INFERENCE ====================
for fname, path in zip(FILE_NAMES, RESULT_FILES):
    print(f"\n🔍 Processing {path}")
    scores = defaultdict(list)

    with open(path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc=fname):
            obj = json.loads(line)
            qid = obj.get("qid")
            docid = obj.get("docid")
            if not qid or not docid:
                continue
            score = get_score(model, obj)
            scores[qid].append((docid, score))

    # === Save in TREC format ===
    out_path = os.path.join(OUTPUT_DIR, f"{fname}_NEUREQ.txt")
    with open(out_path, "w") as outf:
        for qid, doc_scores in scores.items():
            ranked = sorted(doc_scores, key=lambda x: x[1], reverse=True)
            for rank, (docid, score) in enumerate(ranked, start=1):
                outf.write(f"{qid} Q0 {docid} {rank} {score:.6f} NEUREQ\n")
    print(f"✅ Saved: {out_path}")
