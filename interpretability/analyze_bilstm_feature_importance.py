"""
NEUREQ Interpretability: Attention x Gradient Analysis
This script calculates the feature-level attribution for the Phase 2 BiLSTM. 
It identifies which of the 10 clinical eligibility questions most strongly 
drive the model's final patient-trial relevance predictions by combining 
additive attention weights with the gradient magnitudes of the input features.
"""

import os
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_DIR = "models_new/BiLSTM_ADD_ATTN_1196"
MODEL_FILE = os.path.join(MODEL_DIR, "model_epoch_12.pt")

INPUT_FILES = [
    "data/2021/WholeQ_RETRIEVAL_T2021_llm_responses_sanitized.jsonl",
    "data/2021/WholeQ_RM3_RETRIEVAL_T2021_llm_responses_sanitized.jsonl",
    "data/2022/WholeQ_RETRIEVAL_T2022_llm_responses_sanitized.jsonl",
    "data/2022/WholeQ_RM3_RETRIEVAL_T2022_llm_responses_sanitized.jsonl",
]

OUTPUT_DIR = os.path.join(MODEL_DIR, "feature_importance")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ANSWER_MAP = {"YES": 0, "NO": 1, "NA": 2}
QUESTION_NAMES = [f"Q{i}" for i in range(1, 11)]

# ============================================================
# CLINICALBERT (FROZEN)
# ============================================================

CLINICAL_BERT_MODEL = "emilyalsentzer/Bio_ClinicalBERT"
tokenizer = AutoTokenizer.from_pretrained(CLINICAL_BERT_MODEL)
bert_model = AutoModel.from_pretrained(CLINICAL_BERT_MODEL).to(DEVICE)
bert_model.eval()

@torch.no_grad()
def encode_justification(text: str):
    """
    Extracts the 768-dimensional [CLS] token embedding for a given text.
    Shape transformation: string -> (1, 768)
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
# MODEL DEFINITION (Must match training architecture exactly)
# ============================================================

class AdditiveAttention(nn.Module):
    """
    Computes a scalar attention weight for each time step (question).
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, H):
        # H shape: (Batch, Seq=10, Hidden=128)
        # scores shape: (Batch, Seq=10)
        scores = self.v(torch.tanh(self.attn(H))).squeeze(-1)
        
        # alpha shape: (Batch, Seq=10) - The normalized attention weights
        alpha = torch.softmax(scores, dim=1)
        
        # context shape: (Batch, Hidden=128) - The aggregated sequence representation
        context = torch.sum(alpha.unsqueeze(-1) * H, dim=1)
        return context, alpha


class EligibilityBiLSTM(nn.Module):
    """
    The 1-layer bidirectional LSTM scoring model.
    Modified slightly from training to allow gradient extraction of inputs.
    """
    def __init__(self):
        super().__init__()
        self.question_embed = nn.Embedding(10, 8)
        self.answer_embed = nn.Embedding(3, 3)

        self.bilstm = nn.LSTM(
            input_size=8 + 3 + 768, # Total feature dimension = 779
            hidden_size=64,         # Bidirectional output will be 64 * 2 = 128
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

    def forward(self, q_ids, a_ids, j_embs, return_input=False):
        """
        Args:
            return_input (bool): Critical flag for interpretability. 
                                 Forces PyTorch to keep `x` in the computation graph.
        """
        # Embed categorical inputs
        q_emb = self.question_embed(q_ids)      # Shape: (Batch=1, Seq=10, Dim=8)
        a_emb = self.answer_embed(a_ids)        # Shape: (Batch=1, Seq=10, Dim=3)

        # Concatenate into the final feature vector per question
        # Shape: (Batch=1, Seq=10, TotalDim=779)
        x = torch.cat([q_emb, a_emb, j_embs], dim=-1)  

        # === AUTOGRAD MANIPULATION ===
        # Standard PyTorch behavior deletes gradients of non-leaf tensors (like 'x') 
        # to free up GPU memory during the backward pass. We use retain_grad() to 
        # explicitly tell the autograd engine to store x's gradients in x.grad.
        if return_input:
            x.retain_grad()

        # Forward pass through BiLSTM and Attention
        H, _ = self.bilstm(x)                       # H shape: (Batch=1, Seq=10, Dim=128)
        context, attn_weights = self.attention(H)   # context shape: (1, 128), attn: (1, 10)
        logit = self.classifier(context).squeeze(-1)# logit shape: (Batch=1)

        if return_input:
            return logit, attn_weights, x
        else:
            return logit, attn_weights


# ============================================================
# LOAD MODEL
# ============================================================

model = EligibilityBiLSTM().to(DEVICE)
model.load_state_dict(torch.load(MODEL_FILE, map_location=DEVICE))
model.eval()


# ============================================================
# BUILD INPUT TENSORS
# ============================================================

def build_input(cleaned_output):
    """
    Parses the JSON and prepares the inputs for the neural network.
    Defaults missing or malformed questions to "NA".
    """
    question_ids = []
    answer_ids = []
    justification_embs = []

    for qid in range(1, 11):
        q = cleaned_output.get(str(qid), None)

        if q is None:
            response = "NA"
            just_text = ""
        else:
            response = q.get("response", "NA")
            just_text = q.get("justification", "")

        question_ids.append(qid - 1)
        answer_ids.append(ANSWER_MAP.get(response, 2))
        justification_embs.append(encode_justification(just_text))

    # Add the Batch dimension (size 1) required by PyTorch modules
    q_ids = torch.tensor(question_ids, dtype=torch.long).unsqueeze(0).to(DEVICE)
    a_ids = torch.tensor(answer_ids, dtype=torch.long).unsqueeze(0).to(DEVICE)
    j_embs = torch.stack(justification_embs).unsqueeze(0).to(DEVICE)

    return q_ids, a_ids, j_embs


# ============================================================
# ANALYZE SINGLE SAMPLE (THE CORE MECHANISTIC LOGIC)
# ============================================================

def analyze_sample(cleaned_output):
    """
    Executes the backward pass to compute the Attention x Gradient attribution.
    This quantifies exactly how much each of the 10 questions influenced the 
    final relevance score.
    """
    q_ids, a_ids, j_embs = build_input(cleaned_output)

    # === MODE SWITCHING HACK ===
    # cuDNN optimizes LSTM inference heavily. If model.eval() is active, cuDNN 
    # drops intermediate recurrent state activations to save VRAM. However, we NEED 
    # those activations to perform Backpropagation Through Time (BPTT). 
    # Switching to .train() disables this cuDNN optimization, allowing .backward() to run.
    model.train()
    
    # Flush any stale gradients from previous loops
    model.zero_grad()

    # Run the forward pass and intercept the input tensor `x`
    logit, attn_weights, x = model(
        q_ids,
        a_ids,
        j_embs,
        return_input=True
    )

    # Compute the gradients of the scalar `logit` with respect to all leaf nodes 
    # and any tensors marked with `retain_grad()` (which we did for `x`).
    logit.backward()

    # Re-enable evaluation mode to ensure Dropouts behave deterministically elsewhere
    model.eval()

    # --- Step 1: Extract Attention ---
    # Shape drops from (Batch=1, Seq=10) -> (10,)
    attn = attn_weights.squeeze(0).detach().cpu().numpy()

    # --- Step 2: Extract Gradients ---
    # x.grad contains the partial derivatives: d(logit) / d(x_features).
    # It tells us how much the final score would change if we perturbed the input embeddings.
    # Shape drops from (Batch=1, Seq=10, Features=779) -> (10, 779)
    grads = x.grad.squeeze(0).detach().cpu().numpy()

    # Negative gradients imply a feature pushes the score down; positive pushes it up.
    # We care about overall *magnitude* of influence, so we take the absolute value.
    grads_abs = np.abs(grads)

    # The 779 features (Question ID + YES/NO + BERT embedding) all belong to one question.
    # We sum the gradients across the feature dimension (axis=1) to collapse them 
    # into a single scalar representing the total gradient flow for that specific question.
    # Shape drops from (10, 779) -> (10,)
    grad_importance = grads_abs.sum(axis=1)

    # --- Step 3: Combine ---
    # Multiply the attention weight (where the model looked) by the gradient 
    # magnitude (how much changing the input actually affected the output).
    attn_grad = attn * grad_importance

    return {
        "attention": attn,
        "gradient": grad_importance,
        "attn_x_grad": attn_grad,
        "logit": logit.item()
    }


# ============================================================
# VISUALIZATION & AGGREGATION
# ============================================================

def save_bar_plot(values, title, out_path):
    """Generates a standard matplotlib bar chart for visual inspection."""
    plt.figure(figsize=(8, 4))
    plt.bar(QUESTION_NAMES, values)
    plt.ylabel("Importance")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def analyze_file(input_path):
    """Processes an entire test set and averages the attribution scores."""
    dataset_name = os.path.basename(input_path).replace(".jsonl", "")
    print(f"\nAnalyzing {dataset_name}")

    all_attention = []
    all_gradient = []
    all_attn_grad = []

    with open(input_path, "r", encoding="utf-8") as f:
        lines = list(f)

    # Loop through every patient-trial pair and run the backward pass
    for line in tqdm(lines, desc=dataset_name):
        item = json.loads(line)
        cleaned_output = item.get("cleaned_output", None)

        if cleaned_output is None:
            continue

        result = analyze_sample(cleaned_output)

        all_attention.append(result["attention"])
        all_gradient.append(result["gradient"])
        all_attn_grad.append(result["attn_x_grad"])

    if len(all_attention) == 0:
        print("No valid samples found.")
        return

    # Average the (10,) importance arrays across all N samples in the test set
    mean_attention = np.mean(all_attention, axis=0)
    mean_gradient = np.mean(all_gradient, axis=0)
    mean_attn_grad = np.mean(all_attn_grad, axis=0)

    # Normalize the final Attention x Gradient scores so they sum to 1.0.
    # This allows for easy percentage-based interpretation (e.g., "Feature X accounts for 45% of decision weight").
    mean_attn_grad = mean_attn_grad / mean_attn_grad.sum()

    # Save to CSV for reporting
    df = pd.DataFrame({
        "Question": QUESTION_NAMES,
        "Mean_Attention": mean_attention,
        "Mean_Gradient": mean_gradient,
        "Mean_Attention_x_Gradient": mean_attn_grad
    })

    df = df.sort_values(
        by="Mean_Attention_x_Gradient",
        ascending=False
    )

    csv_path = os.path.join(OUTPUT_DIR, f"{dataset_name}_importance.csv")
    df.to_csv(csv_path, index=False)

    plot_path = os.path.join(OUTPUT_DIR, f"{dataset_name}_importance.png")
    save_bar_plot(
        df["Mean_Attention_x_Gradient"].values,
        f"Feature Importance: {dataset_name}",
        plot_path
    )

    print("Top Questions:")
    print(df[["Question", "Mean_Attention_x_Gradient"]].head(10))
    print(f"Saved CSV : {csv_path}")
    print(f"Saved Plot: {plot_path}")

    return df


def aggregate_all(all_dfs):
    """Combines the scores from all 4 datasets into a single global importance metric."""
    merged = pd.DataFrame({"Question": QUESTION_NAMES})

    for name, df in all_dfs.items():
        tmp = df[["Question", "Mean_Attention_x_Gradient"]].rename(
            columns={"Mean_Attention_x_Gradient": name}
        )
        merged = merged.merge(tmp, on="Question")

    # Average the dataset-specific scores to get an overall global importance across years and retrieval methods
    score_cols = [c for c in merged.columns if c != "Question"]
    merged["Overall_Mean"] = merged[score_cols].mean(axis=1)
    merged = merged.sort_values("Overall_Mean", ascending=False)

    out_csv = os.path.join(OUTPUT_DIR, "overall_importance.csv")
    merged.to_csv(out_csv, index=False)

    out_plot = os.path.join(OUTPUT_DIR, "overall_importance.png")
    save_bar_plot(
        merged["Overall_Mean"].values,
        "Overall Question Importance",
        out_plot
    )

    print("\nOverall Importance Ranking")
    print(merged[["Question", "Overall_Mean"]])
    print(f"Saved overall CSV : {out_csv}")
    print(f"Saved overall plot: {out_plot}")


# ============================================================
# MAIN
# ============================================================

def main():
    all_dfs = {}

    for input_path in INPUT_FILES:
        df = analyze_file(input_path)
        if df is not None:
            dataset_name = os.path.basename(input_path).replace(".jsonl", "")
            all_dfs[dataset_name] = df

    if all_dfs:
        aggregate_all(all_dfs)


if __name__ == "__main__":
    main()