# infer_routesae.py

"""
NEUREQ Interpretability: RouteSAE Inference & Analysis Script

Objective:
    Applies the trained Route Sparse Autoencoder (RouteSAE) to an unseen test set 
    (e.g., the TREC 2022 retrieval run). 
    
    This script extracts the hidden states from the LLM for every retrieved 
    patient-trial pair, maps them into the sparse RouteSAE feature space, and 
    records exactly WHICH latent features fired. By cross-referencing these 
    activations with the ground-truth QRELs, we can analyze the model's 
    clinical reasoning on the test set.

Outputs:
    - all_z.pt: The raw sparse feature activations.
    - active_features.json: A highly detailed, human-readable log mapping 
      each patient-trial pair to its top 20 activating latent features.
    - feature_summary.json: Global discriminability metrics for the test set.
"""

import os
import json
import logging
from collections import defaultdict
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

# =============================================================================
# GPU CONFIGURATION
# =============================================================================
# Hardcoded to distribute the massive 32B LLM inference across two GPUs
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# =============================================================================
# INPUT FILES (EDIT THESE FOR DIFFERENT EVALUATION SETTINGS)
# =============================================================================
# The candidate list of patient-trial pairs we want to analyze (from classical retrieval)
RETRIEVAL_FILE = "data/2022/WholeQ_RETRIEVAL_T2022.txt"
# The raw text of the patient case descriptions
TOPICS_CSV = "data/2022/ct_2022_queries.tsv"
# The 448k+ clinical trial corpus in JSONL format
TREC_COLLECTION_PATH = "data/clinicaltrials/json_corpus/corpus.jsonl"
# The official TREC relevance judgments (Used to calculate feature discriminability)
QRELS_FILE = "data/2022/ct_2022_qrels_mapped.txt"

# The foundational Large Language Model used in Phase 1
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
# The optimized weights of the RouteSAE trained via `train_routesae.py`
ROUTESAE_MODEL_PATH = "routesae_outputs/train_1196/models/routesae_final.pt"

# =============================================================================
# SETTINGS
# =============================================================================
OUTPUT_ROOT = "routesae_inference"
MAX_LENGTH = 4096 # Maximum token sequence length for the LLM
FLOAT_TYPE = torch.float16 # FP16 is strictly required to fit 32B parameters in standard VRAM

# --- RouteSAE Architecture Constraints (Must perfectly match training) ---
TOP_K = 64 # The exact number of non-zero features allowed per inference pass
SAE_EXPANSION_FACTOR = 8 # The sparse dictionary is 8x wider than the LLM's hidden state
ROUTING_LAYER_FRAC_START = 0.25 # Begin routing at 25% model depth
ROUTING_LAYER_FRAC_END = 0.75   # End routing at 75% model depth
TOP_FEATURES_TO_SAVE = 20       # To save disk space, only log the 20 strongest feature activations per pair
ROUTESAE_BATCH_SIZE = 64        # RouteSAE is lightweight; we can batch its inference

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =============================================================================
# OUTPUT PATHS AND DIRECTORY SETUP
# =============================================================================
# Dynamically extract the name of the run (e.g., 'WholeQ_RETRIEVAL_T2022')
RUN_STEM = os.path.splitext(os.path.basename(RETRIEVAL_FILE))[0]
RUN_OUTPUT_DIR = os.path.join(OUTPUT_ROOT, RUN_STEM)
HIDDEN_STATES_DIR = os.path.join(RUN_OUTPUT_DIR, "hidden_states")
FEATURES_DIR = os.path.join(RUN_OUTPUT_DIR, "features")

# Create all necessary subdirectories
for d in [OUTPUT_ROOT, RUN_OUTPUT_DIR, HIDDEN_STATES_DIR, FEATURES_DIR]:
    os.makedirs(d, exist_ok=True)

# --- Caching Paths ---
# Because extracting 32B LLM states takes hours, we cache the outputs to disk.
HIDDEN_STATES_PATH = os.path.join(HIDDEN_STATES_DIR, "all_hidden_states.pt")
LABELS_PATH = os.path.join(HIDDEN_STATES_DIR, "labels.pt")
SAMPLE_IDS_PATH = os.path.join(HIDDEN_STATES_DIR, "sample_ids.json")

# --- Interpretability Output Paths ---
ALL_Z_PATH = os.path.join(FEATURES_DIR, "all_z.pt") # Raw sparse activation tensor
SELECTED_LAYERS_PATH = os.path.join(FEATURES_DIR, "selected_layers.pt") # Which LLM layer was routed
ACTIVE_FEATURES_PATH = os.path.join(FEATURES_DIR, "active_features.json") # Detailed pair-to-feature mapping
FEATURE_SUMMARY_PATH = os.path.join(FEATURES_DIR, "feature_summary.json") # Global discriminability stats

LOG_PATH = os.path.join(RUN_OUTPUT_DIR, "inference.log")

# =============================================================================
# LOGGING
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(LOG_PATH)],
)
logger = logging.getLogger(__name__)


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================
def load_topics(path: str) -> Dict[str, str]:
    """Loads test-set patient case descriptions from a TSV file."""
    topics = {}
    with open(path, "r", encoding="utf-8") as f:
        # LOOP EXPLANATION: Read TSV line-by-line to extract Query ID and Text
        for line in f:
            line = line.strip()
            if not line:
                continue
            topic_id, text = line.split("\t", 1)
            topics[str(topic_id)] = text.strip()
    logger.info("Loaded %d topics", len(topics))
    return topics


def load_corpus(path: str) -> Dict[str, Dict]:
    """Loads the massive Clinical Trials JSONL corpus into memory."""
    corpus = {}
    with open(path, "r", encoding="utf-8") as f:
        # LOOP EXPLANATION: Read JSONL line-by-line to prevent standard JSON memory overflow
        for line in f:
            doc = json.loads(line)
            trial_id = doc.get("id", doc.get("_id"))
            corpus[trial_id] = doc
    logger.info("Loaded %d trials", len(corpus))
    return corpus


def load_run_file(path: str) -> List[Dict]:
    """
    Parses the standard TREC retrieval run file.
    These are the exact candidate pairs the system needs to evaluate.
    """
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        # LOOP EXPLANATION: Parse standard TREC format: `query_id Q0 doc_id rank score run_name`
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            samples.append(
                {
                    "topic_id": parts[0],
                    "trial_id": parts[2],
                    "rank": int(parts[3]),
                    "score": float(parts[4]), # Baseline retrieval score
                }
            )
    logger.info("Loaded %d retrieval pairs", len(samples))
    return samples


def load_qrels(path: str) -> Dict[Tuple[str, str], int]:
    """
    Loads ground truth relevance judgments.
    Qrels format: topic_id 0 trial_id relevance (0 = non-relevant, 2 = relevant)
    Missing entries default to 0.
    """
    qrels = {}
    with open(path, "r", encoding="utf-8") as f:
        # LOOP EXPLANATION: Map the human-annotated relevance to a strict binary label
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            topic_id = parts[0]
            trial_id = parts[2]
            rel = int(parts[3])
            label = 1 if rel == 2 else 0 # Strict evaluation: only highly relevant (2) is a positive match
            qrels[(topic_id, trial_id)] = label
    logger.info("Loaded %d qrels entries", len(qrels))
    return qrels


# =============================================================================
# PROMPT RECONSTRUCTION
# =============================================================================
def build_prompt(patient_query: str, trial_doc: Dict) -> str:
    """
    CRITICAL METHOD: Reconstructs the EXACT prompt used to generate the LLM 
    justifications in Phase 1 (`neureq_ph1a.py`).
    
    Why this matters: A neural network's hidden state represents the mathematical 
    context of the specific tokens it processed. If this prompt differs by even a 
    single space from Phase 1, the LLM will generate a different computation graph, 
    and the trained RouteSAE dictionary will fail to map the concepts accurately.
    """
    trial_text = trial_doc.get("contents", "")
    if not trial_text:
        trial_text = json.dumps(trial_doc, ensure_ascii=False)

    prompt = f"""### Role: You are an expert in biomedical AI with access to clinical trial data and the ability to assess the relevance of a given patient case description to a specific clinical trial. Your task is to evaluate whether the trial is relevant to the patient case by answering a set of predefined questions in YES or NO or NA, along with a brief justification for each answer.
---

### Instructions:
- Given a clinical trial description and a patient case description, evaluate relevance based on the 10 feature-based questions provided below.
- Respond with "YES", "NO", or "NA" (Not Available):
  - YES: The patient's details align with the trial's criteria and objectives.
  - NO: The patient's details do not match the trial's requirements.
  - NA: The information is not available or cannot be determined.
- Provide a brief justification for each response, citing relevant details from the patient's symptoms, diagnostics, prior treatment,
  age, endocrinology findings, and other factors.
- If specific details needed for a question are not mentioned, return NA for that question.
- Format the final output as a JSON object, where the key is the question number and the value is an object containing:
  - "response" → "YES", "NO", or "NA"
  - "justification" → A brief explanation
- Strictly output in JSON format only.

---

1. Age Eligibility – Does the patient's age fall within the trial's specified range?
2. Gender Eligibility – Is the trial open to the patient’s gender?
3. Condition Relevance – Do the patient's symptoms, diagnosis, or condition match the trial’s focus?
4. Diagnostic Findings Match – Do lab tests, imaging, or biomarkers align with the trial’s criteria?
5. Prior Treatment Consideration – Has the patient undergone treatments relevant to the trial’s eligibility criteria?
6. Inclusion/Exclusion Criteria – Does the patient meet specific trial conditions (e.g., comorbidities, concurrent medications)?
7. Pathophysiologic Mechanism – Does the patient’s condition suggest an underlying disease mechanism relevant to the trial?
8. Functional Status – Does the patient’s sensory, motor, or cognitive function align with trial requirements?
9. Interest in Experimental Therapy – Has the patient shown willingness for investigational treatments?
10. Treatment Target Alignment – Does the trial’s treatment directly address the patient’s condition or symptoms?

---

### NOTE:
For the first two question check based on following condition.
Question: Does the patient's age fall within the trial's specified age range?
Conditions:
If both minimum and maximum age are specified, check if the patient's age is within the range.
If only a minimum age is specified, check if the patient's age is greater than or equal to the minimum age.
If only a maximum age is specified, check if the patient's age is less than or equal to the maximum age.
If no age restrictions are specified, assume the trial is open to all ages (Answer YES).
If the patient's age doesn't meet any of the above conditions, answer NO.

2. Gender Eligibility:
Question: Is the trial open to participants of the patient's gender?
Conditions:
If gender is not specified in the trial, assume the trial is open to all genders (Answer YES).
If gender is specified (e.g., male, female, or both), check if the patient's gender matches the trial's eligibility.
If the trial specifies a gender restriction (e.g., only male or only female) and the patient doesn't meet that restriction, answer NO.
If gender is not relevant or not mentioned, answer YES.

---

### Query: {patient_query}
---

### Clinical Trial: {trial_text}
---

### Output Format:
Strictly in JSON format only.
Generate a JSON object where each question number is a key, containing a dictionary with:
"response" → "YES" or "NO" or "NA"
"justification" → A brief explanation for the answer
---

### Expected Output Format:
{{
  "1": {{
    "response": "YES/NO/NA",
    "justification": "<A brief explanation for the response based on patient case description and trial details>"
  }},
  ...
  "10": {{
    "response": "YES/NO/NA",
    "justification": "<A brief explanation for the response based on patient case description and trial details>"
  }}
}}

---
### Output:"""

    return prompt.strip()


# =============================================================================
# ROUTESAE ARCHITECTURE CLASSES
# =============================================================================
class TopKActivation(nn.Module):
    """
    Enforces exact sparsity. 
    Mathematical context: Standard SAEs use L1 loss, but L1 mathematically "shrinks" 
    active features (shrinkage bias). By using a hard TopK gate, we preserve the 
    exact magnitude of the clinical feature activation (Gao et al. 2024).
    """
    def __init__(self, k):
        super().__init__()
        self.k = k

    def forward(self, x):
        # x Shape: (Batch, SAE_Width). e.g., (64, 40960)
        
        # Identify the 'k' largest activation values and their exact indices
        vals, idx = torch.topk(x, self.k, dim=-1)
        
        # Create a dense matrix of absolute zeros
        out = torch.zeros_like(x)
        
        # Inject the top 'k' values back into the zero matrix at their original indices.
        # This yields a perfectly sparse tensor with exactly K non-zero elements per row.
        out.scatter_(-1, idx, vals)
        return out


class Router(nn.Module):
    """
    Learns to dynamically route a specific LLM hidden state layer to the SAE.
    This avoids the computationally prohibitive task of training a separate 
    autoencoder for all layers of DeepSeek.
    """
    def __init__(self, hidden_size, num_layers):
        super().__init__()
        self.linear = nn.Linear(hidden_size, num_layers, bias=False)

    def forward(self, x):
        # x Shape: (Batch, L, d). e.g., (64, 32, 5120)

        # 1. Compress the layer dimension by sum-pooling to get a semantic summary
        v = x.sum(dim=1)                                       # Shape: (Batch, d)
        
        # 2. Project the summary vector into logits representing the L layers
        alpha = self.linear(v)                                 # Shape: (Batch, L)
        
        # 3. Convert logits to a routing probability distribution
        p = F.softmax(alpha, dim=-1)                           # Shape: (Batch, L)
        
        # 4. HARD SELECTION: Pick the single layer with the highest probability
        i_star = torch.argmax(p, dim=-1)                       # Shape: (Batch,)

        # 5. Extract the specific d-dimensional vector for that chosen layer
        idx = i_star.view(-1, 1, 1).expand(-1, 1, x.size(-1))
        x_selected = x.gather(1, idx).squeeze(1)               # Shape: (Batch, d)

        # 6. ARCHITECTURE ALIGNMENT: Multiply the selected vector by its scalar probability.
        # While primarily a trick used during training to keep the graph differentiable, 
        # it must be applied during inference to ensure the input scale matches what 
        # the SAE weights were trained on.
        p_star = p.gather(1, i_star.unsqueeze(1)).squeeze(1)   # Shape: (Batch,)
        x_route = p_star.unsqueeze(-1) * x_selected            # Shape: (Batch, d)

        return x_route, p, i_star


class SharedTopKSAE(nn.Module):
    """
    Projects the routed dense vector into a high-dimensional, sparse dictionary.
    """
    def __init__(self, hidden_size, sae_width, k):
        super().__init__()
        self.b_pre = nn.Parameter(torch.zeros(hidden_size))
        self.encoder = nn.Linear(hidden_size, sae_width, bias=False)
        self.topk = TopKActivation(k)
        self.decoder = nn.Linear(sae_width, hidden_size, bias=False)

    def forward(self, x):
        # x Shape: (Batch, d)
        
        # Encoding: Remove bias -> Project to wide space -> Apply exact TopK sparsity
        z = self.topk(self.encoder(x - self.b_pre))          # Shape: (Batch, SAE_Width)
        
        # Decoding: Project the sparse concepts back down to the original LLM dimension
        x_hat = self.decoder(z) + self.b_pre                 # Shape: (Batch, d)
        
        return z, x_hat


class RouteSAE(nn.Module):
    """
    The unified architecture combining the dynamic Router and the Sparse Autoencoder.
    """
    def __init__(self, hidden_size, num_layers, sae_width, k):
        super().__init__()
        self.num_routing_layers = num_layers
        self.router = Router(hidden_size, num_layers)
        self.sae = SharedTopKSAE(hidden_size, sae_width, k)

    def forward(self, x):
        # x Shape: (Batch, L, d)
        x_route, p, i_star = self.router(x)
        z, x_hat = self.sae(x_route)
        loss = F.mse_loss(x_hat, x_route) # Reconstruction loss (Logged but ignored during inference)
        
        return {
            "z": z,                       # The sparse feature activations (The Clinical Concepts!)
            "loss": loss,
            "selected_layer": i_star,     # The LLM layer chosen by the router
            "routing_probs": p,
        }


# =============================================================================
# UTILITIES
# =============================================================================
def get_routing_layers(num_layers: int) -> List[int]:
    """Isolates the middle layers representing deep semantic reasoning."""
    start = int(num_layers * ROUTING_LAYER_FRAC_START)
    end = int(num_layers * ROUTING_LAYER_FRAC_END)
    return list(range(start, end))


# =============================================================================
# HIDDEN STATE EXTRACTION (Test Set)
# =============================================================================
def extract_hidden_states(samples, topics, corpus, tokenizer, llm, routing_layers):
    """
    Executes a forward pass on the frozen 32B LLM to capture the internal state 
    for every pair in the test retrieval run.
    """
    # Short-circuit to load from cache if already processed
    if (
        os.path.exists(HIDDEN_STATES_PATH)
        and os.path.exists(LABELS_PATH)
        and os.path.exists(SAMPLE_IDS_PATH)
    ):
        logger.info("Loading cached hidden states...")
        hidden_states = torch.load(HIDDEN_STATES_PATH, map_location="cpu")
        labels = torch.load(LABELS_PATH, map_location="cpu")
        with open(SAMPLE_IDS_PATH, "r") as f:
            sample_ids = json.load(f)
        return hidden_states, labels, sample_ids

    all_hs = []
    all_labels = []
    sample_ids = []

    llm_device = next(llm.parameters()).device

    # LOOP EXPLANATION: We process the test set sequentially (batch size 1).
    # Why? The DeepSeek 32B model is massive. Processing sequences up to 4096 tokens
    # in batches would immediately cause CUDA Out-Of-Memory (OOM) errors. 
    for sample in tqdm(samples, desc="Extracting hidden states"):
        topic_id = sample["topic_id"]
        trial_id = sample["trial_id"]

        query_text = topics.get(topic_id)
        trial_doc = corpus.get(trial_id)

        if query_text is None or trial_doc is None:
            continue

        prompt = build_prompt(query_text, trial_doc)

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
        )

        input_ids = inputs["input_ids"].to(llm_device)
        attention_mask = inputs["attention_mask"].to(llm_device)

        with torch.no_grad():
            # output_hidden_states=True forces PyTorch to return a tuple containing 
            # the intermediate states from every single transformer block.
            outputs = llm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

        # We extract ONLY at the final token, right before generation starts.
        # This token mathematically encodes the maximal context of the patient-trial comparison.
        last_tok = input_ids.shape[1] - 1

        layer_vecs = []
        # LOOP EXPLANATION (INNER): Iterate through only the designated "middle" semantic layers
        for layer_idx in routing_layers:
            hs_layer = outputs.hidden_states[layer_idx + 1] # Shape: (1, seq_len, d)
            vec = hs_layer[0, last_tok, :].cpu().float()    # Shape: (d,)
            layer_vecs.append(vec)

        # Stack the selected L layers to create the final input tensor for the Router
        all_hs.append(torch.stack(layer_vecs, dim=0))       # Shape: (L, d)
        all_labels.append(sample["label"])                  
        sample_ids.append(sample)

    if len(all_hs) == 0:
        raise RuntimeError("No hidden states extracted.")

    # Stack the entire test set into a single massive tensor
    hidden_states = torch.stack(all_hs)                     # Shape: (Total_Samples, L, d)
    labels = torch.tensor(all_labels, dtype=torch.long)

    # Save to cache
    torch.save(hidden_states, HIDDEN_STATES_PATH)
    torch.save(labels, LABELS_PATH)
    with open(SAMPLE_IDS_PATH, "w") as f:
        json.dump(sample_ids, f, indent=2)

    return hidden_states, labels, sample_ids


# =============================================================================
# FEATURE SUMMARY (Discriminability Analysis)
# =============================================================================
def build_feature_summary(all_z, labels, selected_layers, sample_ids):
    """
    Computes global Discriminability statistics for the test set.
    Matches the formulation D_j = mu_pos - mu_neg defined in the paper.
    """
    pos_mask = labels == 1
    neg_mask = labels == 0

    # Calculate average activation strength for each feature across classes
    mean_pos = all_z[pos_mask].mean(dim=0)
    mean_neg = all_z[neg_mask].mean(dim=0)
    
    # Discriminability metric: 
    # Positive values = Feature signifies Eligibility (Inclusion)
    # Negative values = Feature signifies Mismatch (Exclusion)
    discriminability = mean_pos - mean_neg

    # Identify the Top 20 features for Inclusion and Exclusion
    top_pos = torch.topk(discriminability, 20).indices.tolist()
    top_neg = torch.topk(-discriminability, 20).indices.tolist()

    # Calculate Dead Features (A common issue in SAEs where units never activate)
    active_freq = (all_z > 0).float().mean(dim=0)
    dead_features = int((active_freq == 0).sum().item())

    # Tally which LLM layers the Router preferred for the test set
    routing_counts = torch.bincount(
        selected_layers,
        minlength=int(selected_layers.max().item()) + 1,
    ).tolist()

    summary = {
        "top_positive_feature_indices": top_pos,
        "top_negative_feature_indices": top_neg,
        "discriminability_scores": discriminability.tolist(),
        "active_frequency": active_freq.tolist(),
        "dead_features": dead_features,
        "routing_layer_counts": routing_counts,
        "num_samples": len(sample_ids),
        "num_positive": int(pos_mask.sum().item()),
        "num_negative": int(neg_mask.sum().item()),
    }

    with open(FEATURE_SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)


# =============================================================================
# MAIN INFERENCE LOOP
# =============================================================================
def main():
    # --- 1. Load Data ---
    logger.info("Loading topics...")
    topics = load_topics(TOPICS_CSV)

    logger.info("Loading corpus...")
    corpus = load_corpus(TREC_COLLECTION_PATH)

    logger.info("Loading retrieval file...")
    samples = load_run_file(RETRIEVAL_FILE)

    logger.info("Loading qrels...")
    qrels = load_qrels(QRELS_FILE)

    # LOOP EXPLANATION: Attach the ground-truth label to each test sample.
    # Defaults to 0 (non-relevant) if the pair wasn't judged by human annotators.
    for sample in samples:
        sample["label"] = qrels.get(
            (sample["topic_id"], sample["trial_id"]),
            0,
        )

    # --- 2. Setup Architectures ---
    logger.info("Loading DeepSeek config...")
    config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
    hidden_size = config.hidden_size
    num_layers = config.num_hidden_layers

    routing_layers = get_routing_layers(num_layers)
    num_routing_layers = len(routing_layers)
    sae_width = hidden_size * SAE_EXPANSION_FACTOR

    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading DeepSeek model...")
    llm = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=FLOAT_TYPE,
        device_map="auto",
        trust_remote_code=True,
    )
    llm.eval() # Freeze LLM

    # --- 3. Extract LLM Hidden States ---
    logger.info("Extracting hidden states...")
    hidden_states, labels, sample_ids = extract_hidden_states(
        samples,
        topics,
        corpus,
        tokenizer,
        llm,
        routing_layers,
    )

    # --- 4. Load Trained RouteSAE ---
    logger.info("Loading RouteSAE...")
    routesae = RouteSAE(
        hidden_size=hidden_size,
        num_layers=num_routing_layers,
        sae_width=sae_width,
        k=TOP_K,
    )
    # Inject the trained dictionary weights
    routesae.load_state_dict(torch.load(ROUTESAE_MODEL_PATH, map_location="cpu"))
    routesae.to(DEVICE)
    routesae.eval()

    logger.info("Running batched RouteSAE inference...")

    all_z = []
    all_selected_layers = []
    active_features = []

    # --- 5. Batched RouteSAE Execution ---
    # LOOP EXPLANATION: Unlike the 32B LLM, the SAE is computationally lightweight. 
    # We slice the cached hidden states into chunks (e.g., 64 pairs) to speed up inference.
    for start in tqdm(range(0, hidden_states.size(0), ROUTESAE_BATCH_SIZE), desc="RouteSAE"):
        # Secure boundary calculation to prevent indexing errors on the final partial batch
        end = min(start + ROUTESAE_BATCH_SIZE, hidden_states.size(0))
        batch = hidden_states[start:end].to(DEVICE)

        with torch.no_grad():
            out = routesae(batch)

        # Move the sparse activations and routing choices back to CPU RAM
        z = out["z"].cpu()
        selected = out["selected_layer"].cpu()

        all_z.append(z)
        all_selected_layers.append(selected)

        # LOOP EXPLANATION (INNER): Process each individual sample within the current batch.
        for i in range(z.size(0)):
            # 'start + i' maps the local batch index back to the global dataset index
            sample = sample_ids[start + i]
            z_i = z[i]
            route_idx = int(selected[i].item())

            # Map the local routing index (e.g., 0) back to the actual LLM layer (e.g., 16)
            actual_layer = routing_layers[route_idx]

            # Even though K=64 features fire according to the TopK layer, we only log 
            # the TOP 20 strongest activations per pair to keep JSON file sizes manageable for analysis.
            top_vals, top_idx = torch.topk(z_i, TOP_FEATURES_TO_SAVE)
            top_features = []
            
            # LOOP EXPLANATION (DEEP INNER): Record the active features for this specific pair.
            for idx, val in zip(top_idx.tolist(), top_vals.tolist()):
                if val > 0: # Sanity check: Only log if the feature actually fired
                    top_features.append(
                        {
                            "feature_index": int(idx),
                            "activation": float(val),
                        }
                    )

            # Construct the detailed JSON log mapping the patient-trial to active clinical concepts
            active_features.append(
                {
                    "topic_id": sample["topic_id"],
                    "trial_id": sample["trial_id"],
                    "rank": sample["rank"],
                    "retrieval_score": sample["score"],
                    "label": sample["label"],
                    "selected_route_index": route_idx,
                    "selected_transformer_layer": actual_layer,
                    "top_features": top_features,
                }
            )

    # Reassemble the batches into continuous global tensors
    all_z = torch.cat(all_z, dim=0)
    selected_layers = torch.cat(all_selected_layers, dim=0)

    # --- 6. Save & Summarize ---
    logger.info("Saving outputs...")
    torch.save(all_z, ALL_Z_PATH)
    torch.save(selected_layers, SELECTED_LAYERS_PATH)

    with open(ACTIVE_FEATURES_PATH, "w") as f:
        json.dump(active_features, f, indent=2)

    build_feature_summary(all_z, labels, selected_layers, sample_ids)

    logger.info("Done.")
    logger.info("all_z shape: %s", tuple(all_z.shape))
    logger.info("Saved to: %s", RUN_OUTPUT_DIR)


if __name__ == "__main__":
    main()