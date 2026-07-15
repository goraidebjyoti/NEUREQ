# train_routesae.py
"""
NEUREQ Interpretability: RouteSAE Training Pipeline

Training data source:
    train_1196.jsonl (The synthetic dataset generated via hard-negative mining)

Each record contains:
    - topic_id
    - trial_id
    - label (positive/negative)
    - output (ignored during training, as SAE is unsupervised)

The script:
    1. Loads synthetic patient queries from TSV.
    2. Loads clinical trial texts from corpus.jsonl.
    3. Builds the exact prompt used during synthetic data generation.
    4. Extracts hidden states from DeepSeek-R1-Distill-Qwen-32B.
    5. Trains RouteSAE (Router + Shared TopK Sparse Autoencoder).
    6. Performs feature analysis (Discriminability calculation).

Outputs:
    routesae_outputs/train_1196/
        hidden_states/
        models/
"""

import os
import json
import math
import logging
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# =============================================================================
# GPU CONFIGURATION
# =============================================================================
# The 32B parameter DeepSeek model is massive. We distribute the load across two GPUs.
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# =============================================================================
# INPUT CONFIGURATION
# =============================================================================
TRAIN_DATA_FILE      = "NEUREQ/data/triplet_syn_dataset_1196.jsonl"
TOPICS_CSV           = "NEUREQ/data/synthetic_gold_queries.tsv"
TREC_COLLECTION_PATH = "NEUREQ/data/corpus.jsonl"
MODEL_NAME           = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
OUTPUT_ROOT          = "routesae_outputs"

# =============================================================================
# OUTPUT PATHS
# =============================================================================
RUN_STEM          = os.path.splitext(os.path.basename(TRAIN_DATA_FILE))[0]
RUN_OUTPUT_DIR    = os.path.join(OUTPUT_ROOT, RUN_STEM)
HIDDEN_STATES_DIR = os.path.join(RUN_OUTPUT_DIR, "hidden_states")
MODELS_DIR        = os.path.join(RUN_OUTPUT_DIR, "models")

for d in [OUTPUT_ROOT, RUN_OUTPUT_DIR, HIDDEN_STATES_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

# Caching paths for extracted hidden states to save GPU hours on re-runs
HIDDEN_STATES_PATH   = os.path.join(HIDDEN_STATES_DIR, "all_hidden_states.pt")
LABELS_PATH          = os.path.join(HIDDEN_STATES_DIR, "all_labels.pt")
SAMPLE_IDS_PATH      = os.path.join(HIDDEN_STATES_DIR, "sample_ids.json")

# Model and artifact saving paths
SAE_MODEL_PATH       = os.path.join(MODELS_DIR, "routesae_final.pt")
TRAINING_LOG_PATH    = os.path.join(MODELS_DIR, "training.log")
LOSS_CURVE_PATH      = os.path.join(MODELS_DIR, "loss_curve.json")
FEATURE_ANALYSIS_PATH = os.path.join(MODELS_DIR, "feature_analysis.json")

# =============================================================================
# MODEL SETTINGS
# =============================================================================
# Float16 is required to fit the 32B base model into VRAM during extraction
FLOAT_TYPE = torch.float16
MAX_LENGTH = 4096

# =============================================================================
# ROUTESAE SETTINGS
# =============================================================================
TOP_K                    = 64    # Forces exact sparsity: only 64 active features per forward pass
SAE_EXPANSION_FACTOR     = 8     # The SAE dictionary will be 8x the size of the LLM hidden dimension
ROUTING_LAYER_FRAC_START = 0.25  # Only extract from layers 25% to 75% deep
ROUTING_LAYER_FRAC_END   = 0.75  # (Middle layers contain the most semantic/reasoning context)

# =============================================================================
# TRAINING SETTINGS
# =============================================================================
BATCH_SIZE         = 64
LEARNING_RATE      = 5e-4
NUM_EPOCHS         = 50
WARMUP_RATIO       = 0.05   # phase 1: linear warmup  over first 5%  of steps
STABLE_RATIO       = 0.75   # phase 2: constant LR    for next  75% of steps
                            # phase 3: linear decay   over final 20% of steps
NORM_REG_INTERVAL  = 10     # re-normalise decoder columns every N steps
GRAD_CLIP_NORM     = 1.0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =============================================================================
# LOGGING
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(TRAINING_LOG_PATH),
    ],
)
logger = logging.getLogger(__name__)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_topics(path: str) -> Dict[str, str]:
    """Load patient queries from TSV. Format: topic_id <TAB> query_text"""
    topics = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            topic_id, text = line.split("\t", 1)
            topics[str(topic_id)] = text.strip()
    logger.info("Loaded %d topics", len(topics))
    return topics


def load_corpus(path: str) -> Dict[str, Dict]:
    """Load clinical trial documents from JSONL. Returns {trial_id -> doc dict}"""
    corpus = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            trial_id = doc.get("id", doc.get("_id"))
            corpus[trial_id] = doc
    logger.info("Loaded %d trials", len(corpus))
    return corpus


def load_training_samples(path: str) -> List[Dict]:
    """
    Load (topic_id, trial_id, label) from train_1196.jsonl.
    The output field (10-question answers) is intentionally ignored because SAE 
    training is unsupervised; we only need the labels later for analysis.
    """
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            samples.append({
                "topic_id": str(item["topic_id"]),
                "trial_id": str(item["trial_id"]),
                "label":    1 if item["label"].lower() == "positive" else 0,
            })
    pos = sum(s["label"] == 1 for s in samples)
    neg = sum(s["label"] == 0 for s in samples)
    logger.info("Loaded %d samples  [positive: %d  negative: %d]", len(samples), pos, neg)
    return samples


# =============================================================================
# PROMPT
# =============================================================================

def build_prompt(patient_query: str, trial_doc: Dict) -> str:
    """
    CRITICAL: This must be the EXACT prompt used during Phase 1 (neureq_ph1a.py).
    Hidden states are highly context-dependent. If the prompt string changes by 
    even one space, the LLM will generate completely different hidden states, 
    breaking the interpretability mapping.
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
- Provide a brief justification for each response, citing relevant details from the patient's symptoms, diagnostics, prior treatment, age, endocrinology findings, and other factors.
- If specific details needed for a question are not mentioned, return NA for that question.
- Format the final output as a JSON object, where the key is the question number and the value is an object containing:
  - "response" → "YES", "NO", or "NA"
  - "justification" → A brief explanation
- Strictly output in JSON format only.

---

1. Age Eligibility – Does the patient's age fall within the trial's specified range?
2. Gender Eligibility – Is the trial open to the patient's gender?
3. Condition Relevance – Do the patient's symptoms, diagnosis, or condition match the trial's focus?
4. Diagnostic Findings Match – Do lab tests, imaging, or biomarkers align with the trial's criteria?
5. Prior Treatment Consideration – Has the patient undergone treatments relevant to the trial's eligibility criteria?
6. Inclusion/Exclusion Criteria – Does the patient meet specific trial conditions (e.g., comorbidities, concurrent medications)?
7. Pathophysiologic Mechanism – Does the patient's condition suggest an underlying disease mechanism relevant to the trial?
8. Functional Status – Does the patient's sensory, motor, or cognitive function align with trial requirements?
9. Interest in Experimental Therapy – Has the patient shown willingness for investigational treatments?
10. Treatment Target Alignment – Does the trial's treatment directly address the patient's condition or symptoms?

---

### NOTE:
For the first two questions, apply the following rules.

1. Age Eligibility:
Question: Does the patient's age fall within the trial's specified age range?
Conditions:
- If both minimum and maximum age are specified, check if the patient's age is within the range.
- If only a minimum age is specified, check if the patient's age is greater than or equal to the minimum age.
- If only a maximum age is specified, check if the patient's age is less than or equal to the maximum age.
- If no age restrictions are specified, assume the trial is open to all ages (answer YES).
- If the patient's age does not satisfy the applicable rule, answer NO.

2. Gender Eligibility:
Question: Is the trial open to participants of the patient's gender?
Conditions:
- If gender is not specified in the trial, assume the trial is open to all genders (answer YES).
- If gender is specified (e.g., male, female, or both), check whether the patient's gender matches the trial eligibility.
- If the trial specifies a gender restriction and the patient does not satisfy it, answer NO.
- If gender is not relevant or not mentioned, answer YES.

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
  ...,
  "10": {{
    "response": "YES/NO/NA",
    "justification": "<A brief explanation for the response based on patient case description and trial details>"
  }}
}}

---
### Output:"""

    return prompt.strip()


# =============================================================================
# ROUTING LAYERS
# =============================================================================

def get_routing_layers(num_layers: int) -> List[int]:
    """
    Return transformer-layer indices covering the middle
    [ROUTING_LAYER_FRAC_START, ROUTING_LAYER_FRAC_END) of model depth.
    Matches the paper's [1/4, 3/4] window.
    """
    start  = int(num_layers * ROUTING_LAYER_FRAC_START)
    end    = int(num_layers * ROUTING_LAYER_FRAC_END)
    layers = list(range(start, end))
    logger.info("Routing layers: %d-%d (%d layers)", layers[0], layers[-1], len(layers))
    return layers


# =============================================================================
# HIDDEN STATE EXTRACTION
# =============================================================================

def extract_hidden_states(
    model_name:       str,
    topics:           Dict[str, str],
    corpus:           Dict[str, Dict],
    samples:          List[Dict],
    routing_layers:   List[int],
    save_path:        str,
    labels_save_path: str,
    ids_save_path:    str,
) -> Tuple[torch.Tensor, torch.Tensor, List[Dict]]:
    """
    For every (topic_id, trial_id) pair in samples:
      1. Build the exact prompt used during train_1196.jsonl generation.
      2. Forward-pass through the frozen LLM.
      3. Record the hidden state of the LAST INPUT TOKEN at every routing layer.

    Hidden states are cached — re-running skips this stage automatically.

    Returns
    -------
    hidden_states : (N, L, d)   float32
    labels        : (N,)        int64   (1=positive, 0=negative)
    sample_ids    : list of {topic_id, trial_id, label}
    """
    # ── Cache check ───────────────────────────────────────────────────────────
    if (
        os.path.exists(save_path)
        and os.path.exists(labels_save_path)
        and os.path.exists(ids_save_path)
    ):
        logger.info("Loading cached hidden states...")
        hs     = torch.load(save_path,        map_location="cpu")
        labels = torch.load(labels_save_path, map_location="cpu")
        with open(ids_save_path, "r") as f:
            ids = json.load(f)
        logger.info("Loaded cached hidden states %s", tuple(hs.shape))
        return hs, labels, ids

    # ── Load model ────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=FLOAT_TYPE,
        device_map="auto",        # spreads 32B across all available GPUs
        trust_remote_code=True,
    )
    model.eval()
    logger.info("Model loaded. Extracting hidden states for %d samples...", len(samples))

    all_hs     = []
    all_labels = []
    sample_ids = []
    skipped    = 0

    for sample in tqdm(samples, desc="Extracting hidden states"):
        topic_id = sample["topic_id"]
        trial_id = sample["trial_id"]
        label    = sample["label"]

        patient_query = topics.get(topic_id)
        if patient_query is None:
            logger.warning("topic_id %s not found — skipped", topic_id)
            skipped += 1
            continue

        trial_doc = corpus.get(trial_id)
        if trial_doc is None:
            logger.warning("trial_id %s not found — skipped", trial_id)
            skipped += 1
            continue

        # Build the exact same prompt used to generate train_1196.jsonl
        prompt = build_prompt(patient_query, trial_doc)

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
        )

        first_device   = next(model.parameters()).device
        input_ids      = inputs["input_ids"].to(first_device)
        attention_mask = inputs["attention_mask"].to(first_device)

        # outputs.hidden_states: tuple of (num_layers + 1) tensors (1, seq_len, d)
        #   index 0   = embedding layer output
        #   index i+1 = output of transformer layer i
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True, # Critical flag to intercept the internals
            )

        # Last input token = the token just before generation starts
        # This token encodes the maximal context of the patient-trial comparison
        last_tok = input_ids.shape[1] - 1

        layer_vecs = []
        for layer_idx in routing_layers:
            hs_layer = outputs.hidden_states[layer_idx + 1]         # (1, seq_len, d)
            vec      = hs_layer[0, last_tok, :].cpu().float()       # (d,)  float32
            layer_vecs.append(vec)

        all_hs.append(torch.stack(layer_vecs, dim=0))               # (L, d)
        all_labels.append(label)
        sample_ids.append(sample)

    logger.info("Extraction complete — %d saved, %d skipped", len(all_hs), skipped)

    hidden_states = torch.stack(all_hs)                              # (N, L, d)
    labels_tensor = torch.tensor(all_labels, dtype=torch.long)      # (N,)

    torch.save(hidden_states, save_path)
    torch.save(labels_tensor, labels_save_path)
    with open(ids_save_path, "w") as f:
        json.dump(sample_ids, f, indent=2)

    logger.info("Hidden states shape: %s", tuple(hidden_states.shape))
    return hidden_states, labels_tensor, sample_ids


# =============================================================================
# ROUTESAE COMPONENTS
# =============================================================================

class TopKActivation(nn.Module):
    """
    Keeps the K largest values; zeroes the rest.
    Enforces exact sparsity without L1.
    Mathematical context: Standard SAEs use L1 loss, but L1 mathematically "shrinks" 
    active features (shrinkage bias). By using a hard TopK gate, we preserve the 
    exact magnitude of the clinical feature activation (Gao et al. 2024).
    """
    def __init__(self, k: int):
        super().__init__()
        self.k = k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, M) → sparse z: (batch, M) with exactly K non-zeros per row"""
        vals, idx = torch.topk(x, self.k, dim=-1)
        out = torch.zeros_like(x)
        out.scatter_(-1, idx, vals) # Re-injects the TopK values back at their original index
        return out


class Router(nn.Module):
    """
    Lightweight router that selects the single most-active layer (hard routing).

    Paper §2.2:
      v       = Σ_i x_i              ← sum pool across L routing layers
      α       = W_router @ v          ← linear projection  ∈ ℝ^L
      p       = softmax(α)            ← layer probabilities
      i* = argmax(p)             ← winning layer
      x_route = p_{i*} · x_{i*}      ← probability-scaled (keeps graph differentiable)
    """
    def __init__(self, hidden_size: int, num_layers: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, num_layers, bias=False)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x : (batch, L, d)
        Returns x_route (batch, d), p (batch, L), i_star (batch,)
        """
        v      = x.sum(dim=1)                                         # (batch, d)
        alpha  = self.linear(v)                                       # (batch, L)
        p      = F.softmax(alpha, dim=-1)                             # (batch, L)
        
        # Hard Routing Selection
        i_star = torch.argmax(p, dim=-1)                              # (batch,)

        idx        = i_star.view(-1, 1, 1).expand(-1, 1, x.size(-1))
        x_selected = x.gather(1, idx).squeeze(1)                      # (batch, d)

        # IMPORTANT TRICK: argmax is non-differentiable. To allow backpropagation 
        # to train the Router, we multiply the selected vector by its scalar probability.
        p_star  = p.gather(1, i_star.unsqueeze(1)).squeeze(1)         # (batch,)
        x_route = p_star.unsqueeze(-1) * x_selected                   # (batch, d)

        return x_route, p, i_star


class SharedTopKSAE(nn.Module):
    """
    Shared TopK Sparse Autoencoder.

    Encoder : z   = TopK( W_enc (x − b_pre) )
    Decoder : x̂   = W_dec z  +  b_pre

    Decoder columns are periodically re-normalised to unit length
    (unit norm regularisation, every NORM_REG_INTERVAL steps).
    """
    def __init__(self, hidden_size: int, sae_width: int, k: int):
        super().__init__()
        self.b_pre    = nn.Parameter(torch.zeros(hidden_size))
        self.encoder  = nn.Linear(hidden_size, sae_width, bias=False)
        self.topk     = TopKActivation(k)
        self.decoder  = nn.Linear(sae_width, hidden_size, bias=False)
        self.normalize_decoder()

    @torch.no_grad()
    def normalize_decoder(self):
        """
        Project each decoder column to unit length.
        Prevents the network from cheating by scaling up weights to minimize MSE.
        """
        norms = self.decoder.weight.norm(dim=0, keepdim=True).clamp(min=1e-8)
        self.decoder.weight.div_(norms)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """x: (batch, d) → z: (batch, M) sparse codes, x_hat: (batch, d) reconstruction"""
        z     = self.topk(self.encoder(x - self.b_pre))
        x_hat = self.decoder(z) + self.b_pre
        return z, x_hat


class RouteSAE(nn.Module):
    """
    Route Sparse Autoencoder (Shi et al., EMNLP 2025).
    Router + Shared TopK SAE → unified feature space across multiple layers.
    """
    def __init__(self, hidden_size: int, num_layers: int, sae_width: int, k: int):
        super().__init__()
        self.hidden_size        = hidden_size
        self.num_routing_layers = num_layers
        self.sae_width          = sae_width
        self.k                  = k

        self.router = Router(hidden_size, num_layers)
        self.sae    = SharedTopKSAE(hidden_size, sae_width, k)

    def forward(self, x: torch.Tensor) -> Dict:
        """
        x : (batch, L, d)
        Returns dict with z, loss, i_star
        """
        x_route, p, i_star = self.router(x)
        z, x_hat           = self.sae(x_route)
        loss               = F.mse_loss(x_hat, x_route)   # MSE only — no L1 needed
        return {"z": z, "loss": loss, "i_star": i_star}

    @torch.no_grad()
    def normalize_decoder(self):
        self.sae.normalize_decoder()


# =============================================================================
# DATASET
# =============================================================================

class HiddenStateDataset(Dataset):
    """
    Wraps pre-extracted hidden states and labels.
    SAE training is fully unsupervised — labels are only used in feature analysis.
    """
    def __init__(self, hidden_states: torch.Tensor, labels: torch.Tensor):
        self.hidden_states = hidden_states
        self.labels        = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.hidden_states[idx], self.labels[idx]


# =============================================================================
# LR SCHEDULER  (3-phase schedule from the RouteSAE paper)
# =============================================================================

def build_lr_scheduler(
    optimizer:    torch.optim.Optimizer,
    total_steps:  int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Three-phase schedule (Shi et al., EMNLP 2025):

      Phase 1 — linear warmup  : LR goes 0 → peak  over first WARMUP_RATIO steps
      Phase 2 — stable         : LR stays at peak   for     STABLE_RATIO steps
      Phase 3 — linear decay   : LR goes peak → 0   over remaining steps

    Purpose:
      Warmup  → prevents large unstable gradients at the start of training
      Stable  → allows the model to learn at full learning rate
      Decay   → fine-tunes the model into a better final minimum

    Note: This does NOT affect training speed — only training quality.
    """
    warmup_steps = int(total_steps * WARMUP_RATIO)   # 5%  of total steps
    stable_steps = int(total_steps * STABLE_RATIO)   # 75% of total steps
    decay_steps  = total_steps - warmup_steps - stable_steps  # remaining 20%

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            # Phase 1: linear ramp from 0 to 1.0
            return float(step) / max(1, warmup_steps)
        elif step < warmup_steps + stable_steps:
            # Phase 2: constant at peak LR
            return 1.0
        else:
            # Phase 3: linear decay from 1.0 to 0
            elapsed = step - warmup_steps - stable_steps
            return max(0.0, 1.0 - float(elapsed) / max(1, decay_steps))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# =============================================================================
# TRAINING
# =============================================================================

def train_routesae(model: RouteSAE, loader: DataLoader) -> List[float]:
    """
    Train RouteSAE end-to-end.

    Follows the paper:
      • Adam (β1=0.9, β2=0.999 — PyTorch defaults)
      • 3-phase LR schedule (warmup → stable → decay)
      • Gradient clipping (max norm = 1.0)
      • Decoder unit-norm regularisation every NORM_REG_INTERVAL steps
      • Loss = MSE reconstruction only (TopK enforces sparsity — no L1 needed)
    """
    model = model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 3-phase LR scheduler — better convergence, no effect on speed
    total_steps = NUM_EPOCHS * len(loader)
    scheduler   = build_lr_scheduler(optimizer, total_steps)

    epoch_losses = []
    global_step  = 0

    logger.info(
        "Training  |  epochs=%d  total_steps=%d  "
        "warmup=%d  stable=%d  decay=%d",
        NUM_EPOCHS, total_steps,
        int(total_steps * WARMUP_RATIO),
        int(total_steps * STABLE_RATIO),
        total_steps - int(total_steps * WARMUP_RATIO) - int(total_steps * STABLE_RATIO),
    )

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0

        for batch_hs, _ in tqdm(loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", leave=False):
            # Cast to float32 — SAE arithmetic runs in full precision to avoid NaN loss
            batch_hs = batch_hs.to(DEVICE, dtype=torch.float32)   # (B, L, d)

            optimizer.zero_grad()

            out  = model(batch_hs)
            loss = out["loss"]                 # MSE reconstruction loss only
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)

            optimizer.step()
            scheduler.step()                   # advance the 3-phase LR schedule

            # Decoder unit-norm regularisation (every N steps, per paper)
            if global_step % NORM_REG_INTERVAL == 0:
                model.normalize_decoder()

            running_loss += loss.item()
            global_step  += 1

        avg_loss   = running_loss / len(loader)
        current_lr = scheduler.get_last_lr()[0]
        epoch_losses.append(avg_loss)
        logger.info(
            "Epoch %3d/%d  loss: %.6f  lr: %.2e",
            epoch + 1, NUM_EPOCHS, avg_loss, current_lr,
        )

        # Periodic checkpoint
        if (epoch + 1) % 10 == 0:
            ckpt = SAE_MODEL_PATH.replace(".pt", f"_epoch{epoch+1}.pt")
            torch.save(model.state_dict(), ckpt)
            logger.info("Checkpoint saved → %s", ckpt)

    torch.save(model.state_dict(), SAE_MODEL_PATH)
    logger.info("Final model saved → %s", SAE_MODEL_PATH)

    with open(LOSS_CURVE_PATH, "w") as f:
        json.dump({"epoch_losses": epoch_losses}, f, indent=2)

    return epoch_losses


# =============================================================================
# FEATURE ANALYSIS
# =============================================================================

def analyze_features(model: RouteSAE, loader: DataLoader) -> None:
    """
    Run the trained RouteSAE in inference mode and rank features by how
    strongly they discriminate positive (eligible) from negative samples.

    Discriminability = mean_activation_positive − mean_activation_negative

    Results saved to FEATURE_ANALYSIS_PATH.
    """
    model.eval()
    model = model.to(DEVICE)

    z_list, labels_list, i_star_list = [], [], []

    with torch.no_grad():
        for batch_hs, batch_labels in tqdm(loader, desc="Feature extraction"):
            batch_hs = batch_hs.to(DEVICE, dtype=torch.float32)
            out = model(batch_hs)
            z_list.append(out["z"].cpu())
            labels_list.append(batch_labels)
            i_star_list.append(out["i_star"].cpu())

    all_z      = torch.cat(z_list)        # (N, M)
    all_labels = torch.cat(labels_list)   # (N,)
    all_i_star = torch.cat(i_star_list)   # (N,)

    pos_mask = all_labels == 1
    neg_mask = all_labels == 0

    logger.info(
        "Analysis  |  positive: %d  negative: %d",
        pos_mask.sum().item(), neg_mask.sum().item(),
    )

    # D = mu_pos - mu_neg (Calculates how much stronger a feature fires for relevant trials)
    mean_pos = all_z[pos_mask].float().mean(dim=0)             # (M,)
    mean_neg = all_z[neg_mask].float().mean(dim=0)             # (M,)
    freq_pos = (all_z[pos_mask] > 0).float().mean(dim=0)       # (M,)
    freq_neg = (all_z[neg_mask] > 0).float().mean(dim=0)       # (M,)
    disc     = mean_pos - mean_neg                             # (M,)

    top_pos = disc.topk(20).indices.tolist()
    top_neg = (-disc).topk(20).indices.tolist()

    layer_counts = torch.bincount(
        all_i_star, minlength=model.num_routing_layers
    ).tolist()

    logger.info("Routing layer selection counts: %s", layer_counts)

    logger.info("\n── Top 20 features most active for POSITIVE (eligible) samples ──")
    for fi in top_pos:
        logger.info(
            "  Feature %5d  disc=% .4f  mean_pos=%.4f  mean_neg=%.4f  "
            "freq_pos=%.3f  freq_neg=%.3f",
            fi, disc[fi].item(),
            mean_pos[fi].item(), mean_neg[fi].item(),
            freq_pos[fi].item(), freq_neg[fi].item(),
        )

    logger.info("\n── Top 20 features most active for NEGATIVE (non-eligible) samples ──")
    for fi in top_neg:
        logger.info(
            "  Feature %5d  disc=% .4f  mean_pos=%.4f  mean_neg=%.4f  "
            "freq_pos=%.3f  freq_neg=%.3f",
            fi, disc[fi].item(),
            mean_pos[fi].item(), mean_neg[fi].item(),
            freq_pos[fi].item(), freq_neg[fi].item(),
        )

    results = {
        "top_positive_feature_indices": top_pos,
        "top_negative_feature_indices": top_neg,
        "discriminability_scores":      disc.tolist(),
        "mean_activation_positive":     mean_pos.tolist(),
        "mean_activation_negative":     mean_neg.tolist(),
        "frequency_positive":           freq_pos.tolist(),
        "frequency_negative":           freq_neg.tolist(),
        "routing_layer_counts":         layer_counts,
    }

    with open(FEATURE_ANALYSIS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Feature analysis saved → %s", FEATURE_ANALYSIS_PATH)


# =============================================================================
# MAIN
# =============================================================================

def main():
    logger.info("=" * 60)
    logger.info("  RouteSAE — Clinical Trial Eligibility")
    logger.info("=" * 60)

    # ── Step 1: Load data ─────────────────────────────────────────────────────
    logger.info("\n[1/6] Loading data...")
    topics  = load_topics(TOPICS_CSV)
    corpus  = load_corpus(TREC_COLLECTION_PATH)
    samples = load_training_samples(TRAIN_DATA_FILE)

    # ── Step 2: Model config and routing layers ───────────────────────────────
    logger.info("\n[2/6] Reading model config...")
    config      = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
    hidden_size = config.hidden_size
    num_layers  = config.num_hidden_layers
    logger.info("Model: %d layers  hidden_size=%d", num_layers, hidden_size)

    routing_layers     = get_routing_layers(num_layers)
    num_routing_layers = len(routing_layers)
    sae_width          = hidden_size * SAE_EXPANSION_FACTOR
    logger.info(
        "Routing layers: %d   SAE width: %d (%dx)   K=%d",
        num_routing_layers, sae_width, SAE_EXPANSION_FACTOR, TOP_K,
    )

    # ── Step 3: Extract hidden states (cached after first run) ────────────────
    logger.info("\n[3/6] Extracting hidden states...")
    hidden_states, labels, sample_ids = extract_hidden_states(
        MODEL_NAME,
        topics,
        corpus,
        samples,
        routing_layers,
        HIDDEN_STATES_PATH,
        LABELS_PATH,
        SAMPLE_IDS_PATH,
    )
    logger.info("Hidden states: %s   Labels: %s",
                tuple(hidden_states.shape), tuple(labels.shape))

    # ── Step 4: Build dataset and dataloaders ─────────────────────────────────
    logger.info("\n[4/6] Building dataset...")
    dataset = HiddenStateDataset(hidden_states, labels)

    # Shuffled loader for training (SAE trains unsupervised — labels ignored)
    train_loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=(DEVICE == "cuda"),
    )
    # Unshuffled loader for deterministic feature extraction
    analysis_loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=(DEVICE == "cuda"),
    )
    logger.info("%d samples   %d batches/epoch", len(dataset), len(train_loader))

    # ── Step 5: Initialise and train RouteSAE ────────────────────────────────
    logger.info("\n[5/6] Initialising RouteSAE...")
    model = RouteSAE(hidden_size, num_routing_layers, sae_width, TOP_K)
    total_p  = sum(p.numel() for p in model.parameters())
    router_p = sum(p.numel() for p in model.router.parameters())
    sae_p    = sum(p.numel() for p in model.sae.parameters())
    logger.info(
        "Parameters — total: %s   router: %s   SAE: %s",
        f"{total_p:,}", f"{router_p:,}", f"{sae_p:,}",
    )

    logger.info("\n[6/6] Training RouteSAE...")
    epoch_losses = train_routesae(model, train_loader)
    logger.info("Final training loss: %.6f", epoch_losses[-1])

    # ── Step 6: Feature analysis ──────────────────────────────────────────────
    logger.info("\nAnalysing features...")
    analyze_features(model, analysis_loader)

    logger.info("\n✓ Done.")


if __name__ == "__main__":
    main()