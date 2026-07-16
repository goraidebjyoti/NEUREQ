# SCT-BERT Baseline (Step 3/4): Self-Supervised Training (`sctbert_3.py`)

## Overview
This script executes the core neural training phase of the **SCT-BERT (Self-Supervised MonoBERT)** baseline. 

Traditional supervised re-rankers require expensive, human-annotated datasets containing patient queries mapped to relevant clinical trials. This baseline avoids that bottleneck by learning the underlying semantics of clinical eligibility via a proxy task: binary sequence classification over the trial's own internal fields. The model learns to predict whether a given `Brief Summary` semantically matches a given set of `Eligibility Criteria`.

By training on hundreds of thousands of these self-supervised pairs (generated in Steps 1 & 2), the model implicitly learns medical synonyms, demographic constraints, and clinical inclusion logic.

## Technical Architecture & Mathematical Context

* **Base Model:** `allenai/scibert_scivocab_uncased`. A BERT variant pre-trained on a massive corpus of biomedical research, ensuring the tokenizer and embedding layers already understand complex clinical terminology.
* **Cross-Encoder Formulation:** The model ingests the paired inputs via joint tokenization: `[CLS] Summary [SEP] Criteria [SEP]`. This allows bidirectional self-attention between the two text blocks across all layers of the network.
* **Objective:** Binary Sequence Classification (`num_labels=2`). A linear layer is attached to the pooled `[CLS]` token. The model is optimized using Cross-Entropy Loss to predict 1 (Match) or 0 (Mismatch).
* **Optimization:** Uses `AdamW` combined with Automatic Mixed Precision (`torch.cuda.amp.autocast`) to reduce memory consumption by ~50% and accelerate matrix multiplications on modern NVIDIA hardware.

## File Dependencies

### Inputs
* **Self-Supervised Dataset:** `data/clinicaltrials/train_pairs.jsonl`
  * *Origin:* Output of `sctbert_2.py`.
  * *Schema Requirements:* Each JSON line must contain `summary`, `criteria`, and `label` (integer `1` or `0`).

### Outputs
* **Trained Model Artifacts:** Saved to `models/sct_bert/`.
  * `model.safetensors` (or `pytorch_model.bin`): The fine-tuned neural weights.
  * `config.json`: Architecture definitions.
  * `vocab.txt` & `tokenizer.json`: The SciBERT vocabulary constraints required for Step 4.

## Usage & Configuration

Ensure the `TRAIN_FILE` points to your balanced positive/negative dataset.

```bash
python sctbert_3.py