# Baseline: Self-Supervised MonoBERT Training (`ssmb_1.py`)

## Overview
This script executes the first phase of the Self-Supervised MonoBERT (SSMB) baseline. Because large-scale, human-annotated queries mapping patients to clinical trials are extremely scarce, this script bypasses the need for manual labels through self-supervised learning. 

It trains a binary classification model to recognize the semantic alignment between two distinct textual fields extracted from the same clinical trial: the **Brief Summary** (which resembles a patient narrative) and the **Eligibility Criteria** (the strict inclusion/exclusion rules). By learning to predict whether a given summary matches a given criteria block, the model implicitly learns the domain-specific vocabulary and logic required for downstream patient-to-trial retrieval tasks.

## Technical Architecture
* **Base Model:** Utilizes `allenai/scibert_scivocab_uncased` as the foundational architecture[cite: 7]. SciBERT is heavily pre-trained on biomedical text, making it highly suitable for clinical applications.
* **Architecture:** Formulated as a sequence classification task (`AutoModelForSequenceClassification` with `num_labels=2`)[cite: 7]. The input pairs (Summary and Criteria) are concatenated with a standard `[SEP]` token separator.
* **Optimization:** Employs the `AdamW` optimizer with a learning rate of **2e-5** and a linear learning rate scheduler over **5 epochs**[cite: 7]. 
* **Data Processing:** The dataset is split into 90% training and 10% validation sets[cite: 7]. The `DataLoader` implements dynamic padding via a custom `collate_fn`, accelerating training by only padding sequences to the maximum length of a given batch rather than a static global maximum[cite: 7].

## File Dependencies & Formats

### Inputs (Required)
* **Training Pairs (`data/clinicaltrials/train_pairs.jsonl`)**
  * **Description:** A pre-generated dataset containing contrastive examples of matching and mismatched clinical fields. Generated from Code1->Code2 of SCT_BERT baseline. 
  * **Format:** JSON Lines (JSONL).
  * **Schema Requirements:** Each line must be a valid JSON object containing exactly three keys:
    * `summary` (String): The narrative text simulating a patient query.
    * `criteria` (String): The structured eligibility criteria text.
    * `label` (Integer/String): `1` for a positive/true semantic match, `0` for a negative/mismatched pair.
  * **Example:**
    ```json
    {"summary": "Patient presents with stage II breast cancer...", "criteria": "Inclusion: Confirmed stage II breast cancer. Exclusion: Prior chemotherapy.", "label": 1}
    ```

### Outputs (Generated)
The script creates a structured output directory (`models/self_supervised_monobert/`) containing checkpoints and logs.

* **Epoch Checkpoints (`epoch-1/` to `epoch-5/` and `final/`)**[cite: 7]
  * **Description:** Standard Hugging Face model directory structures containing the weights and tokenizer configurations. These are saved iteratively to allow for early-stopping selection during inference.
  * **Internal Files:**
    * `config.json`: The model architecture configuration.
    * `model.safetensors` (or `pytorch_model.bin`): The serialized model weights.
    * `vocab.txt` / `tokenizer.json` / `tokenizer_config.json`: Vocabulary and tokenizer settings required to encode text identically during inference.
* **Training Log (`train_log.txt`)**[cite: 7]
  * **Description:** A persistent plain-text log tracking model performance.
  * **Format:** `YYYY-MM-DD HH:MM:SS - LEVEL - Message`
  * **Contents:** Records the number of pairs loaded and outputs the average Training Loss, Training Accuracy, Validation Loss, and Validation Accuracy at the completion of every epoch.

## Usage
Ensure the base `scibert` model is accessible via the Hugging Face hub (or locally cached) and that the `train_pairs.jsonl` dataset is correctly positioned. 

Execute the script directly:
```bash
python ssmb_1.py