# NEUREQ Phase 2a: Neural Re-Ranker Training

## Overview
`neureq_ph2a.py` executes the training phase of the NEUREQ framework's second stage. This script defines and trains a custom Bidirectional Long Short-Term Memory (BiLSTM) network with an Additive Attention mechanism. 

The goal of this network is to predict a final clinical relevance score by modeling the structural dependencies between the 10 specific eligibility questions generated during Phase 1. By learning from a synthetic dataset of positive matches and hard negatives, the model acts as an intelligent aggregator, evaluating how explicit medical mismatches (e.g., failed exclusion criteria) interact with positive matches to yield a final Patient-Trial compatibility score.

## Technical Architecture

* **Text Encoder:** A frozen `Bio_ClinicalBERT` (`emilyalsentzer/Bio_ClinicalBERT`) extracts 768-dimensional semantic embeddings from the LLM-generated textual justifications. Freezing the encoder ensures computational efficiency and prevents catastrophic forgetting on a small dataset.
* **Feature Representation:**
  * **Question ID Embedding:** Categorical (0-9) mapped to 8 dimensions.
  * **Answer Embedding:** Categorical (YES/NO/NA mapped to 0/1/2) mapped to 3 dimensions.
  * **Justification:** BERT `[CLS]` token (768 dimensions).
* **Core Network:** A 1-layer BiLSTM (hidden size 64) processes the concatenated 779-dimensional input sequence. Bidirectionality ensures that late-sequence criteria (e.g., Treatment Alignment) can contextualize early-sequence criteria (e.g., Age/Gender).
* **Interpretability Engine:** An Additive Attention layer applies weights to the BiLSTM hidden states. This is technically essential: it provides a quantitative mechanism to trace *which* clinical eligibility questions the network heavily relied upon to make its final prediction, satisfying the need for an interpretable information system.

## File Dependencies

### Inputs
1. **`triplet_syn_dataset_1196.jsonl`**: The synthetic training dataset containing 1,196 labeled query-document pairs (positive matches vs. hard negatives generated via BM25 criteria filtering). The file must contain the structured 10-question JSON objects.

### Outputs
1. **`models_new/BiLSTM_ADD_ATTN_1196/model_epoch_XX.pt`**: The serialized PyTorch state dictionaries containing the model weights. The script saves a checkpoint at the end of every epoch.
2. **`models_new/BiLSTM_ADD_ATTN_1196/training_log.txt`**: A plaintext log containing epoch-wise Train Loss, Validation Loss, and Validation ROC-AUC metrics.

## Data Flow & Pipeline Steps

1. **Initialization:** Device logic checks for a CUDA GPU. `Bio_ClinicalBERT` is loaded into VRAM and frozen via `torch.no_grad()`.
2. **Data Parsing (`EligibilityDataset`):** * Reads the JSONL file.
   * Maps string responses ("YES", "NO", "NA") to integer indices.
   * Generates `[CLS]` embeddings for the natural language justifications on the fly.
3. **Data Loading:** Shuffles and splits the dataset (80% Train, 20% Validation). Batches the tensors for the neural network.
4. **Forward Pass:** The BiLSTM integrates the embeddings. The Additive Attention layer condenses the sequence into a single context vector, and the feed-forward sequence outputs a single logit representing relevance.
5. **Backpropagation:** Computes Binary Cross-Entropy Loss with Logits (`BCEWithLogitsLoss`), updates gradients via the Adam optimizer.
6. **Validation:** Evaluates unseen data to compute Loss and the ROC-AUC score, preventing model overfitting and logging the best epoch dynamically.

## Usage
Ensure your dataset paths in the global variables correspond to the correct localized environment.

```bash
python neureq_ph2a.py