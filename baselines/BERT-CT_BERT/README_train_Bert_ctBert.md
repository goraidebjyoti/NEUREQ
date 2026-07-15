# Baseline Training: BERT & CT-BERT Cross-Encoders (`train_Bert_ctBert.py`)

## Overview
This script is responsible for training the deep learning baselines evaluated in Section 4.2 of the NEUREQ paper. Specifically, it trains the **SIMPLE_BERT** and **CT_MLM_BERT** (PubMedBERT-CT-MLM) re-ranking models. 

Unlike the NEUREQ framework—which extracts structured, clinically motivated eligibility criteria using an LLM—these baselines operate as traditional cross-encoders. They concatenate the raw text of the patient case description and the clinical trial, feed the combined sequence through a transformer, and train a regression head on the `[CLS]` token to predict a relevance score.

Including and rigorously evaluating these baselines demonstrates that NEUREQ's performance gains stem from its explicit, structured eligibility reasoning, rather than simply the use of deep neural networks.

## Technical Architecture

* **Input Formatting:** The script flattens the training triplets into isolated pairs. Using the HuggingFace `AutoTokenizer`, the queries and trials are combined into the standard sequence: `[CLS] Patient Query [SEP] Clinical Trial [SEP] [PAD]`.
* **Token Limits:** Because standard BERT architectures have a hard 512-token limit, the script truncates the patient query to a maximum of 179 tokens and the trial text to 330 tokens. (This token restriction is a primary limitation of standard BERT models when dealing with lengthy clinical documents).
* **Model Head:** A `Linear` regression layer maps the 768-dimensional `[CLS]` token hidden state to a continuous 1D scalar.
* **Loss Function:** The model is optimized using Mean Squared Error (`MSELoss`), targeting `1.0` for positive pairs and `0.0` for hard negatives.

## File Dependencies

### Inputs
* **Dataset:** `data/train/triplet_syn.jsonl`. A text-only variant of the 1196 synthetic triplet dataset. 
  * *Schema Requirements:* Each JSON line must contain `topic` (patient text), `positive_trial` (relevant trial text), and `negative_trial` (hard-negative trial text).

### Outputs
* **Preprocessed Pairs:** `data/train/train_dataset.csv`. An intermediate CSV containing the flattened and shuffled (Query, Document, Label) pairs.
* **Trained Weights:** `models/[MODEL_TYPE]/bert_regression_model.pt`. The serialized PyTorch state dictionary. 
  * If using standard BERT, it saves to `models/SIMPLE_BERT/`.
  * If using PubMedBERT, it saves to `models/CT_MLM_BERT/`.

## Usage & Configuration

To train the respective baseline models, open the script and modify the `MODEL_NAME` constant in the `CONFIGURATION` block.

**To train the Standard BERT Baseline:**
```python
MODEL_NAME = "google-bert/bert-base-uncased"