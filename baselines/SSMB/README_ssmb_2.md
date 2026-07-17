# Baseline: Self-Supervised MonoBERT Inference & Score Interpolation (`ssmb_2.py`)

## Overview
This script executes the second and final phase of the Self-Supervised MonoBERT (SSMB) baseline pipeline. It takes an initial candidate list of clinical trials retrieved by a lexical BM25 engine and re-ranks them using a combination of sparse lexical features and deep semantic features. 

The fine-tuned model from Step 1 functions as a Cross-Encoder, reading both the extracted trial summary and the patient query simultaneously to assign a classification score. The script normalizes the baseline BM25 scores and combines them linearly with the neural cross-encoder probabilities, outputting a highly structured rank file suitable for standardized IR evaluation toolkits like `trec_eval`.

## Technical Architecture
* **Cross-Attention Interaction:** The script pairs the extracted `Brief Summary` (acting as the pseudo-patient case learned during self-supervision) with the real target `Patient Query`. By evaluating them within a single context window, the model leverages full cross-attention between every token in the query and trial summary.
* **Min-Max Score Normalization:** Because raw BM25 logs use an open-ended log-frequency scale and BERT outputs bounded probabilities, raw combinations are mathematically incompatible. The script scales the BM25 values for each independent query strictly into a $[0, 1]$ span.
* **Linear Interpolation Framework:** The final ranking utilizes a weighted linear mixture:
  $$\text{Score}_{\text{final}} = (0.1 \times \text{Score}_{\text{BM25\_Normalized}}) + (0.9 \times \text{Score}_{\text{BERT\_Probability}})$$
* **Batch Execution:** Candidate sets are parsed in configurable batches (default: `16`) with a context truncation floor of `256` tokens to eliminate padding overhead.

## File Dependencies & Formats

### Inputs

1. **Patient Topics File (`data/clinicaltrials/ct_2021_queries.tsv`)**
   * **Format:** Tab-separated values (`.tsv`).
   * **Structure:** Each line must contain an alphanumeric query/topic ID, a literal tab separator, and the raw clinical description text.
   * **Example:**
     ```text
     101   A 45-year-old male presenting with chronic myeloid leukemia resistant to imatinib.
     ```

2. **Global Corpus Collection (`data/clinicaltrials/corpus.jsonl`)**
   * **Format:** JSON Lines (`.jsonl`).
   * **Structure:** Must contain a unique trial ID (`id`) and a comprehensive raw string field (`contents`). The script applies a regex parser (`r"Brief Summary:(.*?)(?:\n[A-Z][a-z]+:|\Z)"`) to extract just the summary. If the explicit section header is missing, it captures the full text as a fallback.

3. **First-Stage Retrieval Run (`data/2021/WholeQ_RM3_RETRIEVAL_T2021.txt`)**
   * **Format:** Standard space-delimited TREC format (`.txt`).
   * **Columns:** `[Query_ID] [Iteration] [Doc_ID] [Rank] [Raw_Score] [Run_Name]`
   * **Example:**
     ```text
     101 Q0 NCT00001234 1 24.897100 BM25_Baseline
     ```

4. **Model Parameters (`models/self_supervised_monobert/epoch-1`)**
   * **Format:** Hugging Face weight directory containing `config.json` and `model.safetensors` compiled in Step 1.

### Outputs

1. **Interpolated Re-ranked Run (`runs/2021/SSMONOBERT_baseline/BM25_CT2021.txt`)**
   * **Format:** 6-column space-delimited TREC ranking file.
   * **Columns:** `[Query_ID] Q0 [Trial_ID] [New_Rank] [Interpolated_Score] SSMonoBERT`
   * **Example:**
     ```text
     101 Q0 NCT00001234 1 0.945120 SSMonoBERT
     101 Q0 NCT00005678 2 0.812304 SSMonoBERT
     ```

## Usage
Confirm your first-stage retrieval run text file exists and that the Step 1 checkpoint weights are stored at `models/self_supervised_monobert/epoch-1`.

Run the re-ranking task through the terminal:
```bash
python ssmb_2.py