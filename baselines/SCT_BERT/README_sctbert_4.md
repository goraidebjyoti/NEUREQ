# SCT-BERT Baseline (Step 4/4): Zero-Shot Inference & Re-ranking (`sctbert_4.py`)

## Overview
This script performs the evaluation/inference phase of the **SCT-BERT (Self-Supervised MonoBERT)** baseline described in Section 4.2 of the NEUREQ paper.

It applies the neural weights fine-tuned in Step 3 to real, unseen patient queries (TREC Clinical Trials Track datasets). The script evaluates candidate trials retrieved by classical first-stage models (e.g., BM25 or BM25+RM3) and overrides their lexical rankings with deep contextual relevance scores, outputting the final re-ranked list for statistical evaluation.

## Technical Architecture: The "Zero-Shot Domain Shift"
The most critical aspect of this script is its **Zero-Shot Transfer** paradigm. 
During training, the model learned a surrogate classification task: assessing if a trial's `Brief Summary` matched a set of `Eligibility Criteria`. 

However, during inference, the `Eligibility Criteria` text is entirely replaced by the real, unformatted `Patient Case Query`. Because the model spent Step 3 learning the fundamental semantic interactions between clinical terminology, symptoms, and demographic constraints, it is able to successfully perform a zero-shot domain shift, directly calculating the compatibility probability of a `(Summary, Patient Query)` pair without ever having seen one during training.

* **Relevance Probability Calculation:** The output logits of the binary classification head are passed through a `Softmax` function. The script specifically isolates the probability of `label=1`, yielding a continuous relevance score between 0.0 and 1.0.

## File Dependencies

### Inputs
1. **Fine-Tuned Model Weights:** `models/sct_bert/` (Generated in Step 3).
2. **Official Evaluation Queries:** `data/[YEAR]/ct_[YEAR]_queries.tsv`.
3. **First-Stage Retrieval Run:** `data/[YEAR]/WholeQ_RM3_RETRIEVAL_T[YEAR].txt` (The top-K candidates to re-rank).
4. **Trial Summaries:** `data/clinicaltrials/positive_pairs.jsonl` (Used as a fast O(1) dictionary to fetch the text summaries for candidate trials).

### Outputs
* **Evaluated TREC Run File:** `runs/[YEAR]/sctbert/[RUN_NAME].txt`.
  * *Format:* `topic_id Q0 trial_id rank score SCTBERT`
  * This sorted, standardized text file is ready to be parsed by `trec_eval` to calculate NDCG, Precision, and Recall metrics.

## Usage & Configuration

Ensure the `MODEL_DIR`, `TOPICS_FILE`, and `CANDIDATE_RUN` paths point to the correct evaluation year and retrieval configuration before execution.

```bash
python sctbert_4.py