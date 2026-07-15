# MFT-MonoT5 Baseline (Step 3/4): Multi-Field Template Segment Scoring (`mft5_3.py`)

## Overview
This script executes the neural scoring phase of the **MFT-MonoT5** baseline evaluated in Section 4.2 of the NEUREQ paper.

A significant limitation of standard transformer-based re-rankers (like T5 or BERT) is their strict 512-token context limit, which is severely inadequate for lengthy clinical trial protocols. This script bypasses this limitation using a **"MaxP" Sliding Window** approach. It parses the candidate clinical trials into distinct fields, slices those fields into overlapping text segments, and scores the relevance of each segment independently using a massive 3-Billion parameter `MonoT5` model. 

This approach serves as a highly robust, state-of-the-art baseline against which NEUREQ’s structured, LLM-based eligibility reasoning is compared.

## Technical Architecture & Mathematical Context

* **Model:** `castorini/monot5-3b-med-msmarco` (A 3-Billion parameter T5-based seq2seq model fine-tuned on MS MARCO and medical data).
* **Sliding Window:** To ensure context isn't lost at arbitrary token cutoffs, the script splits fields into 6-sentence windows, advancing by 3 sentences per step (creating a 50% overlap).
* **MonoT5 Logic & Softmax Extraction:** MonoT5 frames ranking as a sequence-to-sequence text generation task. It is trained to output the token `"true"` if the document is relevant to the query, and `"false"` if not. Instead of just generating text, this script intercepts the raw network logits for the `"true"` and `"false"` tokens and calculates a normalized probability score:
  
  $P(Relevant) = \frac{e^{logit\_true}}{e^{logit\_true} + e^{logit\_false}}$

## File Dependencies

### Inputs
1. **Candidate Ranked List:** The output file from Step 2 (e.g., `data/2022/bm25rm3_rrf_nqs_pd.txt`). Defines the top 100 pairs to evaluate.
2. **Patient Queries:** `data/2022/ct_2022_queries.tsv` (The original evaluation queries).
3. **Corpus:** `data/clinicaltrials/corpus.jsonl` (Used to extract the raw trial text).

### Outputs
* **Segment Scores Cache:** `data/2022/sigir/segment_scores_WholeQ_RM3_T2022.jsonl`.
  * *Format:* A JSONL file where every single sentence segment gets its own entry.
  * *Schema:*
    ```json
    {
      "topic_id": "1",
      "trial_id": "NCT0123456",
      "field": "eligibility",
      "segment": "Patients must be > 18 years old. History of asthma is excluded...",
      "score": 0.875
    }
    ```

## Usage & Configuration

Ensure the `ranked_list_file` in the Configuration block is pointed to the specific Step 2 candidate list you wish to evaluate.

```bash
python mft5_3.py