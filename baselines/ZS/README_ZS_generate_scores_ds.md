# Baseline: Zero-Shot LLM Scoring (`ZS_generate_scores_ds.py`)

## Overview
This script implements the **LLM-ZS (Zero-Shot)** baseline evaluated in Section 4.2 of the NEUREQ paper. It evaluates clinical trial relevance by prompting a frozen Large Language Model (`DeepSeek-R1-Distill-Qwen-32B`) to act as a direct, point-wise scorer. 

Unlike the proposed NEUREQ framework—which decomposes reasoning into ten structured eligibility questions—this baseline forces the LLM to output a single scalar value between `0.0` and `1.0` representing overall patient-trial compatibility.

## Analytical Context: The "Tie-Breaking" Limitation
As detailed in Section 5.8 of the paper ("Failure Analysis"), prompting an LLM to generate raw scalars for complex clinical matching is inherently flawed. The LLM lacks a structured mechanism to aggregate conflicting clinical signals (e.g., matching diagnosis but failing a specific lab exclusion).

Consequently, the zero-shot LLM frequently assigns **identical scores** to many relevant and non-relevant trials within the same candidate list. Because Python's `list.sort()` is stable, these ties are broken arbitrarily based on their initial retrieval order, severely preventing meaningful relevance differentiation and degrading evaluation metrics (like NDCG@10 and MAP) compared to NEUREQ.

## Technical Architecture

* **Model:** `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B`
* **Precision:** 16-bit Float (`torch.float16`) to fit the 32B model into VRAM.
* **Prompt Strategy:** The prompt explicitly forbids the model from generating natural language explanations or chain-of-thought reasoning, forcing a direct scalar output (e.g., `0.85`).
* **Output Parsing:** Uses a strict Regular Expression (`\b(0\.\d{1,5}|1\.0{1,5})\b`) to isolate the predicted score, defaulting to `0.0` if the model hallucinates or fails to follow instructions.

## File Dependencies

### Inputs
1. **First-Stage Retrieval Run:** (e.g., `runs/2021/FIRST_STAGE/WholeQ_RM3_RETRIEVAL_T2022.txt`). Defines the top 100 candidate trials to re-rank per query.
2. **Queries:** `data/2021/ct_2021_queries.tsv` (Patient case descriptions).
3. **Corpus:** `data/clinicaltrials/2023/corpus.jsonl` (Clinical trial full texts).

### Outputs
1. **TREC Run File:** `runs/2021/ZERO_SHOT/[RUN_FILE_NAME]_deepseek_zero_shot.txt`
   * Formatted strictly for `trec_eval` processing: `query_id Q0 doc_id rank score run_name`

## Usage

Ensure your paths in the `=== Paths Configuration ===` block are pointed to the correct dataset year (2021 or 2022) before executing.

```bash
python ZS_generate_scores_ds.py