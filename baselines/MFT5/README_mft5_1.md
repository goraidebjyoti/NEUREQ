# MFT-MonoT5 Baseline (Step 1/4): Neural Query Synthesis (`mft5_1.py`)

## Overview
This script represents the first step of the **MFT-MonoT5** baseline evaluated in Section 4.2 of the NEUREQ paper. 

Retrieving clinical trials based entirely on a raw patient description is notoriously difficult because patient cases and clinical trials use different vocabularies (e.g., a patient case might mention "heart attack", while a trial looks for "myocardial infarction"). 

To address this "vocabulary mismatch," this script implements **Neural Query Synthesis (NQS)**. It uses a sequence-to-sequence model (`doc2query-t5-base-msmarco`) to read the patient case and dynamically generate 40 diverse, synthetic search queries that a human might realistically type into a search engine. These synthetic queries act as a semantic expansion of the original patient case, providing a much broader foundation for the first-stage retrieval step.

## Technical Architecture

* **Model:** `castorini/doc2query-t5-base-msmarco`. A T5 model fine-tuned on the MS MARCO passage ranking dataset to generate queries from documents.
* **Input Token Limit:** Truncated to 512 tokens to handle lengthy clinical narratives.
* **Sampling Strategy:** Uses `do_sample=True` with `top_k=10`. Standard greedy decoding would produce 40 nearly identical strings. By sampling from the top 10 most probable next tokens, the model generates 40 semantically related but lexically diverse queries.

## File Dependencies

### Inputs
* **Patient Case Descriptions:** `data/2022/ct_2022_queries.tsv`
  * *Format expected:* A tab-separated file without headers containing `<topic_id> \t <topic_text>`.
  * *Example:* `1 \t A 58-year-old female presents with stage II breast cancer...`

### Outputs
* **Synthetic Queries File:** `data/2022/synthetic_queries_2022.tsv`
  * *Format:* A tab-separated file WITH headers containing: `topic_id`, `topic_text`, and `synthetic_query`.
  * *Volume:* Generates `N * 40` rows, where `N` is the number of patient topics.

## Usage

Ensure the `queries_file` variable is pointed to the correct evaluation year (e.g., 2021 or 2022).

```bash
python mft5_1.py