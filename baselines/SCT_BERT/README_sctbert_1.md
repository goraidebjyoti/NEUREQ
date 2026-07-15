# SCT-BERT Baseline (Step 1/4): Positive Pair Extraction (`sctbert_1.py`)

## Overview
This script is the first step of the **SCT-BERT (Self-Supervised MonoBERT)** baseline evaluation. Its primary objective is to transform the massive, unstructured ClinicalTrials.gov corpus into a clean, paired dataset that can be used for self-supervised training.

Rather than waiting for expert-annotated clinical patient queries, this baseline assumes that a clinical trial's own `Brief Summary` (which describes the goal/population) and its `Eligibility Criteria` (which describes the requirements) form a natural "relevance pair." This script extracts these two fields from the corpus and pairs them to create positive samples (`label=1`) for the model.

## Technical Architecture
* **Regex Parsing Engine:** Clinical trial raw text is not strictly structured. This script uses non-greedy Regex-based boundary detection to isolate content sections. It identifies a section header (e.g., "Inclusion Criteria:") and extracts the subsequent text until the *next* identifiable section header (e.g., "Status:") is detected.
* **Filtering:** Trials missing either a Brief Summary or Eligibility Criteria are discarded, as they cannot serve as meaningful self-supervised pairs.

## File Dependencies

### Inputs
* **Full Clinical Trial Corpus:** `data/clinicaltrials/json_corpus/corpus.jsonl`
  * *Schema:* A JSONL file containing `id` and `contents` (the full raw text block of the trial).

### Outputs
* **Positive Pairs Cache:** `data/clinicaltrials/positive_pairs.jsonl`
  * *Format:* JSON Lines.
  * *Schema:* ```json
    {
      "trial_id_summary": "NCT123456",
      "trial_id_criteria": "NCT123456",
      "summary": "This study examines the efficacy of...",
      "criteria": "Inclusion: Patients aged 18+ with Stage II cancer...",
      "label": 1
    }
    ```

## Usage
Simply point the `input_file` and `output_file` paths to your local corpus structure and execute:

```bash
python sctbert_1.py