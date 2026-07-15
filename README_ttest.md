# Evaluation & Statistical Significance Testing (`ttest.py`)

## Overview
`ttest.py` is the final quantitative evaluation script in the NEUREQ pipeline. It is designed to automatically generate the formal performance tables presented in the project's research paper. 

The script calculates standard Information Retrieval (IR) metrics (MAP, Precision, Recall, NDCG) for the proposed framework and all evaluated baselines. Critically, it executes a paired Student's t-test to rigorously establish whether NEUREQ's performance gains are statistically significant ($p \le 0.05$) compared to the baseline models.

## Technical Architecture
* **Metric Calculation:** The script uses the `subprocess` module to interface directly with the standard `trec_eval` command-line toolkit. Passing the `-q` flag allows the script to retrieve per-query scores, which are mathematically required to run a paired statistical test.
* **Statistical Testing:** Utilizes `scipy.stats.ttest_rel` to perform a paired two-sided t-test. The test compares the vector of query scores from a baseline model against the vector of query scores from the NEUREQ model.
* **Formatting:** The script automatically generates publication-ready ASCII tables. It maps statistically significant improvements to superscript letters corresponding to the defeated baselines, adhering to standard IR conference formatting conventions.

## File Dependencies

### Inputs
1. **`ct_[YEAR]_qrels_mapped.txt`**: The official TREC gold-standard relevance judgments mapping patient queries to relevant clinical trials.
2. **`runs5/[YEAR]/[MODEL_DIR]/[TEST_TYPE].txt`**: The TREC-formatted run files containing the ranked predictions. This directory must contain the outputs from NEUREQ (Phase 2b) as well as all baseline models being evaluated (e.g., Zero-Shot LLM, MonoBERT, CT-BERT).

### Outputs
1. **`t_test_results/[TEST_TYPE].txt`**: The final formatted evaluation tables containing the metric means and superscript significance indicators.

## Usage
Ensure the `trec_eval` binary is installed and globally accessible in your system's PATH.

```bash
python ttest.py