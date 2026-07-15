# Data Construction: Hard Negative Mining (`create_hardNegative.py`)

## Overview
This script is responsible for synthesizing the training dataset used to train the Phase 2 neural re-ranker (`EligibilityBiLSTM`). It operates by evaluating candidate clinical trials retrieved by a classical BM25 search engine and algorithmically isolating trials that are classified as "Hard Negatives." 

The output is a collection of data triplets—consisting of a Patient Query, a known Relevant Trial, and a Hard Negative Trial—which forces the neural network to learn complex clinical relevance rather than simple keyword matching.

## Data Provenance & Attribution
The foundation of this dataset relies on resources provided by the broader clinical information retrieval community, which this script synthesizes into a novel training artifact:

* **IELAB Synthetic Resources:** The base patient case descriptions (`synthetic_gold_queries.tsv`), the known positive ground-truth pairings (`synthetic_gold_qrels.txt`), and the initial first-stage retrieval rankings (`generated_train_v2_with_gold.bm25.k1=0.82.b=0.68.tsv` containing the top 200 BM25 candidate trials per query) were provided by the IELAB team.
* **Clinical Trial Corpus:** The raw trial documents (`corpus.jsonl`) are drawn from the official TREC Clinical Trials Track 2023 shared corpus, containing a snapshot of 448,528 trials.
* **Our Contribution:** While IELAB provided the positive associations and BM25 candidates, this script represents our novel contribution: leveraging LLM-driven eligibility reasoning to programmatically filter the 200 BM25 candidates down to the absolute most deceptive "Hard Negatives" (trials with high lexical similarity but critical clinical mismatches).

## The Contribution: Why Do We Need Hard Negatives?
In standard Information Retrieval (IR) tasks, a model is trained using positive matches and randomly sampled "easy" negatives. However, clinical trial matching is highly complex. An "easy negative" might be a breast cancer trial for a patient with prostate cancer—a model can easily learn to discard this based on a lack of overlapping vocabulary. 

A **Hard Negative** is a trial that shares heavy lexical overlap with the patient's description (causing it to be ranked highly by initial keyword systems like BM25) but is ultimately incompatible due to strict clinical constraints. For example:
* The patient has the exact correct disease and matches the age demographics.
* However, the patient's liver enzyme levels violate a highly specific exclusion criterion (Q6).
* Or, the patient has undergone a prior therapy that disqualifies them from the study protocol (Q5).

**The value of this script is that it forces the downstream BiLSTM to look past vocabulary.** By explicitly selecting negatives where the LLM has evaluated `NO` to critical questions—*Condition Relevance (Q3), Prior Treatment Consideration (Q5), Inclusion/Exclusion Criteria (Q6), and Treatment Target Alignment (Q10)*—the BiLSTM is trained to weigh structural contradictions heavily. It learns that a single violated exclusion criterion can invalidate dozens of positive lexical matches, aligning the model's behavior with real-world clinical decision-making.

## Technical Architecture & Flow

1. **Data Ingestion:** Loads synthetic patient queries, known positive associations (QRELs), and a candidate list of trials retrieved by BM25.
2. **Sampling & State Management:** Samples batches of 300 queries, maintaining strict tracker files (`processed_queries.txt`) to ensure fault tolerance and prevent duplicate API/GPU compute in the event of an interruption.
3. **LLM Evaluation:** For a given query, the script iterates down the BM25 ranked list of candidate trials. It prompts `DeepSeek-R1-Distill-Qwen-32B` to generate the 10-question eligibility evaluation.
4. **Strict Filtering:** The script parses the JSON response and checks the specific criteria:
   ```python
   if all(cleaned.get(q, {}).get("response") == "NO" for q in ["3", "5", "6", "10"])