# Data Construction: Hard Negative Mining (`create_hardNegative.py`)

## Overview

This script is responsible for synthesizing the training dataset used to train the Phase 2 neural re-ranker (`EligibilityBiLSTM`). It operates by evaluating candidate clinical trials retrieved by a classical BM25 search engine and algorithmically isolating trials that are classified as "Hard Negatives."

The output is a collection of data triplets—consisting of a Patient Query, a known Relevant Trial, and a Hard Negative Trial—which forces the neural network to learn complex clinical relevance rather than simple keyword matching.

---

## Data Provenance & Attribution

The foundation of this dataset relies on resources provided by the broader clinical information retrieval community, which this script synthesizes into a novel training artifact:

- **IELAB Synthetic Resources:** The base patient case descriptions (`synthetic_gold_queries.tsv`), the known positive ground-truth pairings (`synthetic_gold_qrels.txt`), and the initial first-stage retrieval rankings (`generated_train_v2_with_gold.bm25.k1=0.82.b=0.68.tsv` containing the top 200 BM25 candidate trials per query) were provided by the IELAB team.

- **Clinical Trial Corpus:** The raw trial documents (`corpus.jsonl`) are drawn from the official TREC Clinical Trials Track 2023 shared corpus, containing a snapshot of 448,528 trials.

- **Our Contribution:** While IELAB provided the positive associations and BM25 candidates, this script represents our novel contribution: leveraging LLM-driven eligibility reasoning to programmatically filter the 200 BM25 candidates down to the absolute most deceptive "Hard Negatives" (trials with high lexical similarity but critical clinical mismatches).

---

## The Contribution: Why Hard Negatives are required?

In standard Information Retrieval (IR) tasks, a model is trained using positive matches and randomly sampled "easy" negatives. However, clinical trial matching is highly complex. An "easy negative" might be a breast cancer trial for a patient with prostate cancer—a model can easily learn to discard this based on a lack of overlapping vocabulary.

A **Hard Negative** is a trial that shares heavy lexical overlap with the patient's description (causing it to be ranked highly by initial keyword systems like BM25) but is ultimately incompatible due to strict clinical constraints. For example:

- The patient has the exact correct disease and matches the age demographics.
- However, the patient's liver enzyme levels violate a highly specific exclusion criterion (Q6).
- Or, the patient has undergone a prior therapy that disqualifies them from the study protocol (Q5).

**The value of this script is that it forces the downstream BiLSTM to look past vocabulary.** By explicitly selecting negatives where the LLM has evaluated `NO` to critical questions—*Condition Relevance (Q3), Prior Treatment Consideration (Q5), Inclusion/Exclusion Criteria (Q6), and Treatment Target Alignment (Q10)*—the BiLSTM is trained to weigh structural contradictions heavily. It learns that a single violated exclusion criterion can invalidate dozens of positive lexical matches, aligning the model's behavior with real-world clinical decision-making.

---

## Technical Architecture & Flow

1. **Data Ingestion:** Loads synthetic patient queries, known positive associations (QRELs), and a candidate list of trials retrieved by BM25.

2. **Sampling & State Management:** Samples batches of 300 queries, maintaining strict tracker files (`processed_queries.txt`) to ensure fault tolerance and prevent duplicate API/GPU compute in the event of an interruption.

3. **LLM Evaluation:** For a given query, the script iterates down the BM25 ranked list of candidate trials. It prompts `DeepSeek-R1-Distill-Qwen-32B` to generate the 10-question eligibility evaluation.

4. **Strict Filtering:** The script parses the JSON response and checks the specific criteria:

   ```python
   if all(cleaned.get(q, {}).get("response") == "NO" for q in ["3", "5", "6", "10"]):
       # Save as Hard Negative
   ```

---

## File Dependencies & Formats

### Inputs (Required)

These files must be present in the designated paths before running the script.

#### Patient Queries (`synthetic_gold_queries.tsv`)

**Format:** Tab-Separated Values (TSV) without headers.

**Columns:**

```text
topic_id    query_text
```

---

#### Gold Relevance Judgments (`synthetic_gold_qrels.txt`)

**Format:** Standard TREC QREL space-separated text format.

**Columns:**

```text
topic_id 0 trial_id relevance_score
```

---

#### First-Stage Retrieval Run (`generated_train_v2_with_gold.bm25.k1=0.82.b=0.68.tsv`)

**Format:** Tab-Separated Values (TSV) without headers.

**Columns:**

```text
topic_id    trial_id    rank
```

---

#### Clinical Trial Corpus (`corpus.jsonl`)

**Format:** JSON Lines (JSONL).

**Schema:**

```json
{
  "id": "NCT01234567",
  "contents": "Full raw text of the clinical trial..."
}
```

---

#### LLM Prompt (`prompt2.txt`)

**Format:** Plain text template containing the specific generation instructions and formatting constraints for the LLM.

---

### Outputs (Generated)

These files are created and continuously appended to during script execution.

#### Core Hard Negatives (`hard_negatives.jsonl`)

**Description:** The finalized, minimal dataset used for downstream training construction.

**Format:** JSON Lines (JSONL).

**Schema:**

```json
{
  "topic_id": 125,
  "query": "Patient text...",
  "trial_id": "NCT01234567"
}
```

---

#### Audited Hard Negatives (`hard_negatives_with_llm.jsonl`)

**Description:** Includes the full raw text generated by the LLM for interpretability analysis and debugging.

**Format:** JSON Lines (JSONL).

**Schema:**

```json
{
  "topic_id": 125,
  "query": "Patient text...",
  "trial_id": "NCT01234567",
  "llm_response": "Raw text output including <think> tags..."
}
```

---

#### Error Logs (`skipped_responses.jsonl`)

**Description:** Logs trials where the LLM generated malformed JSON or failed formatting instructions.

**Format:** JSON Lines (JSONL).

**Schema:**

```json
{
  "topic_id": 125,
  "trial_id": "NCT01234567",
  "reason": "JSON parsing failed",
  "raw_response": "..."
}
```

---

## State Tracking Files

These files manage the processing state to allow the script to be paused and resumed without duplicating GPU workloads.

- **`selected_queries.txt`**: A plain text list of `topic_ids` that have been queued for processing.

- **`processed_queries.txt`**: A plain text list of `topic_ids` that have successfully yielded a hard negative and are complete.