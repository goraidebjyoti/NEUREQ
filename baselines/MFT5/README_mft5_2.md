# MFT-MonoT5 Baseline (Step 2/4): Retrieval & Reciprocal Rank Fusion (`mft5_2.py`)

## Overview
This script executes the second phase of the **MFT-MonoT5** baseline evaluated in Section 4.2 of the NEUREQ paper.

In Step 1, Neural Query Synthesis (NQS) expanded each patient case description into 40 distinct synthetic queries. This script takes those 40 queries, executes searches against the Pyserini clinical trial index using both standard BM25 and RM3 pseudo-relevance feedback, and merges the resulting ranked lists into a single, cohesive candidate pool using **Reciprocal Rank Fusion (RRF)**.

These fused lists serve as the highly competitive first-stage candidates that the MonoT5 3B model will subsequently re-rank in Step 3.

## Technical Architecture & Mathematical Context

* **Retrieval Engine:** Uses `pyserini.search.lucene.LuceneSearcher` to interface with the pre-built BM25 index of the ClinicalTrials.gov corpus.
* **RM3 Pseudo-Relevance Feedback:** We configure RM3 to extract the top 10 terms from the top 10 retrieved documents, mixing them with the original query at a 50/50 ratio. This mitigates the risk of a synthetic query being overly narrow.
* **Reciprocal Rank Fusion (RRF):** Because BM25 scores across 40 different synthetic queries are uncalibrated (e.g., a score of 12.5 on Query A is not mathematically comparable to a score of 12.5 on Query B), we cannot average them. Instead, we use RRF, which relies purely on the rank position:
  
  $RRF\_Score(d) = \sum_{q \in Q} \frac{1}{k + rank(d, q)}$
  
  *(where $k=60$ is a standard smoothing constant preventing the highest-ranked documents from dominating the score).* Documents that consistently appear highly ranked across multiple synthetic variations will float to the top of the fused list.

## File Dependencies

### Inputs
1. **Pyserini Index:** `indexes/clinical_trials` (Must be generated prior via Pyserini's indexing tools over the JSONL corpus).
2. **Synthetic Queries:** `data/2021/synthetic_queries_2021.tsv` (Output from Step 1).
3. **Original Patient Queries:** `data/2021/ct_2021_queries.tsv` (Used to anchor the synthetic queries with the original context).

### Outputs
The script generates four TREC-formatted text files representing different search strategies. These files dictate which trials the massive 3B model will evaluate in Step 3.
1. `data/2021/bm25_rrf_nqs.txt` (BM25, Synthetic Only)
2. `data/2021/bm25rm3_rrf_nqs.txt` (BM25+RM3, Synthetic Only)
3. `data/2021/bm25_rrf_nqs_pd.txt` (BM25, Synthetic + Original Query)
4. `data/2021/bm25rm3_rrf_nqs_pd.txt` (BM25+RM3, Synthetic + Original Query)

## Usage

Ensure the input file paths are pointed to the correct dataset year (e.g., 2021 or 2022). The script runs locally on the CPU and requires the Java runtime environment for Pyserini/Lucene.

```bash
python mft5_2.py