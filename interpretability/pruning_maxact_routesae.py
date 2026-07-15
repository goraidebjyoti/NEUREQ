# pruning_maxact_routesae.py

"""
NEUREQ Interpretability: Approximate Pruning-MaxAct
=================================================

This script executes the final mechanistic interpretability step detailed in 
Section 3.3 and 3.4 of the NEUREQ framework. It bridges the gap between raw mathematical 
tensors and human-understandable clinical concepts.

Process:
1. Loads the sparse latent activations (`all_z.pt`) and the global discriminability 
   rankings (`feature_summary.json`).
2. Selects the most highly discriminative features (both positive and negative).
3. For each feature, identifies the top 20 patient-trial pairs that caused the highest activation.
4. Reconstructs the exact evaluation prompt.
5. Applies a heuristic "Approximate Pruning-MaxAct" procedure to extract the top 5 
   most informative sentences from the prompt.
6. Exports these concise evidence snippets for manual semantic interpretation.

This avoids the massive computational cost of running exact gradient-based pruning 
on a 32B model, instead relying on keyword-based heuristics to isolate clinical evidence.
"""

import os
import re
import json
from collections import defaultdict

import torch
import pandas as pd
from tqdm import tqdm

# =============================================================================
# GPU CONFIGURATION
# =============================================================================
# GPU is primarily used here for fast tensor operations (topk) on the activation matrix
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# =============================================================================
# INPUT FILES (EDIT THESE TO MATCH YOUR INFERENCE RUN)
# =============================================================================
# Directory containing the outputs from infer_routesae.py
ROUTESAE_RUN_DIR = "routesae_inference/WholeQ_RETRIEVAL_T2022"

# The raw N x M sparse activation tensor
ALL_Z_FILE = os.path.join(
    ROUTESAE_RUN_DIR,
    "features",
    "all_z.pt"
)

# Global test-set statistics containing the top positive/negative feature indices
FEATURE_SUMMARY_FILE = os.path.join(
    ROUTESAE_RUN_DIR,
    "features",
    "feature_summary.json"
)

# Mapping of row indices to patient-trial IDs
SAMPLE_IDS_FILE = os.path.join(
    ROUTESAE_RUN_DIR,
    "hidden_states",
    "sample_ids.json"
)

# Raw text sources required to reconstruct the prompts
TOPICS_TSV = "data/2022/ct_2022_queries.tsv"
CORPUS_JSONL = "data/clinicaltrials/json_corpus/corpus.jsonl"

# =============================================================================
# OUTPUT DIRECTORY
# =============================================================================
# Stores the individual feature evidence files and the final annotation template
OUTPUT_DIR = os.path.join(ROUTESAE_RUN_DIR, "pruning_maxact")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# PRUNING PARAMETERS (Directly mapped to methodology)
# =============================================================================
# Total number of unique features to extract for manual review (e.g., top 25 pos + top 25 neg)
NUM_FEATURES_TO_ANALYZE = 50

# As stated in the methodology: "we examine the top 20 highest activating patient-trial pairs"
TOP_EXAMPLES_PER_FEATURE = 20

# As stated in the methodology: "The five highest-scoring sentences are retained as candidate evidence spans"
TOP_SENTENCES_PER_EXAMPLE = 5


# =============================================================================
# EXACT PROMPT RECONSTRUCTION
# =============================================================================
def build_prompt(patient_query: str, trial_text: str) -> str:
    """
    Reconstructs the Phase 1 LLM evaluation prompt.
    To understand why a latent feature fired, we must look at the exact text 
    the LLM was processing at the time of hidden state extraction.
    """
    prompt = f"""### Role: You are an expert in biomedical AI with access to clinical trial data and the ability to assess the relevance of a given patient case description to a specific clinical trial. Your task is to evaluate whether the trial is relevant to the patient case by answering a set of predefined questions in YES or NO or NA, along with a brief justification for each answer.
---

### Instructions:
- Given a clinical trial description and a patient case description, evaluate relevance based on the 10 feature-based questions provided below.
- Respond with "YES", "NO", or "NA" (Not Available):
  - YES: The patient's details align with the trial's criteria and objectives.
  - NO: The patient's details do not match the trial's requirements.
  - NA: The information is not available or cannot be determined.
- Provide a brief justification for each response, citing relevant details from the patient's symptoms, diagnostics, prior treatment,
  age, endocrinology findings, and other factors.
- If specific details needed for a question are not mentioned, return NA for that question.
- Format the final output as a JSON object, where the key is the question number and the value is an object containing:
  - "response" → "YES", "NO", or "NA"
  - "justification" → A brief explanation
- Strictly output in JSON format only.

---

1. Age Eligibility – Does the patient's age fall within the trial's specified range?
2. Gender Eligibility – Is the trial open to the patient’s gender?
3. Condition Relevance – Do the patient's symptoms, diagnosis, or condition match the trial’s focus?
4. Diagnostic Findings Match – Do lab tests, imaging, or biomarkers align with the trial’s criteria?
5. Prior Treatment Consideration – Has the patient undergone treatments relevant to the trial’s eligibility criteria?
6. Inclusion/Exclusion Criteria – Does the patient meet specific trial conditions (e.g., comorbidities, concurrent medications)?
7. Pathophysiologic Mechanism – Does the patient’s condition suggest an underlying disease mechanism relevant to the trial?
8. Functional Status – Does the patient’s sensory, motor, or cognitive function align with trial requirements?
9. Interest in Experimental Therapy – Has the patient shown willingness for investigational treatments?
10. Treatment Target Alignment – Does the trial’s treatment directly address the patient’s condition or symptoms?

---

### NOTE:
For the first two question check based on following condition.
Question: Does the patient's age fall within the trial's specified age range?
Conditions:
If both minimum and maximum age are specified, check if the patient's age is within the range.
If only a minimum age is specified, check if the patient's age is greater than or equal to the minimum age.
If only a maximum age is specified, check if the patient's age is less than or equal to the maximum age.
If no age restrictions are specified, assume the trial is open to all ages (Answer YES).
If the patient's age doesn't meet any of the above conditions, answer NO.

2. Gender Eligibility:
Question: Is the trial open to participants of the patient's gender?
Conditions:
If gender is not specified in the trial, assume the trial is open to all genders (Answer YES).
If gender is specified (e.g., male, female, or both), check if the patient's gender matches the trial's eligibility.
If the trial specifies a gender restriction (e.g., only male or only female) and the patient doesn't meet that restriction, answer NO.
If gender is not relevant or not mentioned, answer YES.

---

### Query: {patient_query}
---

### Clinical Trial: {trial_text}
---

### Output Format:
Strictly in JSON format only.
Generate a JSON object where each question number is a key, containing a dictionary with:
"response" → "YES" or "NO" or "NA"
"justification" → A brief explanation for the answer
---

### Expected Output Format:
{{
  "1": {{
    "response": "YES/NO/NA",
    "justification": "<A brief explanation for the response based on patient case description and trial details>"
  }},
  ...
  "10": {{
    "response": "YES/NO/NA",
    "justification": "<A brief explanation for the response based on patient case description and trial details>"
  }}
}}

---
### Output:"""
    return prompt.strip()


# =============================================================================
# DATA LOADERS
# =============================================================================
def load_all_z(path):
    """Loads the N x M sparse activation tensor. Maps to CPU RAM for processing."""
    z = torch.load(path, map_location="cpu")
    print(f"Loaded all_z: {tuple(z.shape)}")
    return z


def load_sample_ids(path):
    """Loads the sequential mapping of test-set pairs."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded sample IDs: {len(data)}")
    return data


def load_queries(path):
    """Parses the patient case TSV to retrieve query text."""
    df = pd.read_csv(path, sep="\t", header=None, names=["topic_id", "text"])
    df["topic_id"] = df["topic_id"].astype(str)
    query_map = dict(zip(df["topic_id"], df["text"]))
    print(f"Loaded queries: {len(query_map)}")
    return query_map


def load_corpus(path):
    """Loads the massive trial JSONL to retrieve trial text."""
    corpus = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading corpus"):
            rec = json.loads(line)
            corpus[rec["id"]] = rec.get("contents", "")
    print(f"Loaded corpus docs: {len(corpus)}")
    return corpus


def load_feature_summary(path):
    """Loads the global discriminability stats to determine WHICH features to analyze."""
    with open(path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    return summary


# =============================================================================
# FEATURE SELECTION LOGIC
# =============================================================================
def select_features(summary, n=50):
    """
    Extracts the highest scoring features from the discriminability summary.
    Prioritizes both highly positive (Inclusion) and highly negative (Exclusion) features.
    """
    selected = []

    for key in [
        "top_positive_feature_indices",
        "top_negative_feature_indices",
    ]:
        if key in summary:
            selected.extend(summary[key])

    # Deduplicate while preserving rank order
    seen = set()
    deduped = []
    for x in selected:
        if x not in seen:
            deduped.append(int(x))
            seen.add(int(x))

    return deduped[:n]


# =============================================================================
# HEURISTIC PRUNING (APPROXIMATE MAX-ACT)
# =============================================================================
def split_sentences(text):
    """Simple regex to split large text blocks into sentence arrays."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def sentence_score(sentence, keywords):
    """Counts how many clinical keywords appear in a given sentence."""
    s_lower = sentence.lower()
    score = 0
    for kw in keywords:
        if kw in s_lower:
            score += 1
    return score


def approximate_prune(prompt, max_sentences=5):
    """
    Executes the "Approximate Pruning MaxAct procedure" described in the methodology.
    
    Instead of calculating exact gradient attributions for every token (computationally 
    extreme for 4096 tokens), this applies a fast heuristic:
    It scores sentences based on the density of clinically meaningful keywords 
    and retains only the top `max_sentences` to form a concise "evidence span".
    """
    keywords = [
        "age", "year-old", "male", "female", "gender",
        "diagnosis", "condition", "eligible", "inclusion",
        "exclusion", "criteria", "treatment", "therapy",
        "received", "prior", "ecog", "performance",
        "biomarker", "mutation", "trial", "patient"
    ]

    sentences = split_sentences(prompt)
    scored = []

    # Score every sentence in the prompt
    for sent in sentences:
        scored.append((sentence_score(sent, keywords), sent))

    # Sort descending by keyword density
    scored.sort(key=lambda x: x[0], reverse=True)

    # Keep the top-K sentences that have at least one keyword match
    kept = [s for score, s in scored[:max_sentences] if score > 0]

    # Fallback: if no keywords matched, just grab the first K sentences 
    # (likely structural prompt instructions)
    if not kept:
        kept = sentences[:max_sentences]

    # Combine back into a readable evidence snippet
    return "\n".join(kept)


# =============================================================================
# MAIN EXECUTION LOOP
# =============================================================================
def main():
    print("Loading data...")
    all_z = load_all_z(ALL_Z_FILE)
    sample_ids = load_sample_ids(SAMPLE_IDS_FILE)
    queries = load_queries(TOPICS_TSV)
    corpus = load_corpus(CORPUS_JSONL)
    feature_summary = load_feature_summary(FEATURE_SUMMARY_FILE)

    # 1. Identify which latent dimensions we actually want to interpret
    feature_ids = select_features(
        feature_summary,
        n=NUM_FEATURES_TO_ANALYZE
    )

    print(f"Selected {len(feature_ids)} features for analysis.")

    # Save the manifest of selected features
    with open(os.path.join(OUTPUT_DIR, "selected_features.json"), "w") as f:
        json.dump(feature_ids, f, indent=2)

    # LOOP EXPLANATION: Iterate through the selected discriminative features
    for feature_id in tqdm(feature_ids, desc="Analyzing features"):
        # Isolate the column representing this feature across all patient-trial pairs
        # activations Shape: (Total_Samples,)
        activations = all_z[:, feature_id]

        # 2. Identify the top 20 patient-trial pairs where this feature fired the strongest
        top_vals, top_indices = torch.topk(
            activations,
            k=min(TOP_EXAMPLES_PER_FEATURE, len(activations))
        )

        examples = []

        # LOOP EXPLANATION (INNER): Process each of the top 20 contexts
        for rank, (idx, act) in enumerate(
            zip(top_indices.tolist(), top_vals.tolist()),
            start=1,
        ):
            # Ignore contexts where the feature didn't fire at all
            if act <= 0:
                continue

            sample = sample_ids[idx]

            topic_id = str(sample["topic_id"])
            trial_id = sample["trial_id"]
            label = sample.get("label")

            query_text = queries.get(topic_id, "")
            trial_text = corpus.get(trial_id, "")

            # Reconstruct and prune to create the evidence span
            prompt = build_prompt(query_text, trial_text)
            pruned_text = approximate_prune(
                prompt,
                max_sentences=TOP_SENTENCES_PER_EXAMPLE
            )

            # Package the clinical snippet
            examples.append(
                {
                    "rank": rank,
                    "activation": float(act),
                    "topic_id": topic_id,
                    "trial_id": trial_id,
                    "label": label,
                    "query_preview": query_text[:500], # Keep a short preview for context
                    "pruned_text": pruned_text,        # The actual evidence span
                }
            )

        # 3. Create a dedicated JSON file for this feature
        feature_output = {
            "feature_id": feature_id,
            "num_examples": len(examples),
            "candidate_concept": "MANUAL_INTERPRETATION_REQUIRED", # Placeholder for human annotation
            "top_examples": examples,
        }

        out_path = os.path.join(
            OUTPUT_DIR,
            f"feature_{feature_id}.json"
        )

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(feature_output, f, indent=2, ensure_ascii=False)

    # 4. Generate a consolidated template file for human researchers to fill out
    interpretation_template = {}
    for fid in feature_ids:
        interpretation_template[str(fid)] = ""

    with open(
        os.path.join(OUTPUT_DIR, "feature_interpretations_template.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(interpretation_template, f, indent=2)

    print("\nDone.")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("Inspect each feature_XXXXX.json file and fill in")
    print("feature_interpretations_template.json")


if __name__ == "__main__":
    main()