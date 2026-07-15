# sctbert_1.py

"""
Baseline Model: SCT-BERT (Step 1/4) - Positive Pair Extraction

Objective:
    This script prepares the foundational dataset for the SCT-BERT baseline. 
    To train a model to recognize clinical relevance, we leverage the internal 
    structure of clinical trials: we assume that a trial's own 'Brief Summary' 
    is the gold-standard 'patient query' for that trial's 'Inclusion Criteria'.
    
    This script performs regex-based parsing to extract these two fields from 
    the raw text dump, creating positive training pairs (label=1).
"""

import json
import re
from tqdm import tqdm

# =============================================================================
# CONFIGURATION
# =============================================================================
# Input: The raw ClinicalTrials.gov corpus (JSONL format)
input_file = "data/clinicaltrials/json_corpus/corpus.jsonl"
# Output: A clean JSONL file where every line is a (Summary, Criteria) match
output_file = "data/clinicaltrials/positive_pairs.jsonl"

def extract_field(text, field_names):
    """
    Parses specific sections (e.g., 'Inclusion Criteria') from unstructured trial text.
    
    Logic:
    1. Regex Search: Identifies the header (e.g., 'Eligibility Criteria:').
    2. Extraction: Captures text starting immediately after the header.
    3. Boundary Detection: Automatically stops when it encounters the next 
       standard section header (e.g., '\nStatus:').
    """
    # Create a Regex pattern to find the header case-insensitively
    pattern = r"(" + "|".join([re.escape(f) for f in field_names]) + r")\s*:\s*"
    match = re.search(pattern, text, re.IGNORECASE)
    if not match:
        return None
    
    # Start extraction right after the header match
    start = match.end()
    rest = text[start:]
    
    # Boundary Detection: Regex to find the next section header (a line starting with letters)
    stop_match = re.search(r"\n[A-Za-z ]{2,30}:\s", rest)
    end = stop_match.start() if stop_match else len(rest)
    block = rest[:end].strip()

    # 🚫 Cleanup: If the extraction inadvertently grabbed a stray header line, remove it.
    lines = block.splitlines()
    if lines and any(h.lower() in lines[0].lower() for h in field_names):
        lines = lines[1:]
    return "\n".join(lines).strip()

# Initialize stats for dataset auditing
total, no_summary, no_criteria, written = 0, 0, 0, 0

# =============================================================================
# PROCESSING LOOP
# =============================================================================
print("Beginning corpus extraction...")
with open(input_file, "r") as fin, open(output_file, "w") as fout:
    # LOOP EXPLANATION: Iterate through the entire 448k+ trial corpus
    for line in tqdm(fin, desc="Processing trials"):
        trial = json.loads(line)
        total += 1
        trial_id = trial["id"]
        contents = trial["contents"]

        # 1. Attempt to extract the Brief Summary
        summary = extract_field(contents, ["Brief Summary", "Summary"])
        if not summary:
            no_summary += 1
            continue # If a trial lacks a summary, it cannot be used for training
        
        # 2. Attempt to extract Inclusion/Eligibility Criteria
        criteria = extract_field(contents, ["Eligibility Criteria", "Inclusion Criteria", "Inclusion", "Eligibility"])
        if not criteria:
            no_criteria += 1
            continue # If a trial lacks criteria, it cannot be used for training

        # 3. Write the positive training pair to the output file
        # 'label': 1 signifies a "True" relevance match (Self-Supervised)
        record = {
            "trial_id_summary": trial_id,
            "trial_id_criteria": trial_id,
            "summary": summary,
            "criteria": criteria,
            "label": 1
        }
        fout.write(json.dumps(record) + "\n")
        written += 1

# =============================================================================
# REPORTING
# =============================================================================
print(f"\n✅ Processing complete:")
print(f"Total trials analyzed: {total}")
print(f"Skipped (No summary): {no_summary}")
print(f"Skipped (No criteria): {no_criteria}")
print(f"Positive pairs written: {written}")