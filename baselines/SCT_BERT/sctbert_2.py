# sctbert_2.py

"""
Baseline Model: SCT-BERT (Step 2/4) - Negative Pair Generation

Objective:
    This script finalizes the self-supervised training dataset for SCT-BERT. 
    While Step 1 extracted the "ground truth" positive matches (Trial A Summary + Trial A Criteria), 
    a neural network requires contrastive examples to learn effectively. 
    
    For every positive pair, this script generates two negative pairs (label=0) by 
    randomly sampling the criteria from entirely different clinical trials. This creates 
    a balanced dataset (1 positive : 2 negatives) that teaches the model how to 
    penalize semantic mismatches.
"""

import json
import random
from tqdm import tqdm

# =============================================================================
# REPRODUCIBILITY
# =============================================================================
# Setting a fixed seed ensures that the exact same random negative pairings 
# are generated every time the script is run, allowing for consistent debugging 
# and verifiable academic results.
random.seed(42)

# =============================================================================
# CONFIGURATION
# =============================================================================
# Input: The positive pairs generated in Step 1
pos_file = "data/clinicaltrials/positive_pairs.jsonl"
# Output: The final, combined training dataset ingested by HuggingFace in Step 3
output_file = "data/clinicaltrials/train_pairs.jsonl"

# =============================================================================
# DATA PREPARATION (In-Memory Lookup)
# =============================================================================
print("Loading criteria into memory...")
# We load all extracted criteria into a dictionary keyed by trial ID.
# This allows for O(1) random sampling of negative criteria during the generation loop.
criteria_dict = {}
with open(pos_file, "r") as fin:
    for line in fin:
        rec = json.loads(line)
        criteria_dict[rec["trial_id_criteria"]] = rec["criteria"]

# Extract a flat list of all available trial IDs to sample from
all_trial_ids = list(criteria_dict.keys())
print(f"Loaded {len(all_trial_ids)} unique criteria.")

# =============================================================================
# NEGATIVE SAMPLING & DATASET COMPILATION
# =============================================================================
with open(pos_file, "r") as fin, open(output_file, "w") as fout:
    # LOOP EXPLANATION: Iterate through the "ground truth" positive pairs
    for line in tqdm(fin, desc="Generating train pairs"):
        pos_rec = json.loads(line)
        
        # 1. Write the positive record directly to the final dataset (Label = 1)
        fout.write(json.dumps(pos_rec) + "\n")
        
        # 2. Generate exactly 2 negative records for this specific summary
        # This 1:2 ratio prevents the model from trivially predicting "mismatch" 
        # for everything while still providing enough contrastive signals.
        for _ in range(2):
            # Randomly select a trial ID from the entire corpus
            neg_id = random.choice(all_trial_ids)
            
            # COLLISION CHECK: We must ensure the randomly selected negative ID 
            # is NOT the same as the positive ID. If it is, we would accidentally 
            # label a true match as a '0', severely poisoning the training data.
            while neg_id == pos_rec["trial_id_summary"]:
                neg_id = random.choice(all_trial_ids)
            
            # Construct the negative pair: Original Summary + Random Criteria
            neg_rec = {
                "trial_id_summary": pos_rec["trial_id_summary"], # Keep original summary
                "trial_id_criteria": neg_id,                     # Inject random criteria
                "summary": pos_rec["summary"],
                "criteria": criteria_dict[neg_id],
                "label": 0                                       # Mark as a mismatch
            }
            # Write the negative record to the final dataset
            fout.write(json.dumps(neg_rec) + "\n")

print(f"Training dataset written to {output_file}")