# train_pairs_sctbert.py

import json
import random
from tqdm import tqdm

# Set seed for reproducibility
random.seed(42)

pos_file = "data/clinicaltrials/positive_pairs.jsonl"
output_file = "data/clinicaltrials/train_pairs.jsonl"

# Load all criteria into memory
print("Loading criteria into memory...")
criteria_dict = {}
with open(pos_file, "r") as fin:
    for line in fin:
        rec = json.loads(line)
        criteria_dict[rec["trial_id_criteria"]] = rec["criteria"]

all_trial_ids = list(criteria_dict.keys())
print(f"Loaded {len(all_trial_ids)} unique criteria.")

with open(pos_file, "r") as fin, open(output_file, "w") as fout:
    for line in tqdm(fin, desc="Generating train pairs"):
        pos_rec = json.loads(line)
        
        # Write the positive record directly
        fout.write(json.dumps(pos_rec) + "\n")
        
        # Sample 2 negatives
        for _ in range(2):
            neg_id = random.choice(all_trial_ids)
            while neg_id == pos_rec["trial_id_summary"]:
                neg_id = random.choice(all_trial_ids)
            
            neg_rec = {
                "trial_id_summary": pos_rec["trial_id_summary"],
                "trial_id_criteria": neg_id,
                "summary": pos_rec["summary"],
                "criteria": criteria_dict[neg_id],
                "label": 0
            }
            fout.write(json.dumps(neg_rec) + "\n")

print(f"Training dataset written to {output_file}")