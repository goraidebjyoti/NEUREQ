"""
NEUREQ Evaluation & Statistical Significance Testing
This script automates the evaluation of the NEUREQ ranking outputs against 
several state-of-the-art baselines using the standard `trec_eval` toolkit. 
It subsequently performs a paired Student's t-test to determine if the 
performance improvements of NEUREQ are statistically significant.
"""

import subprocess
import numpy as np
from scipy.stats import ttest_rel
import os

# ============================================================
# CONFIGURATION & CONSTANTS
# ============================================================

YEAR = 2021 # Toggle between 2021 and 2022 TREC datasets

# Retrieval settings to evaluate (e.g., standard BM25 vs. Pseudo-Relevance Feedback)
TEST_TYPE = [
    f"WholeQ_RM3_RETRIEVAL_T{YEAR}",
    f"WholeQ_RETRIEVAL_T{YEAR}",
]

# Directory names for the baseline models and the proposed NEUREQ model
MODEL_DIRS = [
    "FIRST_STAGE",
    "ZERO_SHOT",
    "SIMPLE_BERT",
    "CT_MLM_BERT",
    "MFT",
    "SCT_BERT",
    "MONOBERT_baseline",
    "SSMONOBERT_baseline",
    "NEUREQ" # The proposed framework is placed last for comparison logic
]

# Alphabetic identifiers used for superscript significance notation
MODEL_NAMES = ["a", "b", "c", "d", "e", "f", "g", "h", "i"]

# Human-readable names for the final generated tables
DISPLAY_NAMES = {
    "FIRST_STAGE": "FIRST_STAGE",
    "ZERO_SHOT": "Zero Shot",
    "SIMPLE_BERT": "Simple Bert",
    "CT_MLM_BERT": "CT Bert",
    "MFT": "MFT_Monot5",
    "SCT_BERT": "SCT Bert",
    "MONOBERT_baseline": "Monobert",
    "SSMONOBERT_baseline": "SS Monobert",
    "NEUREQ": "NEUREQ"
}

# File paths
QRELS_FILE = f"data/{YEAR}/ct_{YEAR}_qrels_mapped.txt" # Ground truth relevance judgments
RUNS_DIR = f"runs5/{YEAR}" # Directory containing the .txt run files for all models
OUTPUT_DIR = "t_test_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Metrics to extract from trec_eval
METRICS = {
    "map": "MAP",
    "P.10": "P@10",
    "recall.10": "Recall@10",
    "ndcg_cut.10": "NDCG@10",
    "P.20": "P@20",
    "recall.20": "Recall@20",
    "ndcg_cut.20": "NDCG@20"
}
ORDER = list(METRICS.keys())

# ============================================================
# EVALUATION FUNCTIONS
# ============================================================

def run_trec_eval(run_file, metric):
    """
    Executes the command-line `trec_eval` tool via a Python subprocess.
    
    Args:
        run_file (str): Path to the standard TREC-formatted run file.
        metric (str): The specific metric to calculate (e.g., 'ndcg_cut.10').
        
    Returns:
        dict: A mapping of individual query IDs to their computed metric score.
    """
    # -q flag outputs scores per query, which is required for the paired t-test
    cmd = ["trec_eval", "-q", "-m", metric, QRELS_FILE, run_file]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    scores = {}
    for line in result.stdout.strip().split("\n"):
        parts = line.strip().split()
        # Parse output; ignore the 'all' aggregate row as we need query-level data
        if len(parts) == 3 and parts[1] != "all":
            scores[parts[1]] = float(parts[2])
    return scores

def align_scores(scores1, scores2):
    """
    Aligns the query scores of two models to ensure the paired t-test 
    compares identical queries.
    """
    a, b = [], []
    for qid in scores1:
        if qid in scores2:
            a.append(scores1[qid])
            b.append(scores2[qid])
    return a, b

def superscript(indices):
    """
    Maps baseline indices to superscript letters for formal table formatting.
    Example: If NEUREQ beats models 0 ('a') and 1 ('b'), it returns 'ᵃᵇ'.
    """
    mapping = {
        0: 'ᵃ', 1: 'ᵇ', 2: 'ᶜ', 3: 'ᵈ', 4: 'ᵉ',
        5: 'ᶠ', 6: 'ᵍ', 7: 'ʰ', 8: 'ⁱ'
    }
    return ''.join([mapping.get(i, '?') for i in sorted(indices)])

def evaluate_test_type(test_type):
    """
    Main evaluation pipeline for a specific retrieval setting. Generates 
    the metric scores, performs significance testing, and writes the table.
    """
    num_models = len(MODEL_DIRS)
    all_model_scores = {metric: [] for metric in ORDER}

    # 1. Gather all query-level scores for all models and metrics
    for model_dir in MODEL_DIRS:
        # Handle specific naming convention edge-cases
        run_file = os.path.join(
            RUNS_DIR,
            model_dir,
            f"{test_type}_NN.txt" if model_dir == "QA_NN_NEW" else f"{test_type}.txt"
        )
        
        for metric in ORDER:
            scores = run_trec_eval(run_file, metric)
            all_model_scores[metric].append(scores)

    # 2. Compute mean scores and execute paired t-tests
    table_rows = []
    for i in range(num_models):
        row = [MODEL_NAMES[i], DISPLAY_NAMES[MODEL_DIRS[i]]]
        
        for metric in ORDER:
            this_scores = all_model_scores[metric][i]
            this_mean = np.mean(list(this_scores.values()))
            sup = ""

            # Statistical Significance Testing (Applied strictly to NEUREQ vs Baselines)
            if i == num_models - 1: # Assuming NEUREQ is always the last model in the list
                significant = []
                for j in range(num_models - 1):
                    baseline_scores = all_model_scores[metric][j]
                    a, b = align_scores(baseline_scores, this_scores)
                    
                    if len(a) > 0:
                        # Perform the paired Student's t-test
                        t_stat, p_val = ttest_rel(a, b)
                        
                        # Threshold for significance is strictly p <= 0.05
                        # Also checks if NEUREQ's mean actually improved over the baseline
                        if p_val <= 0.05 and np.mean(b) > np.mean(a):
                            significant.append(j)
                            
                sup = superscript(significant)

            # Format the score to 5 decimal places and append any significance superscripts
            row.append(f"{this_mean:.5f}{sup}")
        table_rows.append(row)

    # 3. Format and export the result table
    col_width = 17
    headers = ["#", "Model"] + [METRICS[m] for m in ORDER]
    header_line = "".join(f"{h:<{col_width}}" for h in headers)
    separator_line = "-" * len(header_line)
    
    table = [header_line, separator_line]
    for row in table_rows:
        table.append("".join(f"{str(col):<{col_width}}" for col in row))

    output_path = os.path.join(OUTPUT_DIR, f"{test_type}.txt")
    with open(output_path, "w") as f:
        f.write(f"# Evaluation Results: {test_type}\n\n")
        f.write("\n".join(table))
        f.write("\n")

# ============================================================
# MAIN EXECUTION
# ============================================================

for test in TEST_TYPE:
    evaluate_test_type(test)

print(f"✅ All formatted result tables saved in '{OUTPUT_DIR}'")