import subprocess
import numpy as np
from scipy.stats import ttest_rel
import os

# Configuration

YEAR = 2021 #or 2022

TEST_TYPE = [
    f"WholeQ_RM3_RETRIEVAL_T{YEAR}",
    f"WholeQ_RETRIEVAL_T{YEAR}",
]

MODEL_DIRS = [
    "FIRST_STAGE",
    "ZERO_SHOT",
    "SIMPLE_BERT",
    "CT_MLM_BERT",
    "MFT",
    "SCT_BERT",
    "MONOBERT_baseline",
    "SSMONOBERT_baseline",
    "NEUREQ"
]

MODEL_NAMES = ["a", "b", "c", "d", "e", "f", "g", "h", "i"]

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

QRELS_FILE = f"data/{YEAR}/ct_{YEAR}_qrels_mapped.txt"
RUNS_DIR = f"runs5/{YEAR}"
OUTPUT_DIR = "t_test_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Metric Config (removed MRR)
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

# Run trec_eval for a file+metric
def run_trec_eval(run_file, metric):
    cmd = ["trec_eval", "-q", "-m", metric, QRELS_FILE, run_file]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    scores = {}
    for line in result.stdout.strip().split("\n"):
        parts = line.strip().split()
        if len(parts) == 3 and parts[1] != "all":
            scores[parts[1]] = float(parts[2])
    return scores

# Align queries across models
def align_scores(scores1, scores2):
    a, b = [], []
    for qid in scores1:
        if qid in scores2:
            a.append(scores1[qid])
            b.append(scores2[qid])
    return a, b

# Superscript helpers
def superscript(indices):
    mapping = {
        0: 'ᵃ', 1: 'ᵇ', 2: 'ᶜ', 3: 'ᵈ', 4: 'ᵉ',
        5: 'ᶠ', 6: 'ᵍ', 7: 'ʰ', 8: 'ⁱ'
    }
    return ''.join([mapping.get(i, '?') for i in sorted(indices)])

# Evaluate for one test type
def evaluate_test_type(test_type):
    num_models = len(MODEL_DIRS)
    all_model_scores = {metric: [] for metric in ORDER}

    # 1. Gather all scores
    for model_dir in MODEL_DIRS:
        run_file = os.path.join(
            RUNS_DIR,
            model_dir,
            f"{test_type}_NN.txt" if model_dir == "QA_NN_NEW" else f"{test_type}.txt"
        )
        for metric in ORDER:
            scores = run_trec_eval(run_file, metric)
            all_model_scores[metric].append(scores)

    # 2. Compute means + t-test (superscripts only on NEUREQ)
    table_rows = []
    for i in range(num_models):
        row = [MODEL_NAMES[i], DISPLAY_NAMES[MODEL_DIRS[i]]]
        for metric in ORDER:
            this_scores = all_model_scores[metric][i]
            this_mean = np.mean(list(this_scores.values()))
            sup = ""

            # If NEUREQ (last model), compare against all baselines
            if i == num_models - 1:
                significant = []
                for j in range(num_models - 1):
                    baseline_scores = all_model_scores[metric][j]
                    a, b = align_scores(baseline_scores, this_scores)
                    if len(a) > 0:
                        t_stat, p_val = ttest_rel(a, b)
                        if p_val <= 0.05 and np.mean(b) > np.mean(a):
                            significant.append(j)
                sup = superscript(significant)

            row.append(f"{this_mean:.5f}{sup}")
        table_rows.append(row)

    # 3. Format result table
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

# Run for all test types
for test in TEST_TYPE:
    evaluate_test_type(test)

print(f"✅ All formatted result tables saved in '{OUTPUT_DIR}'")
