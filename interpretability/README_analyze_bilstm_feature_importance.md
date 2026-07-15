# Interpretability Layer: Attention × Gradient Analysis

## Overview
`analyze_bilstm_feature_importance.py` serves as the primary feature-level interpretability mechanism for the NEUREQ Phase 2 re-ranker. While standard neural networks are often criticized as "black boxes" in clinical settings, this script exposes exactly *which* of the 10 clinical eligibility questions most significantly impact the model's relevance predictions.

It calculates and aggregates feature importance across all evaluated patient-trial pairs, generating normalized CSV rankings and bar plots to visually demonstrate the model's clinical reasoning hierarchy.

## The Mathematical Framework: Attention × Gradient
Relying solely on attention weights can be misleading. An attention head might assign a high weight to a feature, but if that feature is zeroed out or mathematically isolated in subsequent layers, it won't actually affect the prediction. 

To achieve true mechanistic interpretability, we combine **Attention** (where the model looks) with **Gradient Attribution** (how much that location changes the output). 

For a given patient-trial prediction $y$, the importance $I_i$ of the $i$-th eligibility question is calculated as:

$$I_{i} = \alpha_{i} \times \sum_{j=1}^{d} \left| \frac{\partial y}{\partial x_{i,j}} \right|$$

Where:
* $\alpha_i$ is the scalar attention weight assigned by the Additive Attention layer to question $i$.
* $x_{i,j}$ represents the concatenated feature vector for question $i$ (combining the Question ID, the YES/NO/NA answer, and the 768-dim BERT justification embedding), where $d = 779$.
* $\frac{\partial y}{\partial x_{i,j}}$ is the partial derivative (gradient) of the predicted logit with respect to that specific feature dimension, extracted via PyTorch autograd.

## Technical Implementation Details
1. **Autograd Manipulation:** By default, PyTorch discards gradients for intermediate tensors to save memory. To execute the formula above, the script modifies the BiLSTM's `forward` pass to call `x.retain_grad()` on the concatenated feature tensor.
2. **Backward Routing:** A `.backward()` pass is executed for every single evaluation sample. The script temporarily forces the model into `.train()` mode during this step, which is a hard requirement for cuDNN to correctly route gradients backwards through the recurrent LSTM time steps.
3. **Aggregation:** The script sums the absolute gradients across all 779 feature dimensions, multiplies the result by the attention weight, averages these scores across the entire dataset, and normalizes the final distribution to sum to 1.0.

## File Dependencies

### Inputs
1. **Model Checkpoint:** `models_new/BiLSTM_ADD_ATTN_1196/model_epoch_12.pt`
2. **Evaluated Test Sets:** The sanitized JSONL outputs from Phase 1b (e.g., `WholeQ_RETRIEVAL_T2021_llm_responses_sanitized.jsonl`).

### Outputs
*All outputs are saved to `models_new/BiLSTM_ADD_ATTN_1196/feature_importance/`*
1. **Dataset-Specific Metrics:** `.csv` and `.png` files detailing the feature importance distribution for each specific retrieval setting (e.g., 2021 WholeQ, 2022 RM3).
2. **Global Aggregation (`overall_importance.csv` / `.png`):** A consolidated ranking that averages the metrics across all four test sets, representing the model's generalized feature prioritization.

## Usage
```bash
python analyze_bilstm_feature_importance.py