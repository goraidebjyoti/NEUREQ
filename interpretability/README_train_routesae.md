# Interpretability Layer: RouteSAE Training & Feature Analysis (`train_routesae.py`)

## Overview
`train_routesae.py` implements the mechanistic interpretability core of the NEUREQ framework. While Phase 1 of the pipeline utilizes `DeepSeek-R1-Distill-Qwen-32B` to generate natural language evaluations, large language models reason in dense, entangled hidden states that are fundamentally uninterpretable to humans. 

This script applies a **Route Sparse Autoencoder (RouteSAE)** to map the LLM's opaque internal representations (extracted during clinical trial evaluation) into a high-dimensional, sparse dictionary. By mapping states to sparse activations, we isolate discrete "concepts" the LLM uses to reason about clinical eligibility. We then analyze which concepts activate for eligible trials vs. ineligible hard-negative trials.

## Mathematical Architecture: RouteSAE

Training standard SAEs on modern LLMs requires learning a separate autoencoder for every single transformer layer, which is computationally infeasible. We implement the unified architecture proposed by Shi et al. (2025):

1. **Dynamic Router:** Instead of extracting from one layer, we extract states from the LLM's middle semantic layers (25% to 75% depth). The Router uses mean pooling and a learned projection to output a probability distribution across layers, selecting the single layer most salient for the current patient-trial input.
2. **TopK Activation:** The selected dense vector is projected into a wider feature space ($8\times$ the hidden size). Standard SAEs use $L1$ regularization to force sparsity, but $L1$ shrinks activation magnitudes. We use a `TopK` activation function (Gao et al., 2024), which strictly zeroes out all but the $K=64$ largest activations, preserving true magnitude.
3. **Reconstruction:** The sparse vector $z$ is projected back to the original dimension. The network is trained purely on Mean Squared Error (MSE) loss between the reconstructed vector and the original routed vector.

## Pipeline Execution Flow

1. **Prompt Reconstruction:** The script reads the synthetic triplet training dataset (`train_1196.jsonl`). Crucially, it rebuilds the *exact* prompt used during Phase 1 generation.
2. **State Extraction (Cached):** A forward pass is executed on the 32B model. The hidden state vectors for the *last input token* (the semantic bottleneck before generation) across the middle transformer layers are extracted and cached to disk to prevent redundant inference.
3. **Training:** The RouteSAE is trained over 50 epochs using a 3-phase learning rate schedule (Linear Warmup $\rightarrow$ Stable Peak $\rightarrow$ Linear Decay) and intermittent decoder column normalization to ensure convergence.
4. **Discriminability Analysis:** The trained SAE runs inference over the dataset. For every learned feature, the script calculates a discriminability score:
   $$D_i = \mu_{pos}(z_i) - \mu_{neg}(z_i)$$
   Features with high positive scores represent clinical concepts driving "Eligibility" reasoning, while high negative scores represent concepts driving "Exclusion" reasoning.

## File Dependencies

### Inputs
* `triplet_syn_dataset_1196.jsonl`: The label source (1 = Positive, 0 = Hard Negative).
* `synthetic_gold_queries.tsv` & `corpus.jsonl`: To reconstruct the input prompt.

### Outputs
Saved to `routesae_outputs/train_1196/`:
* `hidden_states/all_hidden_states.pt`: Cached 32-bit float LLM tensors.
* `models/routesae_final.pt`: The trained model weights.
* `models/feature_analysis.json`: The final discriminability rankings detailing which of the thousands of sparse features correspond to clinical inclusion vs. exclusion.

## Usage
```bash
python train_routesae.py