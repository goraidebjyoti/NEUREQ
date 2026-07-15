# mft5_1.py

"""
Baseline Model: MFT-MonoT5 (Step 1/4) - Neural Query Synthesis (NQS)

Objective:
    This script executes the first step of the MFT-MonoT5 baseline evaluated 
    in Section 4.2 of the NEUREQ paper. Clinical patient descriptions are often 
    long, complex, and lexically misaligned with trial documents. 
    
    To bridge this lexical gap, we apply Neural Query Synthesis (NQS). We pass 
    the patient description into a T5 model (doc2query-t5-base-msmarco) to 
    generate 40 diverse, synthetic search queries that a human might use to 
    find relevant trials. These synthetic queries will be fused in Step 2.
"""

import pandas as pd
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm

# =============================================================================
# CONFIGURATION
# =============================================================================
# --- File Paths ---
# Input: The raw text of the patient case descriptions (TREC 2022)
queries_file = "data/2022/ct_2022_queries.tsv"     
# Output: The generated synthetic queries mapped back to their topic IDs
output_file = "data/2022/synthetic_queries_2022.tsv"  

# --- Model Parameters ---
# The doc2query model fine-tuned on MS MARCO for query expansion
model_name = "castorini/doc2query-t5-base-msmarco"

# Number of distinct synthetic queries to generate per patient case. 
# 40 is a standard robust expansion factor in modern Information Retrieval.
num_queries = 40                 

# Maximum token limits for the T5 sequence-to-sequence model
# 512 accommodates the lengthy clinical patient descriptions.
max_input_length = 512
# 64 limits the generated synthetic queries to standard search-engine lengths.
max_output_length = 64

# Automatically use GPU if available for faster generation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================
# INITIALIZATION
# =============================================================================
print(f"Loading {model_name} onto {device}...")
# T5Tokenizer converts raw text into sub-word token IDs required by T5
tokenizer = T5Tokenizer.from_pretrained(model_name)

# T5ForConditionalGeneration is the standard seq2seq architecture used for 
# text-to-text generation tasks (like translation, summarization, or doc2query)
model = T5ForConditionalGeneration.from_pretrained(model_name)
model.to(device)

# =============================================================================
# DATA LOADING
# =============================================================================
# Read the TSV containing the official TREC topics. 
# Expected format: <topic_id> \t <topic_text>
df = pd.read_csv(queries_file, sep="\t", header=None, names=["topic_id", "topic_text"])

# List to store the generated outputs
results = []

# =============================================================================
# NEURAL QUERY SYNTHESIS (GENERATION LOOP)
# =============================================================================
# LOOP EXPLANATION: Iterate row-by-row over the patient descriptions.
for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating synthetic queries"):
    topic_id = row["topic_id"]
    topic_text = row["topic_text"]

    # 1. Encoding: Convert the patient case into a tensor of token IDs.
    # truncation=True ensures we don't crash the model if a case exceeds 512 tokens.
    input_ids = tokenizer.encode(
        topic_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_length
    ).to(device)

    # 2. Decoding (Generation): Generate multiple independent synthetic queries.
    outputs = model.generate(
        input_ids=input_ids,
        max_length=max_output_length,
        # do_sample=True activates non-deterministic generation.
        # Instead of just taking the most likely next token (greedy decoding), 
        # it samples from the probability distribution, creating diverse queries.
        do_sample=True,
        # top_k=10 restricts the sampling pool to the 10 most likely next tokens.
        # This prevents the model from generating gibberish while maintaining diversity.
        top_k=10,                 
        # Generate exactly 40 distinct sequences in a single batched forward pass.
        num_return_sequences=num_queries
    )

    # 3. Post-Processing: Decode the generated token IDs back into human-readable text.
    for o in outputs:
        # skip_special_tokens=True removes padding and End-Of-Sequence (EOS) tokens.
        query_text = tokenizer.decode(o, skip_special_tokens=True)
        # Store the mapping: Which topic generated this specific synthetic query?
        results.append([topic_id, topic_text, query_text])

# =============================================================================
# OUTPUT SAVING
# =============================================================================
# Convert the flattened results back into a structured DataFrame
out_df = pd.DataFrame(results, columns=["topic_id", "topic_text", "synthetic_query"])

# Save as a Tab-Separated Values (TSV) file. This format is safer than CSV 
# because clinical texts frequently contain commas.
out_df.to_csv(output_file, sep="\t", index=False)

print(f"✅ Done! Saved {len(results)} synthetic queries to {output_file}")