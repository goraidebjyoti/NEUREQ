# mft5_3.py

"""
Baseline Model: MFT-MonoT5 (Step 3/4) - Multi-Field Template Segment Scoring

Objective:
    This script executes the neural scoring phase of the MFT-MonoT5 baseline. 
    It takes the candidate lists generated in Step 2 and re-ranks them using 
    a massive 3-Billion parameter T5 model (monot5-3b-med-msmarco).

    Because clinical trials greatly exceed T5's 512-token limit, this script 
    implements the "MaxP" sliding-window approach:
    1. Parse the trial into structural fields (Title, Condition, Eligibility, Description).
    2. Split lengthy fields into overlapping 6-sentence segments.
    3. Feed each segment into the MonoT5 model via a strict prompt template.
    4. Calculate the P(true) relevance probability for every segment independently.
"""

import os
import re
import json
import nltk
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Required for sentence tokenization in the sliding window
nltk.download("punkt")

# =============================================================================
# CONFIGURATION & FILE PATHS
# =============================================================================
# --- Inputs ---
# The raw clinical trial corpus required to extract the text fields
corpus_file = "data/clinicaltrials/corpus.jsonl"  
# The original, official patient case descriptions (We evaluate against the original query)
queries_file = "data/2022/ct_2022_queries.tsv"                
# The candidate list retrieved in Step 2 (e.g., the fused RRF results)
# You should point this to one of the 4 output files from Step 2.
ranked_list_file = "data/2022/bm25rm3_rrf_nqs_pd.txt"               

# --- Outputs ---
# An intermediate cache storing the exact score for every individual sentence segment
output_cache = "data/2022/sigir/segment_scores_WholeQ_RM3_T2022.jsonl"

os.makedirs(os.path.dirname(output_cache), exist_ok=True)

# --- Hyperparameters ---
# The medically fine-tuned 3 Billion parameter MonoT5 model
model_name = "castorini/monot5-3b-med-msmarco"  
# Reduced batch size because 3B parameters consume massive VRAM
batch_size = 16                                 
# Sliding Window Parameters: 6 sentences per window, advancing by 3 sentences each step (overlap)
nlength, nstride = 6, 3                         

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def sliding_segments(text, nlength=6, nstride=3):
    """
    Splits a lengthy text block into overlapping windows of `nlength` sentences.
    This ensures no critical clinical context is lost precisely at the boundary 
    of a rigid token cut-off.
    """
    if not text or not isinstance(text, str):
        return []
    
    # Tokenize the raw text into distinct sentences
    sents = nltk.sent_tokenize(text)
    segments, i = [], 0
    
    # LOOP EXPLANATION: Slide the window across the sentence array
    while i < len(sents):
        # Join 'nlength' sentences together into a single segment string
        seg = " ".join(sents[i:i+nlength])
        if seg.strip():
            segments.append(seg)
        # Stop if the current window reaches the end of the document
        if i + nlength >= len(sents):
            break
        # Advance the window by 'nstride' sentences (creating an overlap of nlength - nstride)
        i += nstride
    return segments

def parse_fields(contents):
    """
    Uses Regular Expressions to extract specific structural fields from the 
    raw, flat ClinicalTrials.gov text dump. MonoT5 performs better when 
    fed structured metadata (like Title and Condition) alongside the main text.
    """
    def extract(pattern, text, stop_keywords=None):
        # Find the starting anchor (e.g., "Eligibility Criteria:")
        match = re.search(pattern, text, re.IGNORECASE)
        if not match:
            return ""
        
        # Extract everything after the anchor
        start = match.end()
        rest = text[start:]
        
        # Truncate the extraction if it hits the anchor of the NEXT section
        if stop_keywords:
            stops = [rest.find(k) for k in stop_keywords if rest.find(k) != -1]
            if stops:
                rest = rest[:min(stops)]
        return rest.strip()

    # Extract the Trial Title
    title = extract(r"Study Title:\s*", contents,
                    ["Official Title:", "Brief Summary:", "Condition:", "Eligibility", "Status:"])
    # Extract the Target Medical Condition
    condition = extract(r"Condition:\s*", contents,
                        ["Interventions:", "Eligibility", "Status:", "Phase:", "Facility:"])
    # Extract the formal Eligibility Criteria
    eligibility = extract(r"(Eligibility Criteria:|Inclusion Criteria:|Inclusion:|Eligibility:)\s*",
                          contents,
                          ["Status:", "Phase:", "Condition:", "Facility:", "Brief Summary:", "Detailed Description:"])
    # Extract the general summary/description
    description = extract(r"Brief Summary:\s*", contents,
                          ["Eligibility", "Status:", "Condition:", "Phase:", "Facility:"])
    if not description:
        description = extract(r"Detailed Description:\s*", contents,
                              ["Eligibility", "Status:", "Condition:", "Phase:", "Facility:"])
    
    return title, condition, eligibility, description

def build_template(query_text, title, condition, segment, field="eligibility"):
    """
    Formats the extracted text into the exact template string the MonoT5 model 
    was fine-tuned on. Deviating from this string structure destroys performance.
    """
    if field == "eligibility":
        return f"Query: {query_text} Document: title: {title}\ncondition: {condition}\neligibility: {segment}\nRelevant:"
    elif field == "description":
        return f"Query: {query_text} Document: title: {title}\ncondition: {condition}\ndescription: {segment}\nRelevant:"
    else:
        raise ValueError("field must be eligibility or description")

# =============================================================================
# MODEL & DATA LOADING
# =============================================================================
print(f"Loading {model_name} ...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# AutoModelForSeq2SeqLM is required for T5. 
# device_map="auto" allows the massive 3B model to automatically shard its weights 
# across multiple GPUs if one GPU does not have enough VRAM.
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"    
)
model.eval()

# Load the official patient queries
df_queries = pd.read_csv(queries_file, sep="\t", header=None, names=["topic_id", "query_text"])

# Load the candidate trials retrieved in Step 2.
# We map topic_id -> set of unique trial_ids to avoid duplicate processing.
candidates = {}  
with open(ranked_list_file, "r") as f:
    for line in f:
        topic_id, _, trial_id, rank, score, run_id = line.strip().split()
        if int(rank) > 100:  # We only re-rank the top 100 candidates
            continue
        candidates.setdefault(topic_id, set()).add(trial_id)

print(f"Loaded candidates from {ranked_list_file}: {sum(len(v) for v in candidates.values())} (topic,trial) pairs.")

print("Loading corpus into memory...")
corpus = {}
with open(corpus_file, "r") as f:
    for line in tqdm(f, desc="Reading corpus"):
        trial = json.loads(line)
        corpus[trial["id"]] = trial["contents"]
print(f"Loaded {len(corpus)} trials into memory.")

# =============================================================================
# CORE SCORING FUNCTION (MonoT5 Logic)
# =============================================================================
def score_batch(model, tokenizer, texts, max_len=512):
    """
    Executes the batched forward pass through MonoT5 to calculate P(true).
    """
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_len
    ).to(model.device)

    # torch.cuda.amp.autocast() enables Automatic Mixed Precision, speeding up 
    # FP16 tensor operations on modern GPUs.
    with torch.no_grad(), torch.cuda.amp.autocast():
        # MonoT5 frames relevance ranking as a sequence-to-sequence task.
        # It is trained to output the literal text string "true" if relevant, and "false" if not.
        id_true = tokenizer("true", add_special_tokens=False).input_ids[0]
        id_false = tokenizer("false", add_special_tokens=False).input_ids[0]

        # We force the model's decoder to evaluate the probabilities of these specific tokens
        labels_true = torch.full((len(texts), 1), id_true, dtype=torch.long).to(model.device)
        labels_false = torch.full((len(texts), 1), id_false, dtype=torch.long).to(model.device)

        # Forward pass to get the logits (raw, unnormalized predictions)
        out_true = model(**inputs, labels=labels_true)
        out_false = model(**inputs, labels=labels_false)

        # Extract the specific logit values for the "true" and "false" tokens
        logit_true = out_true.logits[:, 0, id_true]
        logit_false = out_false.logits[:, 0, id_false]

        # Calculate the Softmax probability of "true": exp(true) / (exp(true) + exp(false))
        # This provides a normalized continuous relevance score between 0.0 and 1.0.
        probs = torch.exp(logit_true) / (torch.exp(logit_true) + torch.exp(logit_false))
        
        return probs.cpu().tolist()

# =============================================================================
# MAIN PROCESSING LOOP
# =============================================================================
# Open the JSONL cache file to store the segment scores incrementally
with open(output_cache, "w") as fout:
    
    # LOOP 1: Iterate through every patient query
    for qrow in tqdm(df_queries.itertuples(), total=len(df_queries), desc="Processing queries"):
        topic_id, query_text = str(qrow.topic_id), qrow.query_text

        if topic_id not in candidates:
            continue
        topic_candidates = candidates[topic_id]

        # LOOP 2: Iterate through the top 100 candidate trials for this specific query
        for trial_id in tqdm(topic_candidates, desc=f"Topic {topic_id}", leave=False):
            if trial_id not in corpus:
                continue
            contents = corpus[trial_id]

            # Parse the trial into structural components
            title, condition, eligibility, description = parse_fields(contents)

            # LOOP 3: Process the "eligibility" and "description" fields independently
            for field, text in [("eligibility", eligibility), ("description", description)]:
                
                # Split the text into overlapping 6-sentence chunks
                segments = sliding_segments(text, nlength, nstride)
                if not segments:
                    continue

                # Format every segment into the strict MonoT5 text template
                templates = [build_template(query_text, title, condition, seg, field) for seg in segments]

                # LOOP 4: Batch the segments to maximize GPU throughput
                for i in range(0, len(templates), batch_size):
                    batch_templates = templates[i:i+batch_size]
                    
                    # Calculate P(true) for the batch
                    scores = score_batch(model, tokenizer, batch_templates)
                    
                    # Write the score for every individual segment to disk
                    for seg, score in zip(batch_templates, scores):
                        fout.write(json.dumps({
                            "topic_id": topic_id,
                            "trial_id": trial_id,
                            "field": field,
                            "segment": seg,
                            "score": score
                        }) + "\n")

print(f"✅ Done! Saved segment scores to {output_cache}")