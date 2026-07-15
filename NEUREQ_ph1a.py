import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm
import os

# Confine execution to a specific GPU to manage resources
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# ==========================================
# GLOBAL CONFIGURATION & FILE PATHS
# ==========================================
PROMPTS_FILE = "prompt.txt" # Contains the base instruction template for the LLM
QUERIES_FILE = "data/2021/ct_2021_queries.tsv" # TREC 2021 queries (Patient descriptions)
CORPUS_FILE = "data/clinicaltrials/corpus.jsonl" # Complete Clinical Trials corpus (448,528 trials)
RUN_FILE = "data/2021/WholeQ_RM3_RETREIVAL_T2021.txt" # First-stage retrieval results (top-k trials per query)
TRACK_FILE = "data/track.json" # State file to track progress and allow resuming
RESULTS_FILE = f"{RUN_FILE.split('.')[0]}_llm_responses.jsonl" # Output file for LLM generations

# ==========================================
# MODEL CONFIGURATION
# ==========================================
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
TEMPERATURE = 0.5
DO_SAMPLE = True # Enables probabilistic sampling for generation diversity

START_FROM_ZERO = False  # Set to True to overwrite track file and start from the beginning
TRACKING_KEY = "LLM_ANSWER_GENERATION"

# Configure bitsandbytes for 16-bit (FP16) precision
# This avoids the overhead of 4-bit/8-bit quantization while managing the 32B model's memory footprint
bnb_config = BitsAndBytesConfig(
    load_in_4bit=False,  
    load_in_8bit=False,  
    bnb_4bit_compute_dtype=torch.float16,  
)

# Load the tokenizer associated with the DeepSeek model
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load the causal language model with explicit FP16 casting
MODEL = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    torch_dtype=torch.float16,  
    device_map="auto" # Automatically maps layers to available GPU memory
)

print("Model loaded successfully in 16-bit using bitsandbytes!")


def load_prompt_template(file_path=PROMPTS_FILE):
    """
    Reads and returns the prompt template from the specified text file.
    This template dictates the LLM's role and the required JSON output structure.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().strip()
    
def reset_track_file(track_file=TRACK_FILE):
    """
    Initializes or resets the tracking JSON file to index 0.
    Used when a completely fresh run is required (START_FROM_ZERO = True).
    """
    with open(track_file, "w", encoding="utf-8") as f:
        json.dump({TRACKING_KEY: 0}, f)

def read_track_file(track_file=TRACK_FILE):
    """
    Reads the tracking file to determine the last successfully processed query-document pair.
    Allows the script to resume gracefully after an interruption.
    """
    try:
        with open(track_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get(TRACKING_KEY, 0)
    except FileNotFoundError:
        return 0

def update_track_file(index, track_file=TRACK_FILE):
    """
    Writes the current processing index to the tracking file.
    Called iteratively after each successful LLM generation.
    """
    with open(track_file, "w", encoding="utf-8") as f:
        json.dump({TRACKING_KEY: index}, f)

def generate_response(query, doc, max_new_tokens=4096):
    """
    Constructs the final prompt and queries the LLM for a response.
    
    Args:
        query (str): The patient case description.
        doc (str): The clinical trial text.
        max_new_tokens (int): Maximum generation length (vital for verbose reasoning models).
        
    Returns:
        str: The raw generated text from the LLM.
    """
    # Inject patient (query) and trial (doc) data into the template placeholders
    prompt = TEMPLATE.replace("{0}", query).replace("{1}", doc)
    
    # Tokenize input and move to GPU
    input_ids = TOKENIZER(prompt, return_tensors="pt").input_ids.to(MODEL.device)
    
    # Generate response without computing gradients to save memory
    with torch.no_grad():
        gen_tokens = MODEL.generate(input_ids, 
                                    max_new_tokens=max_new_tokens, 
                                    temperature=TEMPERATURE, 
                                    do_sample=DO_SAMPLE)
        
    # Decode the generated tokens back to a standard string
    gen_text = TOKENIZER.batch_decode(gen_tokens, skip_special_tokens=True)[0]
    return gen_text

def clean_generated_text(text):
    """
    Sanitizes and extracts the structured JSON payload from the raw LLM output.
    This specifically handles edge-cases introduced by reasoning models like DeepSeek-R1.
    
    Args:
        text (str): The raw LLM generation.
        
    Returns:
        dict: The parsed JSON dictionary, or None if parsing fails.
    """
    try:
        # Strip internal chain-of-thought tokens (DeepSeek typically encapsulates reasoning in <think> tags)
        text = text.split('</think>')[-1]
        
        # Isolate the main JSON object block starting with the first brace
        text = '{' + '{'.join(text.split('{')[1:])
        
        # Restrict the string to exactly 10 closing braces. 
        # Note: This is a strict heuristic designed specifically for the 10-question NEUREQ schema.
        text = '}'.join(text.split('}')[:11]) + '}'
        
        return json.loads(text)
    except Exception as e:
        # Log parsing errors for auditing edge-cases later
        print(f"{'='*15}Skipping due to error: {e}")
        return None

def load_data(queries_file=QUERIES_FILE, corpus_file=CORPUS_FILE, run_file=RUN_FILE):
    """
    Aggregates data from queries, the trial corpus, and the retrieval run file 
    to create actionable Patient-Trial (Query-Document) pairs.
    """
    print("Reading queries (TSV)...")
    queries = {}
    with open(queries_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            qid, query = int(parts[0]), parts[1]
            queries[qid] = query
    print(f"Loaded {len(queries)} queries.")

    print("Reading corpus...")
    corpus = {}
    with open(corpus_file, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            corpus[str(item["id"])] = item["contents"]

    qd_pairs = []
    print("Reading run file...")
    with open(run_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            # Standard TREC run format: qid Q0 docid rank score run_name
            qid, docid = int(parts[0]), str(parts[2])
            
            # Map the IDs to their actual textual content
            if qid in queries and docid in corpus:
                qd_pairs.append({"qid": qid, "query": queries[qid], "docid": docid, "doc": corpus[docid]})
    
    print(f"Loaded {len(qd_pairs)} query-document pairs.")
    return qd_pairs

def process_qd_pairs(qd_pairs, output_file=RESULTS_FILE, track_file=TRACK_FILE):
    """
    Main execution loop. Processes each query-document pair through the LLM, 
    sanitizes the output, and writes the results incrementally to disk.
    """
    if START_FROM_ZERO:
        reset_track_file(track_file)
    start_index = read_track_file(track_file)

    total_pairs = len(qd_pairs)
    print(f"Resuming from pair index: {start_index} of {total_pairs}")

    # Use tqdm for a clear visual progress bar during long evaluation pipelines
    with tqdm(total=total_pairs, initial=start_index, desc="Processing", unit="pair") as pbar:
        for i in range(start_index, total_pairs):
            qid, query = qd_pairs[i]["qid"], qd_pairs[i]["query"]
            docid, doc = qd_pairs[i]["docid"], qd_pairs[i]["doc"]
            
            # Request inference from LLM
            result = generate_response(query, doc)
            
            # Sanitize raw string to dictionary
            cleaned_output = clean_generated_text(result)
            
            # Construct the final schema entry
            entry = {"qid": qid, "docid": docid, "result": result, "cleaned_output": cleaned_output}
            
            # Append immediately to prevent data loss on crash
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")  
            
            # Update state file
            update_track_file(i + 1, track_file)
            pbar.update(1)

if __name__ == "__main__":
    # Ensure variables from the global scope are initialized
    TEMPLATE = load_prompt_template()
    qd_pairs = load_data()
    process_qd_pairs(qd_pairs)
    print("Processing completed!")