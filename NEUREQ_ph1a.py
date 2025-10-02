import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Global file variables
PROMPTS_FILE = "prompt.txt"
QUERIES_FILE = "data/2021/ct_2021_queries.tsv" #Put the correct queries file 
CORPUS_FILE = "data/clinicaltrials/corpus.jsonl" #Trial Corpus of 448528 trials
RUN_FILE = "data/2021/WholeQ_RM3_RETREIVAL_T2021.txt" #Select the first stage retrieval txt file
TRACK_FILE = "data/track.json"
RESULTS_FILE = f"{RUN_FILE.split('.')[0]}_llm_responses.jsonl"

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
TEMPERATURE = 0.5
DO_SAMPLE = True

START_FROM_ZERO = False  # <<-- Set this to True to start fresh
TRACKING_KEY = "LLM_ANSWER_GENERATION"

# Configure bitsandbytes for 16-bit (FP16)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=False,  # Ensure it's not in 4-bit
    load_in_8bit=False,  # Ensure it's not in 8-bit
    bnb_4bit_compute_dtype=torch.float16,  # Set computation dtype to FP16
)

# Load tokenizer
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load model with 16-bit precision
MODEL = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    torch_dtype=torch.float16,  # Ensure model is in FP16
    device_map="auto"
)

print("Model loaded successfully in 16-bit using bitsandbytes!")


def load_prompt_template(file_path=PROMPTS_FILE):
    """Load the prompt template from the given file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().strip()
    
def reset_track_file(track_file=TRACK_FILE):
    """Reset the track.json file to start from the beginning."""
    with open(track_file, "w", encoding="utf-8") as f:
        json.dump({TRACKING_KEY: 0}, f)

def read_track_file(track_file=TRACK_FILE):
    """Read the track.json file to get the last completed index."""
    try:
        with open(track_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get(TRACKING_KEY, 0)
    except FileNotFoundError:
        return 0

def update_track_file(index, track_file=TRACK_FILE):
    """Update the track.json file with the latest completed index."""
    with open(track_file, "w", encoding="utf-8") as f:
        json.dump({TRACKING_KEY: index}, f)

def generate_response(query, doc, max_new_tokens=4096):
    """Generate a response from the LLM given the query and document."""
    prompt = TEMPLATE.replace("{0}", query).replace("{1}", doc)
    input_ids = TOKENIZER(prompt, return_tensors="pt").input_ids.to(MODEL.device)
    with torch.no_grad():
        gen_tokens = MODEL.generate(input_ids, 
                                    max_new_tokens=max_new_tokens, 
                                    temperature=TEMPERATURE, 
                                    do_sample=DO_SAMPLE)
    gen_text = TOKENIZER.batch_decode(gen_tokens, skip_special_tokens=True)[0]
    return gen_text

def clean_generated_text(text):
    """Extract JSON format from generated text."""
    try:
        text = text.split('</think>')[-1]
        text = '{' + '{'.join(text.split('{')[1:])
        text = '}'.join(text.split('}')[:11]) + '}'
        return json.loads(text)
    except Exception as e:
        print(f"{'='*15}Skipping due to error: {e}")
        return None

def load_data(queries_file=QUERIES_FILE, corpus_file=CORPUS_FILE, run_file=RUN_FILE):
    """Load queries, corpus, and run file to create q-d pairs."""
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
            qid, docid = int(parts[0]), str(parts[2])
            if qid in queries and docid in corpus:
                qd_pairs.append({"qid": qid, "query": queries[qid], "docid": docid, "doc": corpus[docid]})
    
    print(f"Loaded {len(qd_pairs)} query-document pairs.")
    return qd_pairs

def process_qd_pairs(qd_pairs, output_file=RESULTS_FILE, track_file=TRACK_FILE):
    """Process multiple query-document pairs and store results in a JSON file."""
    if START_FROM_ZERO:
        reset_track_file(track_file)
    start_index = read_track_file(track_file)

    total_pairs = len(qd_pairs)
    print(f"Resuming from pair index: {start_index} of {total_pairs}")

    with tqdm(total=total_pairs, initial=start_index, desc="Processing", unit="pair") as pbar:
        for i in range(start_index, total_pairs):
            print(f"Processing pair {i + 1} of {total_pairs}...")
            qid, query = qd_pairs[i]["qid"], qd_pairs[i]["query"]
            docid, doc = qd_pairs[i]["docid"], qd_pairs[i]["doc"]
            result = generate_response(query, doc)
            
            cleaned_output = clean_generated_text(result)
            
            entry = {"qid": qid, "docid": docid, "result": result, "cleaned_output": cleaned_output}
            
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")  
            
            update_track_file(i + 1, track_file)
            pbar.update(1)

if _name_ == "_main_":
    TEMPLATE = load_prompt_template()
    qd_pairs = load_data()
    process_qd_pairs(qd_pairs)
    print("Processing completed!")
