import os
import json
import random
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import pandas as pd

# ========== CONFIGURATION ==========
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
GOLD_QUERIES = "data/clinicaltrials/synthetic_gold_queries.tsv" #List of synthetic queries/patient case descriptions
GOLD_QRELS = "data/clinicaltrials/synthetic_gold_qrels.txt" #One relevant trial for each synthetic query/patient case description
HN_FILE = "data/clinicaltrials/train_run/generated_train_v2_with_gold.bm25.k1=0.82.b=0.68.tsv"
CORPUS_FILE = "data/clinicaltrials/corpus.jsonl" #trial corpus of 448528 trials
PROMPT_FILE = "prompt2.txt"
SELECTED_QUERIES_FILE = "data/train/selected_queries.txt"
PROCESSED_TRACKER_FILE = "data/train/processed_queries.txt"
OUTPUT_FILE = "data/train/hard_negatives.jsonl"
OUTPUT_FILE_WITH_LLM = "data/train/hard_negatives_with_llm.jsonl"
SKIPPED_FILE = "data/train/skipped_responses.jsonl"

TEMPERATURE = 0.5
DO_SAMPLE = True

# ========== LOAD DEEPSEEK MODEL ==========
print("🔹 Loading model...")
bnb_config = BitsAndBytesConfig(load_in_4bit=False, load_in_8bit=False, bnb_4bit_compute_dtype=torch.float16)
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
MODEL = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    device_map="auto"
)
print("✅ Model loaded successfully in 16-bit using bitsandbytes!\n")

# ========== LOAD FILES ==========
print("🔹 Loading queries...")
queries_df = pd.read_csv(GOLD_QUERIES, sep="\t", header=None, names=["topic_id", "query"])
queries_df = queries_df[queries_df["topic_id"] >= 76]
print(f"✅ Queries loaded! ({len(queries_df)} queries)")

print("🔹 Loading BM25 ranking file...")
hn_df = pd.read_csv(HN_FILE, sep="\t", header=None, names=["topic_id", "trial_id", "rank"])
print(f"✅ BM25 file loaded! ({len(hn_df)} rows)")

print("🔹 Loading corpus...")
corpus = {}
with open(CORPUS_FILE, "r", encoding="utf-8") as f:
    for line in f:
        doc = json.loads(line)
        corpus[doc["id"]] = doc["contents"]
print(f"✅ Corpus loaded! ({len(corpus)} trials)\n")

# ========== TRACKING ==========
if os.path.exists(SELECTED_QUERIES_FILE):
    with open(SELECTED_QUERIES_FILE) as f:
        selected_ids = set(int(line.strip()) for line in f)
else:
    selected_ids = set()

if os.path.exists(PROCESSED_TRACKER_FILE):
    with open(PROCESSED_TRACKER_FILE) as f:
        processed_ids = set(int(line.strip()) for line in f)
else:
    processed_ids = set()

available = queries_df[~queries_df["topic_id"].isin(selected_ids)]
sampled = available.sample(300, random_state=random.randint(1, 99999))

with open(SELECTED_QUERIES_FILE, "a") as f:
    for tid in sampled["topic_id"]:
        f.write(f"{tid}\n")

print("🔹 Sampled 300 new queries. Starting processing...\n")

with open(PROMPT_FILE, "r", encoding="utf-8") as f:
    prompt_template = f.read()

qrel_df = pd.read_csv(GOLD_QRELS, sep="\t", header=None, names=["topic_id", "dummy", "trial_id", "rel"])
topic_to_rel = dict(zip(qrel_df["topic_id"], qrel_df["trial_id"]))

# ========== HELPER FUNCTIONS ==========
def clean_generated_text(text):
    """Extract JSON portion from the generated text."""
    try:
        text = text.split('</think>')[-1]
        text = '{' + '{'.join(text.split('{')[1:])
        text = '}'.join(text.split('}')[:11]) + '}'
        return json.loads(text)
    except Exception as e:
        print(f"{'='*10} Skipping due to error: {e}")
        return None

# ========== MAIN PROCESSING ==========
with open(OUTPUT_FILE, "a", encoding="utf-8") as out_file, \
     open(OUTPUT_FILE_WITH_LLM, "a", encoding="utf-8") as out_file_with_llm, \
     open(PROCESSED_TRACKER_FILE, "a", encoding="utf-8") as tracker, \
     open(SKIPPED_FILE, "a", encoding="utf-8") as skipped_file:

    for _, row in tqdm(sampled.iterrows(), total=len(sampled), desc="Processing queries"):
        topic_id = int(row["topic_id"])
        query_text = row["query"]

        if topic_id in processed_ids:
            continue

        print(f"\n🔷 Processing Topic ID: {topic_id}")

        rel_trial = topic_to_rel.get(topic_id)
        trials = hn_df[hn_df["topic_id"] == topic_id]

        for _, trial_row in trials.iterrows():
            trial_id = trial_row["trial_id"]
            if trial_id == rel_trial:
                continue

            print(f"  ➡ Evaluating Trial ID: {trial_id}")

            trial_text = corpus.get(trial_id, None)
            if not trial_text:
                print("    ⚠ Trial text not found in corpus.")
                continue

            prompt = prompt_template.format(query_text, trial_text)

            inputs = TOKENIZER(prompt, return_tensors="pt", truncation=True).to(MODEL.device)
            outputs = MODEL.generate(**inputs, temperature=TEMPERATURE, do_sample=DO_SAMPLE, max_new_tokens=8192)
            response = TOKENIZER.decode(outputs[0], skip_special_tokens=True)

            print("    🔷 Raw model response:", response[:500].replace("\n", " "))

            cleaned = clean_generated_text(response)
            if cleaned is None:
                skipped_entry = {
                    "topic_id": topic_id,
                    "trial_id": trial_id,
                    "reason": "JSON parsing failed",
                    "raw_response": response
                }
                skipped_file.write(json.dumps(skipped_entry) + "\n")
                skipped_file.flush()
                continue

            if all(cleaned.get(q, {}).get("response") == "NO" for q in ["3", "5", "6", "10"]):
                entry_basic = {"topic_id": topic_id, "query": query_text, "trial_id": trial_id}
                json.dump(entry_basic, out_file)
                out_file.write("\n")

                entry_with_llm = {"topic_id": topic_id, "query": query_text, "trial_id": trial_id, "llm_response": response}
                json.dump(entry_with_llm, out_file_with_llm)
                out_file_with_llm.write("\n")

                print(f"    ✅ Hard negative found and saved for Trial ID: {trial_id}")
                break
            else:
                print("    ❌ Trial not a hard negative.")

        tracker.write(f"{topic_id}\n")
        tracker.flush()

print("\n✅ All queries processed. Output written to:", OUTPUT_FILE)
print("✅ Hard negatives with LLM responses saved to:", OUTPUT_FILE_WITH_LLM)
print("⚠ Skipped/error cases written to:", SKIPPED_FILE)
