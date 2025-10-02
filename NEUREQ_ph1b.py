#Sanitising the raw LLM responses

import json

# ======= Global Variables =======
INPUT_JSONL = "data/2021/WholeQ_RM3_RETRIEVAL_T2021_llm_responses.jsonl" #Put the raw llm response generated LLM jsonl file
OUTPUT_JSONL = f"{INPUT_JSONL.split('.')[0]}_sanitized.jsonl"
# ================================

def clean_generated_text(text, idx):
    """Extract JSON format from generated text."""
    try:
        text = text.split('</think>')[-1]
        text = '{' + '{'.join(text.split('{')[1:])
        text = '}'.join(text.split('}')[:11]) + '}'
        parsed = json.loads(text)

        # Check if parsed has exactly 10 keys
        if isinstance(parsed, dict) and len(parsed) == 10:
            return parsed
        else:
            print(f"===== Skipping idx: {idx} due to insufficient keys in cleaned_output")
            return None
    except Exception as e:
        print(f"{'='*15} Skipping idx: {idx} due to error: {e}")
        return None

def process_jsonl(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', encoding='utf-8') as outfile:
        for idx, line in enumerate(infile):
            try:
                data = json.loads(line)
                cleaned = clean_generated_text(data.get("result"), idx)
                out_data = {
                        "qid": data.get("qid"),
                        "docid": data.get("docid"),
                        "cleaned_output": cleaned
                }
                outfile.write(json.dumps(out_data) + '\n')
            except Exception as e:
                print(f"Error processing line: {e}")

if _name_ == "_main_":
    process_jsonl(INPUT_JSONL, OUTPUT_JSONL)
