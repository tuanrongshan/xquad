import json
import logging
import os
import re
import concurrent.futures
from datetime import datetime
from datasets import load_dataset, Dataset
from tqdm import tqdm
from huggingface_hub import login
from dotenv import load_dotenv

from inference import Inferencer

load_dotenv()
login(token=os.getenv("HF_TOKEN"))

inferencer = Inferencer()
dataset = load_dataset("trs4630/xquad-triplet-raw")

MAX_THREADS = 100

LOG_FILE = f"validation_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ],
    force=True,
)
logger = logging.getLogger(__name__)

def check_answer_existence(context: str, query: str):
    prompt = f"""
    ### TASK: Answer Existence Audit
    Check if the specific answer to the "User Query" can still be found or inferred from the provided "Context".
    
    ### DATA:
    - User Query: {query}
    - Context: {context}
    
    ### INSTRUCTIONS:
    1. Status: State "EXISTS" if the answer is present, or "CLEAN" if it is successfully removed.
    2. Proof: 
       - If the context is NOT in English: Provide the relevant sentence in the [Original Language] AND its [English Translation].
       - If the context is in English: Provide the [Original Sentence].
    3. Reasoning: Briefly explain why the answer is or isn't there.
    
    Output Format:
    Status: [EXISTS/CLEAN]
    Proof: [Sentence/Translation or "N/A"]
    Reasoning: [Brief Explanation]
    """
    result = inferencer.model_inference(prompt, temperature=0.2)
    return result

def check_information_loss(positive: str, negative: str):
    prompt = f"""
    ### TASK: Information Loss Analysis (Diff)
    Compare the "Positive Document" (Original) with the "Negative Document" (Modified) to identify any lost information OTHER than the intended answer removal.
    
    ### DATA:
    - Positive: {positive}
    - Negative: {negative}
    
    ### INSTRUCTIONS:
    1. Analysis: List any significant facts, dates, or entities present in Positive but missing in Negative (excluding the answer itself).
    2. Proof: For each lost fact, provide:
       - [Original Language Segment] -> [English Translation]
    3. Integrity Score: Rate from 1-10 (10 = only the answer was removed, 1 = too much content was deleted).
    
    Output Format:
    Lost Facts: [List or "None"]
    Proofs: [Original -> English]
    Integrity Score: [1-10]
    """
    result = inferencer.model_inference(prompt, temperature=0.2)
    return result

def check_language_consistency(context: str, lang: str):
    prompt = f"""
    ### TASK: Language Consistency Audit
    Check if the "Entire Context" is strictly written in the designated "Target Language".
    
    ### DATA:
    - Target Language: {lang}
    - Entire Context: {context}
    
    ### INSTRUCTIONS:
    1. Status: State "PASS" if the text is entirely in the Target Language. State "FAIL" if any sentence, phrase, or word (excluding universal proper nouns/names) is left in another language (e.g., mistakenly kept in English).
    2. Violations: If FAIL, quote the specific segments that are in the wrong language. If PASS, output "None".
    3. Reasoning: Briefly explain the decision.
    
    Output Format:
    Status: [PASS/FAIL]
    Violations: [Quotes or "None"]
    Reasoning: [Brief Explanation]
    """
    result = inferencer.model_inference(prompt, temperature=0.2)
    return result

def process_row(data):
    is_valid = True
    log_msgs = []
    
    for lang, neg_context in data["negative"].items():
        log_msgs.append("-" * 80)
        log_msgs.append(f"LANG: {lang} | QUERY: {data['query']}")
        
        ans_check = check_answer_existence(neg_context, data["query"])
        log_msgs.append(f"\n[ANSWER CHECK]\n{ans_check}")
        
        pos_context = data["positive"][lang]
        loss_check = check_information_loss(pos_context, neg_context)
        log_msgs.append(f"\n[INFO LOSS CHECK]\n{loss_check}")

        lang_check = check_language_consistency(neg_context, lang)
        log_msgs.append(f"\n[LANGUAGE CHECK]\n{lang_check}")

        # Parse LLM responses to determine validity
        ans_exists = bool(re.search(r"Status:\s*EXISTS", ans_check, re.IGNORECASE))
        has_lost_facts = not bool(re.search(r"Lost Facts:\s*\[?[\"']?None[\"']?\]?", loss_check, re.IGNORECASE))
        lang_fails = bool(re.search(r"Status:\s*FAIL", lang_check, re.IGNORECASE))
        
        if ans_exists or has_lost_facts or lang_fails:
            log_msgs.append(f"\n=> VALIDATION FAILED on {lang}. Dropping entire row.\n")
            is_valid = False
            break  # Stop checking other languages to save API calls

    if is_valid:
        log_msgs.append("\n=> VALIDATION PASSED. Keeping row.\n")
        
    return is_valid, data, "\n".join(log_msgs)

logger.info(f"=== Starting Validation & Filtering at {datetime.now()} ===")

filtered_queries = []
filtered_positives = []
filtered_negatives = []

with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
    futures = [executor.submit(process_row, data) for data in dataset["train"]]
    for future in tqdm(concurrent.futures.as_completed(futures), total=len(dataset["train"])):
        is_valid, data, log_output = future.result()
        logger.info(log_output)
        if is_valid:
            filtered_queries.append(data["query"])
            filtered_positives.append(data["positive"])
            filtered_negatives.append(data["negative"])

logger.info(f"=== Validation Completed. Log saved to {LOG_FILE} ===")
logger.info(f"Filtered Dataset Size: {len(filtered_queries)} / {len(dataset['train'])}")

filtered_dataset = Dataset.from_dict({
    "query": filtered_queries,
    "positive": filtered_positives,
    "negative": filtered_negatives,
})

logger.info("Pushing filtered dataset to hub: trs4630/xquad-triplet")
filtered_dataset.push_to_hub("trs4630/xquad-triplet")
