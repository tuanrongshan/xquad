import json
import logging
import os
import concurrent.futures
from datetime import datetime
from datasets import load_dataset, Dataset
from tqdm import tqdm
from huggingface_hub import login
from dotenv import load_dotenv

from validators import (
    check_answer_existence,
    check_information_loss,
    check_language_consistency,
    parse_validation,
)

load_dotenv()
login(token=os.getenv("HF_TOKEN"))

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

        passed, _ = parse_validation(ans_check, loss_check, lang_check)
        if not passed:
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
