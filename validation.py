import json
import logging
from datetime import datetime
from datasets import load_dataset
from tqdm import tqdm

from inference import Inferencer

inferencer = Inferencer()
dataset = load_dataset("trs4630/xquad-triplet")

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

def check_answer_exist(context: str, query: str):
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

logger.info(f"=== Starting Validation at {datetime.now()} ===")
for data in tqdm(dataset["train"].select(range(65))):
    for lang, neg_context in data["negative"].items():
        logger.info("-" * 80)
        logger.info(f"LANG: {lang} | QUERY: {data["query"]}")
        
        ans_check = check_answer_exist(neg_context, data["query"])
        logger.info(f"\n[ANSWER CHECK]\n{ans_check}")
        
        pos_context = data["positive"][lang]
        loss_check = check_information_loss(pos_context, neg_context)
        logger.info(f"\n[INFO LOSS CHECK]\n{loss_check}")

        logger.info("-" * 80 + "\n")

logger.info(f"=== Validation Completed. Log saved to {LOG_FILE} ===")
