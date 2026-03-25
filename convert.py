import json
import os
import concurrent.futures
from typing import List, Dict, Any
from datasets import Dataset
from tqdm import tqdm
from collections import defaultdict
from inference import Inferencer
from huggingface_hub import login
from dotenv import load_dotenv

load_dotenv()

# Login to huggingface
login(token=os.getenv("HF_TOKEN"))

MAX_THREADS = 50
MAX_ARTICLE = 1e8
MAX_PARAGRAPH = 1e8
MAX_QA = 1e8

lang_code_dict = {
    "arabic": "ar",
    "german": "de",
    "greek": "el",
    "english": "en",
    "spanish": "es",
    "hindi": "hi",
    "romanian": "ro",
    "russian": "ru",
    "thai": "th",
    "turkish": "tr",
    "vietnamese": "vi",
    "chinese": "zh",
}

inferencer = Inferencer()

# --- Core Logic Functions ---

def trim_answer(context: str, answer: str, lang: str) -> str:
    """
    Removes answer-related information.
    """
    prompt = f"""
    TASK: Remove all traces of the specific "Target Answer" from the provided "Source Context".
    
    GUIDELINES:
    1. Identify and delete sentences, phrases, or specific keywords that directly or indirectly reveal the Target Answer.
    2. LANGUAGE CONSTRAINT: You MUST output the text in {lang}. Do NOT translate the content to English.
    3. Maintain the grammatical flow and continuity of the remaining text.
    4. Do NOT invent new information to replace what was deleted.
    5. Output ONLY the modified text.

    [Target Answer]: {answer}
    [Source Context]: {context}
    
    Modified Text ({lang} only):
    """
    return inferencer.model_inference(prompt, temperature=0.3).strip()

def rephrase_context(context: str, lang: str) -> str:
    """
    Pephrases the context to increase lexical diversity while preserving the remaining non-answer facts.
    """
    prompt = f"""
    TASK: Rewrite the following text to enhance its linguistic variety while maintaining the original meaning of the remaining information.
    
    GUIDELINES:
    1. Use different vocabulary, synonyms, and sentence structures (Paraphrasing).
    2. LANGUAGE CONSTRAINT: You MUST output the text in {lang}. Do NOT translate the content to English.
    3. Ensure the tone remains formal and informative, similar to a reference document.
    4. Do NOT add any external information or facts not present in the source.
    5. The output must be logically consistent and professional.
    6. Output ONLY the rewritten text.

    [Source Text]: {context}
    
    Rewritten Text ({lang} only):
    """
    return inferencer.model_inference(prompt, temperature=0.8).strip()

def process_task(task_info: Dict[str, Any]):
    """
    Processes a single language context for a specific question.
    """
    lang = task_info['lang']
    context = task_info['context']
    answers = task_info['answers']
    q_id = task_info['q_id']
    query_text = task_info['query_text']
    
    # 1. Trim
    tmp_context = context
    for answer in answers:
        tmp_context = trim_answer(tmp_context, answer["text"], lang.capitalize())
    
    # 2. Rephrase
    negative_context = rephrase_context(tmp_context, lang.capitalize())
    
    return {
        "q_id": q_id,
        "query": query_text,
        "lang": lang,
        "positive": context,
        "negative": negative_context
    }

# --- Data Loading ---
xquad_data = {}
for lang, code in lang_code_dict.items():
    with open(f"xquad.{code}.json", "r", encoding="utf-8") as f:
        xquad_data[lang] = json.load(f)

# --- Task Generation ---
all_tasks = []
for article_id in range(min(len(xquad_data["english"]["data"]), MAX_ARTICLE)):
    for paragraph_id in range(min(len(xquad_data["english"]["data"][article_id]["paragraphs"]), MAX_PARAGRAPH)):
        for qas_id in range(min(len(xquad_data["english"]["data"][article_id]["paragraphs"][paragraph_id]["qas"]), MAX_QA)):
            
            # Unique ID to group them later
            q_id = f"art{article_id}_par{paragraph_id}_qa{qas_id}"
            query_text = xquad_data["english"]["data"][article_id]["paragraphs"][paragraph_id]["qas"][qas_id]["question"]
            
            for lang in xquad_data.keys():
                paragraph_data = xquad_data[lang]["data"][article_id]["paragraphs"][paragraph_id]
                all_tasks.append({
                    "q_id": q_id,
                    "query_text": query_text,
                    "lang": lang,
                    "context": paragraph_data["context"],
                    "answers": paragraph_data["qas"][qas_id]["answers"]
                })

# --- Parallel Execution ---
print(f"Executing {len(all_tasks)} tasks across {MAX_THREADS} threads...")
results = []
with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
    futures = [executor.submit(process_task, task) for task in all_tasks]
    for future in tqdm(concurrent.futures.as_completed(futures), total=len(all_tasks)):
        results.append(future.result())

# --- Re-grouping Results ---
grouped_data = defaultdict(lambda: {"query": "", "positive": {}, "negative": {}})

for res in results:
    qid = res["q_id"]
    grouped_data[qid]["query"] = res["query"]
    grouped_data[qid]["positive"][res["lang"]] = res["positive"]
    grouped_data[qid]["negative"][res["lang"]] = res["negative"]

# --- Formatting for Dataset ---
final_queries = []
final_positives = []
final_negatives = []

for qid in grouped_data:
    final_queries.append(grouped_data[qid]["query"])
    final_positives.append(grouped_data[qid]["positive"])
    final_negatives.append(grouped_data[qid]["negative"])

dataset = Dataset.from_dict({
    "query": final_queries,
    "positive": final_positives,
    "negative": final_negatives,
})

dataset.push_to_hub("trs4630/xquad-triplet")