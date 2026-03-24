import json
from typing import List, Dict, Any
from datasets import Dataset
from tqdm import tqdm
from inference import Inferencer

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

xquad_data = {}
for lang, code in lang_code_dict.items():
    with open(f"xquad.{code}.json", "r", encoding="utf-8") as f:
        xquad_data[lang] = json.load(f)

query = []
positive = []
negative = []
id = []

inferencer = Inferencer()

try:
    print(inferencer.model_inference("hello!"))
except:
    print("ERROR: ???")

def fake_context(context: str, answers: List[Dict[str, Any]], lang: str) -> str:
    for answer in answers:
        context = trim_answer(context, answer["text"], lang)
    return rephrase_context(context, lang)

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

MAX_ARTICLE = 10
MAX_PARAGRAPH = 3
MAX_QA = 3

for article_id in tqdm(range(min(len(xquad_data["english"]["data"]), MAX_ARTICLE))):
    for paragraph_id in tqdm(range(min(len(xquad_data["english"]["data"][article_id]["paragraphs"]), MAX_PARAGRAPH))):
        for qas_id in tqdm(range(min(len(xquad_data["english"]["data"][article_id]["paragraphs"][paragraph_id]["qas"]), MAX_QA))):
            query.append(xquad_data["english"]["data"][article_id]["paragraphs"][paragraph_id]["qas"][qas_id]["question"])
            positive_doc = {}
            negative_doc = {}
            id = xquad_data["english"]["data"][article_id]["paragraphs"][paragraph_id]["qas"][qas_id]["id"]
            for lang, data in tqdm(xquad_data.items()):
                article = data["data"][article_id]
                positive_doc[lang] = article["paragraphs"][paragraph_id]["context"]
                negative_doc[lang] = fake_context(
                    article["paragraphs"][paragraph_id]["context"],
                    article["paragraphs"][paragraph_id]["qas"][qas_id]["answers"],
                    lang.capitalize()
                )
                assert id == article["paragraphs"][paragraph_id]["qas"][qas_id]["id"]
            positive.append(positive_doc)
            negative.append(negative_doc)

data = {
    "query": query,
    "positive": positive,
    "negative": negative,
}

dataset = Dataset.from_dict(data)
dataset.push_to_hub("trs4630/xquad-triplet")