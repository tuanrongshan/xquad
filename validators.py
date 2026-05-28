import re

from inference import Inferencer

inferencer = Inferencer()


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
    return inferencer.model_inference(prompt, temperature=0.2)


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
    return inferencer.model_inference(prompt, temperature=0.2)


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
    return inferencer.model_inference(prompt, temperature=0.2)


def parse_validation(ans_check: str, loss_check: str, lang_check: str):
    """Returns (is_valid, feedback) where feedback holds the verbatim judge
    outputs for each failed dimension (None for dimensions that passed)."""
    ans_exists = bool(re.search(r"Status:\s*EXISTS", ans_check, re.IGNORECASE))
    has_lost_facts = not bool(re.search(r"Lost Facts:\s*\[?[\"']?None[\"']?\]?", loss_check, re.IGNORECASE))
    lang_fails = bool(re.search(r"Status:\s*FAIL", lang_check, re.IGNORECASE))
    is_valid = not (ans_exists or has_lost_facts or lang_fails)
    return is_valid, {
        "answer_leak": ans_check if ans_exists else None,
        "info_loss": loss_check if has_lost_facts else None,
        "lang_issue": lang_check if lang_fails else None,
    }
