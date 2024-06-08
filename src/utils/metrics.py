import string
from typing import List
from collections import Counter
import regex
from rouge import Rouge

def normalize_text(s: str) -> str:
    """Normalization from the SQuAD evaluation script.

    See https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/
    """

    def remove_articles(text):
        return regex.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def best_subspan_em(prediction: str, ground_truths: List[str]) -> float:
    normalized_prediction = normalize_text(prediction)

    for ground_truth in ground_truths:
        normalized_ground_truth = normalize_text(ground_truth)
        if normalized_ground_truth.lower() in normalized_prediction.lower():
            return 1.0
    return 0.0


def exact_match(prediction: str, ground_truths: List[str]) -> float:
    normalized_prediction = normalize_text(prediction)

    for ground_truth in ground_truths:
        normalized_ground_truth = normalize_text(ground_truth)
        if normalized_prediction.lower() == normalized_ground_truth.lower():
            return 1.0
    return 0.0


def f1_score(prediction, ground_truths):
    '''F1 Score from the HotpotQA evaluation script.
    
    See https://github.com/hotpotqa/hotpot/blob/master/hotpot_evaluate_v1.py
    '''

    normalized_prediction = normalize_text(prediction)
    normalized_ground_truth = normalize_text(ground_truths[0])

    ZERO_METRIC = 0.0

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


benchmark_function_map = {
    'accuracy': best_subspan_em,
    'em': exact_match,
    'f1': f1_score,
}