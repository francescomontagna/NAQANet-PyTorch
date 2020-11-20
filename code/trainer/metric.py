"""
Some functions are takend from the official evaluation script for v1.1 of the SQuAD dataset.
"""

import re
import string
import sys
from collections import Counter


def normalize_answer(s):
    """
    Lower text and remove punctuation, articles and extra whitespace.
    :param s: original string
    :return: normalized string
    """
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    """
    Calculate F1 score given prediction and true answer strings.
    :param prediction: prediction string
    :param ground_truth: answer string
    :return: F1 score
    """
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    """
    Calculate exact match score given prediction and true answer strings.
    :param prediction: prediction string
    :param ground_truth: answer string
    :return: EM score
    """
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    """
    Calculate the maximum metric value when we have multiple ground truths.
    i.e., for each question, we have multiple answers.
    :param metric_fn: the function to calculate metric
    :param prediction: our model predicted answer string
    :param ground_truths: the list of answer strings
    :return: the maximum metric value by comparing our prediction
             to each ground_truth
    """
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def evaluate(eval_dict, predictions):
    """
    Evaluate performance, calculate metrics EM and F1.
    :param dataset: the dictionary of 'data' in json file.
    :param predictions: the dictionary of our predictions.
                        (k, v) is like (qa['id'], prediction string)
    """
    f1 = exact_match = total = 0

    keys = eval_dict.keys()
    for q_id, pred in predictions.items():
        total += 1
        if q_id not in keys:
            message = 'Unanswered question ' + q_id + \
                      ' will receive score 0.'
            print(message, file=sys.stderr)
            continue
        ground_truths = eval_dict[q_id]
        exact_match += metric_max_over_ground_truths(
            exact_match_score, pred, ground_truths)
        f1 += metric_max_over_ground_truths(
            f1_score, pred, ground_truths)

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    return {'exact_match': exact_match, 'f1': f1}


if __name__ == "__main__":
    pass