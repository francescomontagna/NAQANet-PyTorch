from typing import Tuple, List, Union, Dict, Any

from code.util import compute_em, compute_f1, metric_max_over_ground_truths


def convert_tokens(eval_dict, qa_id, y_start, y_end):
    """Convert predictions to tokens from the context.
    Args:
        eval_dict (dict): Dictionary with eval info for the dataset. This is
            used to perform the mapping from IDs and indices to actual text.
        qa_id (int): QA example IDs.
        y_start (int): start predictions index - word level.
        y_end (int): end predictions index - word level.
    Returns:
        pred_dict (dict): Dictionary index IDs -> predicted answer text.
    """
    
    context = eval_dict[str(qa_id)]["context"]
    spans = eval_dict[str(qa_id)]["spans"]
    start_idx = spans[y_start][0]
    end_idx = spans[y_end][1]
    pred = context[start_idx: end_idx]

    return pred


def eval_dicts(gold_dict, pred_dict):
    avna = f1 = em = total = 0
    for key, value in pred_dict.items():
        total += 1
        ground_truths = answer_json_to_strings(gold_dict[key]['answer'])[0]
        prediction = value
        em += metric_max_over_ground_truths(compute_em, prediction, ground_truths)
        f1 += metric_max_over_ground_truths(compute_f1, prediction, ground_truths)

    eval_dict = {'EM': 100. * em / total,
                 'F1': 100. * f1 / total}

    return eval_dict

def answer_json_to_strings(answer: Dict[str, Any]) -> Tuple[Tuple[str, ...], str]:
    """
    Takes an answer JSON blob from the DROP data release and converts it into strings used for
    evaluation.
    """
    if answer["number"]:
        return tuple([str(answer["number"])]), "number"
    elif answer["spans"]:
        return tuple(answer["spans"]), "span" if len(answer["spans"]) == 1 else "spans"
    else:
        return tuple(["{0} {1} {2}".format(answer["date"]["day"],
                                           answer["date"]["month"],
                                           answer["date"]["year"])]), "date"
