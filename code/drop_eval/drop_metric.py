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
        print(f"Ground_truth: {ground_truths}")
        prediction = value
        print("Predicted_value: " + prediction)
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


# class DropEmAndF1(Metric):
#     """
#     This :class:`Metric` takes the best span string computed by a model, along with the answer
#     strings labeled in the data, and computes exact match and F1 score using the official DROP
#     evaluator (which has special handling for numbers and for questions with multiple answer spans,
#     among other things).
#     """
#     def __init__(self) -> None:
#         self._total_em = 0.0
#         self._total_f1 = 0.0
#         self._count = 0

#     def __call__(self, prediction: Union[str, List], ground_truths: List):  # type: ignore
#         """
#         Parameters
#         ----------
#         prediction: ``Union[str, List]``
#             The predicted answer from the model evaluated. This could be a string, or a list of string
#             when multiple spans are predicted as answer.
#         ground_truths: ``List``
#             All the ground truth answer annotations.
#         """
#         # If you wanted to split this out by answer type, you could look at [1] here and group by
#         # that, instead of only keeping [0].
#         ground_truth_answer_strings = [answer_json_to_strings(annotation)[0] for annotation in ground_truths]
#         exact_match, f1_score = metric_max_over_ground_truths(
#                 drop_em_and_f1,
#                 prediction,
#                 ground_truth_answer_strings
#         )
#         self._total_em += exact_match
#         self._total_f1 += f1_score
#         self._count += 1

#     def get_metric(self, reset: bool = False) -> Tuple[float, float]:
#         """
#         Returns
#         -------
#         Average exact match and F1 score (in that order) as computed by the official DROP script
#         over all inputs.
#         """
#         exact_match = 100.0 * self._total_em / self._count if self._count > 0 else 0
#         f1_score = 100.0 * self._total_f1 / self._count if self._count > 0 else 0
#         if reset:
#             self.reset()
#         return exact_match, f1_score

#     @overrides
#     def reset(self):
#         self._total_em = 0.0
#         self._total_f1 = 0.0
#         self._count = 0

#     def __str__(self):
#         return f"DropEmAndF1(em={self._total_em}, f1={self._total_f1})"