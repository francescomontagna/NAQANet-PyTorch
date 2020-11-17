import torch
from torch.nn.functional import softmax
import os

def get_best_span(span_start_logits: torch.Tensor, span_end_logits: torch.Tensor) -> torch.Tensor:
    """
    This acts the same as the static method ``BidirectionalAttentionFlow.get_best_span()``
    in ``allennlp/models/reading_comprehension/bidaf.py``. We keep it here so that users can
    directly import this function without the class.
    We call the inputs "logits" - they could either be unnormalized logits or normalized log
    probabilities.  A log_softmax operation is a constant shifting of the entire logit
    vector, so taking an argmax over either one gives the same result.
    """
    if span_start_logits.dim() != 2 or span_end_logits.dim() != 2:
        raise ValueError("Input shapes must be (batch_size, passage_length)")
    batch_size, passage_length = span_start_logits.size()
    device = span_start_logits.device
    # (batch_size, passage_length, passage_length)
    span_log_probs = span_start_logits.unsqueeze(2) + span_end_logits.unsqueeze(1)
    # Only the upper triangle of the span matrix is valid; the lower triangle has entries where
    # the span ends before it starts.
    span_log_mask = torch.triu(torch.ones((passage_length, passage_length), device=device)).log()
    valid_span_log_probs = span_log_probs + span_log_mask

    # Here we take the span matrix and flatten it, then find the best span using argmax.  We
    # can recover the start and end indices from this flattened list using simple modular
    # arithmetic.
    # (batch_size, passage_length * passage_length)
    best_spans = valid_span_log_probs.view(batch_size, -1).argmax(-1)
    span_start_indices = best_spans // passage_length
    span_end_indices = best_spans % passage_length
    return torch.stack([span_start_indices, span_end_indices], dim=-1)

# Only for debugging
def get_embeddings(batch: torch.tensor, emb_size):
    """

    :param batch: torch.tensor of size (N, L) with indices of context/sentence of interest
    :return: torch.tensor of size (N, L, E)
    """
    emb_size = emb_size
    working_directory = os.getcwd()
    path = os.path.join(working_directory, os.pardir, "sample_embeddings.txt")
    with open(path, "r") as f:
        embeddings = f.readlines()

    batch_embeddings = []
    for sequence in batch:
        sentence = []
        for i, index in enumerate(sequence):
            if int(index) != 0:
                collection = list(map(float, embeddings[index].split()[1:emb_size+1]))
            else:
                collection = [0. for _ in range(emb_size)]
            sentence.append(collection)
        batch_embeddings.append(sentence)

    return torch.tensor(batch_embeddings)


def info_value_of_dtype(dtype: torch.dtype):
    """
    Returns the `finfo` or `iinfo` object of a given PyTorch data type. Does not allow torch.bool.
    """
    if dtype == torch.bool:
        raise TypeError("Does not support torch.bool")
    elif dtype.is_floating_point:
        return torch.finfo(dtype)
    else:
        return torch.iinfo(dtype)


def min_value_of_dtype(dtype: torch.dtype):
    """
    Returns the minimum value of a given PyTorch data type. Does not allow torch.bool.
    """
    return info_value_of_dtype(dtype).min


def masked_softmax(
        vector: torch.Tensor,
        mask: torch.BoolTensor,
        dim: int = -1):
    if mask is None:
        return torch.nn.functional.softmax(vector, dim=dim)

    masked_vector = vector.masked_fill(mask.unsqueeze(1), min_value_of_dtype(vector.dtype))
    return softmax(masked_vector, dim=dim)

def set_mask(tensor: torch.tensor, negated: bool = False) -> torch.tensor:
    """
    :param tensor:
    :param negated: negated wrt attention. I can reuse the same with negation
    :return:
    """
    
    c_mask = (torch.zeros_like(tensor) != tensor) # actually, when there is the pad character
    if negated:
        return torch.tensor([[False if el == 0 else True for el in sentence] for sentence in c_mask.sum(-1)])

    return torch.tensor([[True if el == 0 else False for el in sentence] for sentence in c_mask.sum(-1)])


def mask_logits(target, mask):
    mask = mask.type(torch.float32)
    return target * mask + (1 - mask) * (-1e30)  # !!!!!!!!!!!!!!!  do we need * mask after target?
