import torch
import os

# Only for debugging
def get_embeddings(batch: torch.tensor, emb_size):
    """
    :param batch: torch.tensor of size (N, L) with indices of context/sentence of interest
    :param emb_size: size of the embeddings
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
    return target * mask + (1 - mask) * (-1e30)