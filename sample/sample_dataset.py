import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


def generate_batch(batch):
    text = [entry[0] for entry in batch] # text è gia il vettore degli embeddings dopo get items
    labels = torch.tensor([entry[1] for entry in batch])
    text = pad_sequence(text, batch_first=True)
    return text, labels


class MultilingualDataset(Dataset):

    def __init__(self, corpus: pd.Series, labels: pd.Series, vocab, dtype: torch.dtype):
        assert len(corpus) == len(labels), "X and Y must have the same length"
        self.corpus = corpus
        self.labels = labels
        self.vocab = vocab
        self.dtype = dtype

    def __getitem__(self, i):
        text = self.corpus.iloc[i]
        tensor = self.vocab.encode(text.split())

        # Restituisce vettori degli embeddings per corpus[i], label[i]
        return tensor, torch.tensor(self.labels.iloc[i], dtype=self.dtype)

    def __len__(self):
        return len(self.corpus)