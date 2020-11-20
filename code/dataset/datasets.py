import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from trainer.util import tokenize
import nltk

nltk.download('punkt')

def generate_batch(batch):
    context_tokens = [entry[0] for entry in batch]
    context_emb = [entry[1] for entry in batch]
    question_emb = [entry[2] for entry in batch]
    context_emb = pad_sequence(context_emb, batch_first=True)
    question_emb = pad_sequence(question_emb, batch_first=True)

    Y1 = torch.tensor([entry[3] for entry in batch])
    Y2 = torch.tensor([entry[4] for entry in batch])
    question_ids = [entry[5] for entry in batch]

    return context_tokens, context_emb, question_emb, Y1, Y2, question_ids

def char2word_index(labels:list, context:str):
    """

    :param labels: [start_char_index, end_char_index]
    :param context: context sentence
    :return: [start_word_index, end_word_index]
    """
    start_index, end_index = labels[0], labels[1]
    start_token = 'starttoken'
    end_token = 'endtoken'
    updated_context = context[0:start_index] + start_token + context[start_index:end_index] + end_token

    tokens = tokenize(updated_context)

    y1 = y2 = 0
    for index in range(len(tokens)):
        token = tokens[index]
        if token.find(start_token) != -1:
            y1 = index
        if token.find(end_token) != -1:
            y2 = index
            return [y1, y2]

    assert "Error: returning -1"
    return [-1, -1]


class MultilingualDataset(Dataset):

    def __init__(self, data: pd.DataFrame, vocabs):
        self.q_id = data.id
        self.context_passages = data.context
        self.questions = data.question
        self.labels = data.label
        self.answer = data.answer
        self.vocabs = vocabs

        # for row in self.context_passages:
        #     assert row.lang in vocabs, f"Item {row.index} with unsupported language: {row.lang}"

    def __getitem__(self, i):
        question_id = self.q_id.iloc[i]
        context = self.context_passages.iloc[i]
        question = self.questions.iloc[i]
        # context_emb = self.vocabs.get(context.lang).encode(context.text)
        # question_emb = self.vocabs.get(question.lang).encode(question.text)
        context_tokens = tokenize(context)
        context_emb = self.vocabs.encode(context_tokens)
        question_emb = self.vocabs.encode(tokenize(question))
        y1, y2 = char2word_index(self.labels.iloc[i], context) # ground truth span indices

        # Restituisce vettori degli embeddings per corpus[i], label[i]
        return context_tokens, context_emb, question_emb, y1, y2, question_id

    def __len__(self):
        return len(self.context_passages)