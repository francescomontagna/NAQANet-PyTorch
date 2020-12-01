
# TODO: turn it into drop
class Drop(data.Dataset):
    """Stanford Question Answering Dataset (SQuAD).
    Each item in the dataset is a tuple with the following entries (in order):
        - context_idxs: Indices of the words in the context.
            Shape (context_len,).
        - context_char_idxs: Indices of the characters in the context.
            Shape (context_len, max_word_len).
        - question_idxs: Indices of the words in the question.
            Shape (question_len,).
        - question_char_idxs: Indices of the characters in the question.
            Shape (question_len, max_word_len).
        - y1: Index of word in the context where the answer begins.
            -1 if no answer.
        - y2: Index of word in the context where the answer ends.
            -1 if no answer.
        - id: ID of the example.
    Args:
        data_path (str): Path to .npz file containing pre-processed dataset.
        use_v2 (bool): Whether to use SQuAD 2.0 questions. Otherwise only use SQuAD 1.1.
    """
    def __init__(self, data_path, use_v2=False):
        super(SQuAD, self).__init__()

        dataset = np.load(data_path)
        self.context_idxs = torch.from_numpy(dataset['context_idxs']).long()
        self.context_char_idxs = torch.from_numpy(dataset['context_char_idxs']).long()
        self.question_idxs = torch.from_numpy(dataset['ques_idxs']).long()
        self.question_char_idxs = torch.from_numpy(dataset['ques_char_idxs']).long()
        self.number_idxs = torch.from_numpy(dataset['number_idxs']).long()
        self.start_idxs = torch.from_numpy(dataset['start_idxs']).long()
        self.end_idxs = torch.from_numpy(dataset['end_idxs']).long()
        self.counts = torch.from_numpy(dataset['counts']).long()
        self.add_sub_expressions = torch.from_numpy(dataset['add_sub_expressions']).long()
        # self.query_ids = torch.from_numpy(dataset['query_ids']).long()

    def __getitem__(self, idx):
        idx = self.valid_idxs[idx]
        example = (self.context_idxs[idx],
                   self.context_char_idxs[idx],
                   self.question_idxs[idx],
                   self.question_char_idxs[idx],
                   self.number_idxs[idx], 
                   self.start_idxs[idx],
                   self.end_idxs[idx],
                   self.counts[idx],
                   self.add_sub_expressions[idx]
                   ) # query id ?

        return example

    def __len__(self):
        return len(self.valid_idxs)

# Need to add padding -1 to start/end indices, number indices, counts and add_sub_expressions
def collate_fn(examples):
    """Create batch tensors from a list of individual examples returned
    by `DROP.__getitem__`. Merge examples of different length by padding
    all examples to the maximum length in the batch.
    Args:
        examples (list): List of tuples of the form (context_idxs, context_char_idxs,
        question_idxs, question_char_idxs, number_indices, start_indices, end_indices
        counts, add_sub_expressions).
    Returns:
        examples (tuple): Tuple of tensors (context_idxs, context_char_idxs, question_idxs,
        question_char_idxs, y1s, y2s, ids). All of shape (batch_size, ...), where
        the remaining dimensions are the maximum length of examples in the input.
    Adapted from:
        https://github.com/yunjey/seq2seq-dataloader
    """
    def merge_0d(scalars, dtype=torch.int64):
        return torch.tensor(scalars, dtype=dtype)

    def merge_1d(arrays, dtype=torch.int64, pad_value=0, pad_with=0):
        lengths = [(a != pad_value).sum() for a in arrays]
        padded = torch.zeros(len(arrays), max(lengths), dtype=dtype) + pad_with
        for i, seq in enumerate(arrays):
            end = lengths[i]
            padded[i, :end] = seq[:end]
        return padded

    def merge_2d(matrices, dtype=torch.int64, pad_value=0, pad_with=0):
        heights = [(m.sum(1) != pad_value).sum() for m in matrices]
        widths = [(m.sum(0) != pad_value).sum() for m in matrices]
        padded = torch.zeros(len(matrices), max(heights), max(widths), dtype=dtype) + pad_with
        for i, seq in enumerate(matrices):
            height, width = heights[i], widths[i]
            padded[i, :height, :width] = seq[:height, :width]
        return padded


    def pad_1d(tensors_list, pad_with):
        lengths = [t.size(0) for t in tensors_list]
        max_length = max(lengths)
        for tensor in tensors_list:
            

    # For tokens padding character is 0 (<PAD> index in vocab)
    # For indices padding character is -1.

    # Group by tensor type
    context_idxs, context_char_idxs, \
        question_idxs, question_char_idxs, \
        y1s, y2s, ids = zip(*examples)

    # Merge into batch tensors
    context_idxs = merge_1d(context_idxs)
    context_char_idxs = merge_2d(context_char_idxs)
    question_idxs = merge_1d(question_idxs)
    question_char_idxs = merge_2d(question_char_idxs)
    y1s = merge_0d(y1s)
    y2s = merge_0d(y2s)
    ids = merge_0d(ids)

    return (context_idxs, context_char_idxs,
            question_idxs, question_char_idxs,
            y1s, y2s, ids)