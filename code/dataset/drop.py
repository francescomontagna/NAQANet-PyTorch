import torch
import torch.utils.data as data
import numpy as np

from code.args_drop import get_train_args


class DROP(data.Dataset):
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
    def __init__(self, data_path):
        super(DROP, self).__init__()

        dataset = np.load(data_path, allow_pickle=True)
        self.context_idxs = torch.from_numpy(dataset['context_idxs']).long()
        self.context_char_idxs = torch.from_numpy(dataset['context_char_idxs']).long()
        self.question_idxs = torch.from_numpy(dataset['ques_idxs']).long()
        self.question_char_idxs = torch.from_numpy(dataset['ques_char_idxs']).long()
        # self.number_idxs = torch.from_numpy(dataset['number_idxs']).long()
        self.start_idxs = torch.from_numpy(dataset['start_idxs']).long()
        self.end_idxs = torch.from_numpy(dataset['end_idxs']).long()
        self.counts = torch.from_numpy(dataset['counts']).long()
        # self.add_sub_expressions = torch.from_numpy(dataset['add_sub_expressions']).long()
        self.ids = torch.from_numpy(dataset['ids']).long()

    def __getitem__(self, idx):
        example = (self.context_idxs[idx],
                   self.context_char_idxs[idx],
                   self.question_idxs[idx],
                   self.question_char_idxs[idx],
                #    self.number_idxs[idx], 
                   self.start_idxs[idx],
                   self.end_idxs[idx],
                   self.counts[idx],
                #    self.add_sub_expressions[idx],
                   self.ids[idx]
                   )

        return example

    def __len__(self):
        return self.context_idxs.size(0)

# Need to add padding -1 to start/end indices, number indices, counts and add_sub_expressions
def collate_fn(examples):
    """Create batch tensors from a list of individual examples returned
    by `DROP.__getitem__`. Merge examples of different length by padding
    all examples to the maximum length in the batch.
    Args:
        examples (list): List of tuples of the form (context_idxs, context_char_idxs,
        question_idxs, question_char_idxs, number_indices, start_indices, end_indices
        counts, add_sub_expressions, ids).
    Returns:
        examples (tuple): Tuple of tensors (context_idxs, context_char_idxs,
        question_idxs, question_char_idxs, number_indices, start_indices, end_indices
        counts, add_sub_expressions, ids). All of shape (batch_size, ...), where
        the remaining dimensions are the maximum length of examples in the input.
    Adapted from:
        https://github.com/yunjey/seq2seq-dataloader
    """
    def merge_0d(scalars, dtype=torch.int64):
        return torch.tensor(scalars, dtype=dtype).unsqueeze(-1)

    def merge_1d(arrays, dtype=torch.int64, pad_value=0):
        lengths = [(a != pad_value).sum() for a in arrays]
        max_length = max(lengths)
        padded = torch.zeros(len(arrays), max(max_length, 1), dtype=dtype) + pad_value
        for i, seq in enumerate(arrays):
            end = lengths[i]
            padded[i, :end] = seq[:end]
        return padded

    def merge_2d(matrices, dtype=torch.int64, pad_value=0, add_sub = False):
        if pad_value == 0:
            heights = [((m).sum(1) != pad_value).sum() for m in matrices]
            widths = [((m).sum(0) != pad_value).sum() for m in matrices]
        else:
            heights = [((m != pad_value).sum(1) != 0).sum() for m in matrices]
            widths = [((m != pad_value).sum(0) != 0).sum() for m in matrices]
        padded = torch.zeros(len(matrices), max(heights), max(widths), dtype=dtype) + pad_value
        for i, seq in enumerate(matrices):
            height, width = heights[i], widths[i]
            padded[i, :height, :width] = seq[:height, :width]
        return padded

    # For tokens padding character is 0 (<PAD> index in vocab)
    # For indices padding character is -1.

    # Group by tensor type
    context_idxs, context_char_idxs, \
        question_idxs, question_char_idxs, \
        start_indices, end_indices, \
        counts, ids = zip(*examples)

    # Merge into batch tensors
    context_idxs = merge_1d(context_idxs)
    context_char_idxs = merge_2d(context_char_idxs)
    question_idxs = merge_1d(question_idxs)
    question_char_idxs = merge_2d(question_char_idxs)

    # number_indices = merge_1d(number_indices, pad_value = -1)
    start_indices = merge_1d(start_indices, pad_value = -1)
    end_indices = merge_1d(end_indices, pad_value = -1)
    counts = merge_0d(counts) # TODO check
    # add_sub_expressions = merge_2d(add_sub_expressions, pad_value = -1)

    ids = merge_0d(ids)

    return (context_idxs, context_char_idxs,
            question_idxs, question_char_idxs,
            start_indices, end_indices,
            counts, ids
            )

if __name__ == "__main__":
    args = get_train_args()
    torch.manual_seed(224)
    print("Building datasets...", end = " ")
    train_dataset = DROP(args.train_record_file)
    train_loader = data.DataLoader(train_dataset,
                                   batch_size=args.batch_size,
                                   shuffle=True,
                                   num_workers=args.num_workers,
                                   collate_fn=collate_fn)
    dev_dataset = DROP(args.dev_record_file)
    dev_loader = data.DataLoader(dev_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 num_workers=args.num_workers,
                                 collate_fn=collate_fn)
    print("Done!")

    for i, example in enumerate(train_loader):
        context_idxs, context_char_idxs, \
        question_idxs, question_char_idxs, \
        start_indices, end_indices, \
        counts, ids = example

        print(start_indices.size())
        print(end_indices.size())
        # print(context_idxs.size())
        # print(question_char_idxs.size())
        # print(counts.size())
        # print(ids.size())
        
        if i >= 3:
            break