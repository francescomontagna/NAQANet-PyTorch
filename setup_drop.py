import spacy
import ujson as json
import string
import itertools
import os
import numpy as np

from word2number.w2n import word_to_num
from collections import defaultdict
from subprocess import run
from typing import Dict, List, Union, Tuple, Any
from collections import Counter
from tqdm import tqdm

from code.args_drop import get_setup_drop_args # TODO put everything in single .py file

max_count = 100000

def download_url(url, output_path, show_progress=True):
    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)

    if show_progress:
        # Download with a progress bar
        with DownloadProgressBar(unit='B', unit_scale=True,
                                 miniters=1, desc=url.split('/')[-1]) as t:
            urllib.request.urlretrieve(url,
                                       filename=output_path,
                                       reporthook=t.update_to)
    else:
        # Simple download with no progress bar
        urllib.request.urlretrieve(url, output_path)


def url_to_data_path(url):
    return os.path.join('./data/', url.split('/')[-1])


def download(args):
    downloads = [
        # Can add other downloads here (e.g., other word vectors)
        ('GloVe word vectors', args.glove_url),
    ]

    for name, url in downloads:
        output_path = url_to_data_path(url)
        if not os.path.exists(output_path):
            print(f'Downloading {name}...')
            download_url(url, output_path)

        if os.path.exists(output_path) and output_path.endswith('.zip'):
            extracted_path = output_path.replace('.zip', '')
            if not os.path.exists(extracted_path):
                print(f'Unzipping {name}...')
                with ZipFile(output_path, 'r') as zip_fh:
                    zip_fh.extractall(extracted_path)

    print('Downloading spacy language model...')
    run(['python3', '-m', 'spacy', 'download', 'en'])


WORD_NUMBER_MAP = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}

IGNORED_TOKENS = {"a", "an", "the"}
STRIPPED_CHARACTERS = string.punctuation + "".join(["‘", "’", "´", "`", "_"])


# Removed date, since we do not need it
def extract_answer_info_from_annotation(
    answer_annotation: Dict[str, Any]
    ) -> Tuple[str, List[str]]:
    answer_type = None
    if answer_annotation["spans"]:
        answer_type = "spans"
    elif answer_annotation["number"]:
        answer_type = "number"
    elif any(answer_annotation["date"].values()):
        answer_type = "date"

    answer_content = answer_annotation[answer_type] if answer_type is not None else None

    answer_texts: List[str] = []
    if answer_type is None:  # No answer
        pass
    elif answer_type == "spans":
        # answer_content is a list of string in this case
        answer_texts = answer_content
    elif answer_type == "date":
        # answer_content is a dict with "month", "day", "year" as the keys
        date_tokens = [
            answer_content[key]
            for key in ["month", "day", "year"]
            if key in answer_content and answer_content[key]
        ]
        answer_texts = date_tokens
    elif answer_type == "number":
        # answer_content is a string of number
        answer_texts = [answer_content]
    return answer_type, answer_texts


def convert_word_to_number(word: str, try_to_include_more_numbers=False):
    """
    Currently we only support limited types of conversion.
    """
    if try_to_include_more_numbers:
        # strip all punctuations from the sides of the word, except for the negative sign
        punctruations = string.punctuation.replace("-", "")
        word = word.strip(punctruations)
        # some words may contain the comma as deliminator
        word = word.replace(",", "")
        # word2num will convert hundred, thousand ... to number, but we skip it.
        if word in ["hundred", "thousand", "million", "billion", "trillion"]:
            return None
        try:
            number = word_to_num(word)
        except ValueError:
            try:
                number = int(word)
            except ValueError:
                try:
                    number = float(word)
                except ValueError:
                    number = None
        return number
    else:
        no_comma_word = word.replace(",", "")
        if no_comma_word in WORD_NUMBER_MAP:
            number = WORD_NUMBER_MAP[no_comma_word]
        else:
            try:
                number = int(no_comma_word)
            except ValueError:
                number = None
        return number

def find_valid_add_sub_expressions(
        numbers: List[int], targets: List[int], max_number_of_numbers_to_consider: int = 2
    ) -> List[List[int]]:
    valid_signs_for_add_sub_expressions = []
    # TODO: Try smaller numbers?
    for number_of_numbers_to_consider in range(2, max_number_of_numbers_to_consider + 1):
        possible_signs = list(itertools.product((-1, 1), repeat=number_of_numbers_to_consider))
        for number_combination in itertools.combinations(
            enumerate(numbers), number_of_numbers_to_consider
        ):
            indices = [it[0] for it in number_combination]
            values = [it[1] for it in number_combination]
            for signs in possible_signs:
                eval_value = sum(sign * value for sign, value in zip(signs, values))
                if eval_value in targets:
                    labels_for_numbers = [0] * len(numbers)  # 0 represents ``not included''.
                    for index, sign in zip(indices, signs):
                        labels_for_numbers[index] = (
                            1 if sign == 1 else 2
                        )  # 1 for positive, 2 for negative
                    valid_signs_for_add_sub_expressions.append(labels_for_numbers)
    return valid_signs_for_add_sub_expressions

def find_valid_spans(
    passage_tokens: List[str], answer_texts: List[str] # answer texts = tokenized and recomposed answer texts
) -> List[Tuple[int, int]]:
    normalized_tokens = [
        token.lower().strip(STRIPPED_CHARACTERS) for token in passage_tokens
    ]
    word_positions: Dict[str, List[int]] = defaultdict(list) # ?
    for i, token in enumerate(normalized_tokens):
        word_positions[token].append(i) # dict telling index at which appears each word in the passage
    
    spans = []
    for answer_text in answer_texts:
        answer_tokens = answer_text.lower().strip(STRIPPED_CHARACTERS).split()
        num_answer_tokens = len(answer_tokens)
        if answer_tokens[0] not in word_positions:
            continue
        for span_start in word_positions[answer_tokens[0]]:
            span_end = span_start  # span_end is _inclusive_
            answer_index = 1
            while answer_index < num_answer_tokens and span_end + 1 < len(normalized_tokens):
                token = normalized_tokens[span_end + 1]
                if answer_tokens[answer_index].strip(STRIPPED_CHARACTERS) == token:
                    answer_index += 1
                    span_end += 1
                elif token in IGNORED_TOKENS:
                    span_end += 1
                else:
                    break
            if num_answer_tokens == answer_index: # if I found as many consecutive match as I expected, this is a matching passage
                spans.append((span_start, span_end))
    return spans # list of all matching passage slices
        

def find_valid_counts(count_numbers: List[int], targets: List[int]) -> List[int]:
    valid_indices = []
    for index, number in enumerate(count_numbers):
        if number in targets:
            valid_indices.append(index)
    return valid_indices

def word_tokenize(sent):
    doc = nlp(sent)
    return [token.text for token in doc]

def convert_idx(text, tokens):
    current = 0
    spans = []
    for token in tokens:
        current = text.find(token, current)
        if current < 0:
            print(f"Token {token} cannot be found")
            raise Exception()
        spans.append((current, current + len(token)))
        current += len(token)
    return spans


def process_file(filename, data_type, word_counter, char_counter):
    print(f"Pre-processing {data_type} examples...")
    max_c_len =max_q_len =  0
    examples = []
    eval_examples = {}
    total = 0
    with open(filename, "r") as fh:
        source = json.load(fh)
        for article in tqdm(source.values()): # e.g. "nfl201" : {"passage" : 'this is the passage', "qa_pairs" : [{"question" : 'this is a question', "answer" : {...}, ...}]
            passage = article["passage"] # one passage for each article
            passage = passage.replace(
                "''", '" ').replace("``", '" ')
            passage_tokens = word_tokenize(passage)
            if len(passage_tokens)  > max_c_len:
                max_c_len = len(passage_tokens)
            passage_chars = [list(token) for token in passage_tokens]
            spans = convert_idx(passage, passage_tokens) # [[0, 3], [3, 10], .... [35, 41]] each element is a token represented as [start_index, end_index]
            for token in passage_tokens:
                word_counter[token] += len(article["qa_pairs"]) # += number of qa pairs ???
                for char in token:
                    char_counter[char] += len(article["qa_pairs"])
            for qa_pair in article["qa_pairs"]:
                total += 1
                ques = qa_pair["question"].replace(
                    "''", '" ').replace("``", '" ')
                ques_tokens = word_tokenize(ques)
                if len(ques_tokens) > max_q_len:
                    max_q_len = len(ques_tokens)
                ques_chars = [list(token) for token in ques_tokens]
                for token in ques_tokens:
                    word_counter[token] += 1
                    for char in token:
                        char_counter[char] += 1
                
                answer_annotation = qa_pair['answer']
                # answer type: "number" or "span". answer texts: number or list of spans
                answer_type, answer_texts = extract_answer_info_from_annotation(answer_annotation)

                # Tokenize and recompose the answer text in order to find the matching span based on token
                tokenized_answer_texts = []
                for answer_text in answer_texts:
                    answer_tokens = word_tokenize(answer_text)
                    tokenized_answer_texts.append(" ".join(token for token in answer_tokens))
                
                numbers_in_passage = []
                number_indices = []
                for token_index, token in enumerate(passage_tokens):
                    number = convert_word_to_number(token)
                    if number is not None:
                        numbers_in_passage.append(number)
                        number_indices.append(token_index)
                numbers_as_tokens = [str(number) for number in numbers_in_passage]

                valid_passage_spans = (
                find_valid_spans(passage_tokens, tokenized_answer_texts)
                if tokenized_answer_texts
                else []
                )

                target_numbers = []
                # `answer_texts` is a list of valid answers.
                for answer_text in answer_texts:
                    number = convert_word_to_number(answer_text)
                    if number is not None:
                        target_numbers.append(number)
                valid_signs_for_add_sub_expressions: List[List[int]] = []
                valid_counts: List[int] = []
                if answer_type in ["number", "date"]:
                    valid_signs_for_add_sub_expressions = find_valid_add_sub_expressions(
                        numbers_in_passage, target_numbers
                    )
                if answer_type in ["number"]:
                    # Support count number 0 ~ max_count
                    # Does not support float
                    numbers_for_count = list(range(max_count))
                    valid_counts = find_valid_counts(numbers_for_count, target_numbers)

                type_to_answer_map = {
                    "passage_span": valid_passage_spans,
                    "addition_subtraction": valid_signs_for_add_sub_expressions,
                    "counting": valid_counts,
                }
                
                # print(f"Type to answer map: {type_to_answer_map}") 

                answer_info = {
                    "answer_texts": answer_texts,  # this `answer_texts` will not be used for evaluation
                    "answer_passage_spans": valid_passage_spans,
                    "signs_for_add_sub_expressions": valid_signs_for_add_sub_expressions,
                    "counts": valid_counts,
                }

                # single question answer pair
                example = {"context_tokens": passage_tokens,
                            "context_chars": passage_chars,
                            "ques_tokens": ques_tokens,
                            "ques_chars": ques_chars,
                            "number_indices": number_indices,
                            "answer_info": answer_info
                            }

                examples.append(example)

                # print(f"Answer info: {answer_info}") 
                # print("")
            break 

    return examples


# Used both for word and char embeddings. # No changes
def get_embedding(counter, data_type, limit=-1, emb_file=None, vec_size=None, num_vectors=None):
    print(f"Pre-processing {data_type} vectors...")
    embedding_dict = {}
    filtered_elements = [k for k, v in counter.items() if v > limit] # word not included if associated to few QA pairs. limit is actually -1
    if emb_file is not None:
        assert vec_size is not None
        with open(emb_file, "r", encoding="utf-8") as fh: # open glove/fasttext
            for line in tqdm(fh, total=num_vectors): # for each line = embedding
                array = line.split()
                word = "".join(array[0:-vec_size])
                vector = list(map(float, array[-vec_size:]))
                if word in counter and counter[word] > limit: # if the word is in our data, add the embedding to the matrix
                    embedding_dict[word] = vector
                break
        print(f"{len(embedding_dict)} / {len(filtered_elements)} tokens have corresponding {data_type} embedding vector")
    else:
        assert vec_size is not None
        for token in filtered_elements:
            embedding_dict[token] = [np.random.normal(
                scale=0.1) for _ in range(vec_size)]
            break
        print(f"{len(filtered_elements)} tokens have corresponding {data_type} embedding vector")

    NULL = "--NULL--"
    OOV = "--OOV--"
    token2idx_dict = {token: idx for idx, token in enumerate(embedding_dict.keys(), 2)}
    token2idx_dict[NULL] = 0
    token2idx_dict[OOV] = 1
    embedding_dict[NULL] = [0. for _ in range(vec_size)]
    embedding_dict[OOV] = [0. for _ in range(vec_size)]
    idx2emb_dict = {idx: embedding_dict[token]
                    for token, idx in token2idx_dict.items()}
    emb_mat = [idx2emb_dict[idx] for idx in range(len(idx2emb_dict))]
    return emb_mat, token2idx_dict


def build_features(args, examples, data_type, out_file, word2idx_dict, char2idx_dict, is_test=False):
    para_limit = args.test_para_limit if is_test else args.para_limit
    ques_limit = args.test_ques_limit if is_test else args.ques_limit
    ans_limit = args.ans_limit
    char_limit = args.char_limit

    def drop_example(ex, is_test_=False):
        if is_test_:
            drop = False
        else:
            drop = len(ex["context_tokens"]) > para_limit or \
                   len(ex["ques_tokens"]) > ques_limit

        return drop

    print(f"Converting {data_type} examples to indices...")
    total = 0
    total_ = 0
    meta = {}
    context_idxs = []
    context_char_idxs = []
    ques_idxs = []
    ques_char_idxs = []
    number_idxs = []
    answers_info = []
    print(f"Len examples: {len(examples)}")
    for n, example in tqdm(enumerate(examples)):
        total_ += 1 # count total number of examples in the dataset
        if drop_example(example, is_test):
            continue

        total += 1

        def _get_word(word):# return corresponding word index
            for each in (word, word.lower(), word.capitalize(), word.upper()):
                if each in word2idx_dict:
                    return word2idx_dict[each]
            return 1 # '--OOV--'

        def _get_char(char): # return corresponding char index
            if char in char2idx_dict:
                return char2idx_dict[char]
            return 1

        context_idx = np.zeros([para_limit], dtype=np.int32)
        context_char_idx = np.zeros([para_limit, char_limit], dtype=np.int32)
        ques_idx = np.zeros([ques_limit], dtype=np.int32)
        ques_char_idx = np.zeros([ques_limit, char_limit], dtype=np.int32)

        for i, token in enumerate(example["context_tokens"]):
            context_idx[i] = _get_word(token)
        context_idxs.append(context_idx)

        for i, token in enumerate(example["ques_tokens"]):
            ques_idx[i] = _get_word(token)
        ques_idxs.append(ques_idx)

        for i, token in enumerate(example["context_chars"]):
            for j, char in enumerate(token):
                if j == char_limit:
                    break
                context_char_idx[i, j] = _get_char(char)
        context_char_idxs.append(context_char_idx)

        for i, token in enumerate(example["ques_chars"]):
            for j, char in enumerate(token):
                if j == char_limit:
                    break
                ques_char_idx[i, j] = _get_char(char)
        ques_char_idxs.append(ques_char_idx)

        number_indices = example["number_indices"]
        annotation = example["answer_info"]
        number_idxs.append(number_indices)
        answers_info.append(annotation)

        # for k, v in example.items():
        #     print(f"({k}, {v})")

    np.savez(out_file,
             context_idxs=np.array(context_idxs),
             context_char_idxs=np.array(context_char_idxs),
             ques_idxs=np.array(ques_idxs),
             ques_char_idxs=np.array(ques_char_idxs),
             number_idxs=np.array(number_idxs),
             answers_info=np.array(answers_info))
    print(f"Built {total} / {total_} instances of features in total")
    meta["total"] = total
    return meta


def save(filename, obj, message=None):
    if message is not None:
        print(f"Saving {message}...")
        with open(filename, "w") as fh:
            json.dump(obj, fh)


def pre_process(args):
    word_counter = Counter()
    char_counter = Counter()
    
    # process training dataset
    train_examples = process_file(args.train_file, "train", word_counter, char_counter)
    word_emb_mat, word2idx_dict = get_embedding(word_counter, "word",
        emb_file=args.glove_path, vec_size=args.glove_dim, num_vectors=args.glove_num_vecs)
    char_emb_mat, char2idx_dict = get_embedding(char_counter, "char",
         emb_file=None, vec_size=args.char_dim)

    # process dev dataset
    eval_examples = process_file(args.dev_file, "dev", word_counter, char_counter)

    # build golden files
    build_features(args, train_examples, "train", args.train_record_file, word2idx_dict, char2idx_dict)
    dev_meta = build_features(args, eval_examples, "dev", args.dev_record_file, word2idx_dict, char2idx_dict)

    # save generated files
    save(args.word_emb_file, word_emb_mat, message="word embedding")
    save(args.char_emb_file, char_emb_mat, message="char embedding")

    # TODO decide eval dict form and define it in process_file
    # save(args.train_eval_file, train_eval, message="train eval")
    # save(args.dev_eval_file, dev_eval, message="dev eval")
    save(args.word2idx_file, word2idx_dict, message="word dictionary")
    save(args.char2idx_file, char2idx_dict, message="char dictionary")
    save(args.dev_meta_file, dev_meta, message="dev meta")


if __name__ == "__main__":
    # Get command-line args
    args = get_setup_drop_args()

    # Download resources
    download(args)

    nlp = spacy.blank("en")
    pre_process(args)
