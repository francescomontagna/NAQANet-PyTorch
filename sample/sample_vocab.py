import io
import logging
import torch


LOG = logging.getLogger(__name__)


def load_vocabs(config):
    LOG.debug("Loading vectors")
    common = SampleVocab("common_embeddings.txt")
    LOG.debug("Common vectors: %s", str(common.vectors.size()))

    vocabs = dict()
    for lang in config.languages:
        vocabs[lang] = SampleVocab("sample_embeddings.txt", base=common)
        LOG.debug("Vectors %s: %s", lang.upper(), str(vocabs[lang].vectors.size()))
    return vocabs



class SampleVocab:
    """Class that defines token and word vectors, mapping their respective indices.
    """

    def __init__(self, emb_path, emb_dim=300, base=None):
        """
        Creates a new vocab instance.
        :param emb_path:    path to the embeddings
        :param emb_dim:     number of vector features
        :param base:        Vocab instance to use as starting point (e.g. common vectors)
        """
        super().__init__()
        self.token2index = dict()
        self.index2token = dict()
        self.vectors = torch.empty(0, dtype=torch.float)
        self._embedding_dim = emb_dim
        if base is not None:
            LOG.info("Starting from existing base: %s", base.language)
            self.index2token = dict(base.index2token)
            self.token2index = dict(base.token2index)
            assert emb_dim == base.vectors.size(1), "Common vectors and expected dim do not match"
            self.vectors = base.vectors.clone()
        self._load_vectors(emb_path)

    def __len__(self):
        """Returns the length of the dictionary.
        """
        return len(self.token2index)

    def _load_vectors(self, path):
        """Loads the given file containing word and their embeddings.
        Result is then stored in this vocab instance.
        """
        LOG.info(f"Loading {path}...")
        with io.open(path, mode="r", encoding="utf-8", newline="\n", errors="strict") as f:
            # check embedding size
            index = len(self.vectors)
            d = 300 # embeddings size
            n = 9 # vocab size
            assert self._embedding_dim == d, "Expected and actual vector dimensions do not match"
            # preallocate the required memory for the embeddings
            new_vectors = torch.empty((n, d), dtype=torch.float)
            self.vectors = torch.cat((self.vectors, new_vectors), dim=0)
            # read and add to main torch matrix
            for line in f:
                tokens = line.rstrip().split(' ')
                word = tokens[0]
                if word in self.token2index:
                    # skip if already stored as base
                    continue
                self.token2index[word] = index
                self.index2token[index] = word
                self.vectors[index] = torch.tensor([float(v) for v in tokens[1:]], dtype=torch.float)
                index += 1

    def _get_index(self, token):
        """Gets the index for a single token.
        If the token exists, its index is returned, if it does not exist, but the token is numeric,
        a special <NUM> index is returned,
        """
        index = self.token2index.get(token)
        if index is None:
            try:
                _ = float(token)
                return self.token2index["<NUM>"]
            except ValueError:
                return self.token2index["<UNK>"]
        return index

    def encode(self, tokens):
        """Retrieves a matrix composed by token embeddings.
        """
        indices = [self._get_index(t.lower()) for t in tokens]
        return self.vectors[indices]
