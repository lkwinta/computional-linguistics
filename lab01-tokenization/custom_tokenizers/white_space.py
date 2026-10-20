import json
import re
from collections import Counter
from tqdm.auto import tqdm

class WhiteSpaceTokenizer:
    UNKNOWN_TOKEN = "<UNK>"

    def __init__(self):
        self.vocab = {}
        self.vocab_size = 0

    def train(self, train_path: str, vocab_size: int):
        self.vocab = WhiteSpaceTokenizer._create_vocab(train_path, vocab_size - 1)
        self.vocab[self.UNKNOWN_TOKEN] = len(self.vocab)
        self.unk_token_id = self.vocab[self.UNKNOWN_TOKEN]

        if vocab_size - 1 > len(self.vocab):
            raise ValueError("The specified vocab size is larger than the number of unique tokens in the training data.")

        self.vocab_size = len(self.vocab)

    def save(self, model_path: str):
        with open(model_path, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f)

    def load(self, model_path: str):
        with open(model_path, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)
        self.unk_token_id = self.vocab[self.UNKNOWN_TOKEN]
        self.vocab_size = len(self.vocab)

    @staticmethod
    def _create_vocab(train_path: str, vocab_size: int):
        vocab = {}
        counter = Counter()

        with open(train_path, "r", encoding="utf-8") as f:
            total_lines = sum(1 for _ in f)

        with open(train_path, "r", encoding="utf-8") as f:
            for line in tqdm(f, total=total_lines, desc="Reading lines"):
                tokens = re.findall(r"\w+|[^\w\s]", line)
                counter.update(tokens)

        # Keep only the most common tokens up to the specified vocab size
        counter = counter.most_common(vocab_size)

        for idx, (token, _) in enumerate(counter):
            vocab[token] = idx

        return vocab

    def encode(self, text):
        tokens = re.findall(r"\w+|[^\w\s]", text)
        return [self.vocab.get(token, self.unk_token_id) for token in tokens]

    def decode(self, tokens):
        inv_vocab = {idx: token for token, idx in self.vocab.items()}
        return " ".join([inv_vocab.get(token_id, "<UNK>") for token_id in tokens])
