import io
import sentencepiece as spm
from tqdm.auto import tqdm

class SentencePieceBPETokenizer:
    def __init__(self):
        self.tokenizer = None
        self.vocab_size = 0

    def train(self, train_path: str, vocab_size: int):
        model_buffer = io.BytesIO()

        with open(train_path, "r", encoding="utf-8") as f:
            total_lines = sum(1 for _ in f)

        with open(train_path, "r", encoding="utf-8") as f:
            spm.SentencePieceTrainer.train(
                sentence_iterator=iter(tqdm(
                    f,
                    total=total_lines,
                    desc="Reading corpus"
                )),
                model_writer=model_buffer,
                vocab_size=vocab_size,
                model_type="bpe",
                character_coverage=1.0,
                hard_vocab_limit=True
            )

        self.tokenizer = spm.SentencePieceProcessor(
            model_proto=model_buffer.getvalue()
        )
        self.vocab_size = self.tokenizer.get_piece_size()

    def save(self, model_path: str):
        with open(model_path, "wb") as f:
            f.write(self.tokenizer.serialized_model_proto())

    def load(self, model_path: str):
        self.tokenizer = spm.SentencePieceProcessor(
            model_file=model_path
        )
        self.vocab_size = self.tokenizer.get_piece_size()

    def encode(self, text):
        return self.tokenizer.encode(text)

    def decode(self, tokens):
        return self.tokenizer.decode(tokens)
