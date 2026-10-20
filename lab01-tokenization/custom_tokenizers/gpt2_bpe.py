from transformers import AutoTokenizer

class GPT2BPETokenizer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
        self.vocab_size = self.tokenizer.vocab_size

    def encode(self, text):
        return self.tokenizer.encode(text)

    def decode(self, tokens):
        return self.tokenizer.decode(tokens)
