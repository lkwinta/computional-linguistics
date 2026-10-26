import argparse
import os
from pathlib import Path

from custom_tokenizers.sentence_piece_bpe import SentencePieceBPETokenizer
from custom_tokenizers.white_space import WhiteSpaceTokenizer

BASE_DIR = Path(__file__).resolve().parent

def train_sentencepiece(args):
    tokenizer = SentencePieceBPETokenizer()
    tokenizer.train(BASE_DIR / args.corpus, vocab_size=args.vocab_size)
    tokenizer.save(f"{BASE_DIR / args.models_dir}/sp_bpe.model")

    print(f"Saved SentencePiece BPE tokenizer to: {args.models_dir}/sp_bpe.model")

def train_whitespace(args):
    tokenizer = WhiteSpaceTokenizer()
    tokenizer.train(BASE_DIR / args.corpus, args.vocab_size)
    tokenizer.save(f"{BASE_DIR / args.models_dir}/ws.model")

    print(f"Saved WhiteSpace tokenizer to: {args.models_dir}/ws.model")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=str, default="corpus/train.txt")
    parser.add_argument("--vocab_size", type=int, default=50257)
    parser.add_argument("--models_dir", type=str, default="models")
    args = parser.parse_args()

    print(f"Training tokenizers with corpus: {args.corpus}, vocab size: {args.vocab_size}, models directory: {BASE_DIR / args.models_dir}")

    os.makedirs(BASE_DIR / args.models_dir, exist_ok=True)
    
    train_sentencepiece(args)
    train_whitespace(args)
    

if __name__ == "__main__":
    main()
