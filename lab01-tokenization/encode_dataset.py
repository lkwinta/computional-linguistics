import argparse
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from custom_tokenizers import (
    GPT2BPETokenizer,
    SentencePieceBPETokenizer,
    WhiteSpaceTokenizer,
)


BASE_DIR = Path(__file__).resolve().parent
CHUNK_SIZE = 1024 * 1024
SPLIT_FILES = {
    "train": "train.txt",
    "val": "val.txt",
}


def read_chunks(input_file):
    while chunk := input_file.read(CHUNK_SIZE):
        yield chunk + input_file.readline()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=BASE_DIR / "corpus",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BASE_DIR / "encoded_dataset",
    )
    args = parser.parse_args()

    whitespace_tokenizer = WhiteSpaceTokenizer()
    whitespace_tokenizer.load(str(BASE_DIR / "models" / "ws.model"))

    sentencepiece_tokenizer = SentencePieceBPETokenizer()
    sentencepiece_tokenizer.load(
        str(BASE_DIR / "models" / "sp_bpe.model")
    )

    tokenizers = {
        "gpt2": GPT2BPETokenizer(),
        "whitespace": whitespace_tokenizer,
        "sentencepiece": sentencepiece_tokenizer,
    }

    for name, tokenizer in tokenizers.items():
        tokenizer_dir = args.output_dir / name
        tokenizer_dir.mkdir(parents=True, exist_ok=True)

        for split, filename in SPLIT_FILES.items():
            input_path = args.input_dir / filename
            output_path = tokenizer_dir / f"{split}.bin"

            with (
                input_path.open("r", encoding="utf-8") as input_file,
                output_path.open("wb") as output_file,
            ):
                chunks = read_chunks(input_file)
                for text in tqdm(chunks, desc=f"{name}: {split}"):
                    token_ids = tokenizer.encode(text)
                    np.asarray(token_ids, dtype=np.uint16).tofile(
                        output_file
                    )


if __name__ == "__main__":
    main()
