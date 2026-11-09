import argparse
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from custom_tokenizers import (
    SentencePieceBPETokenizer,
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

def total_chunks_in_file(input_file):
    total_chunks = 0
    while chunk := input_file.read(CHUNK_SIZE):
        total_chunks += 1
        input_file.readline()
    input_file.seek(0)
    return total_chunks


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

    sentencepiece_tokenizer = SentencePieceBPETokenizer()
    sentencepiece_tokenizer.load(
        str(BASE_DIR / "output" / "sp_bpe.model")
    )

    tokenizer_dir = args.output_dir
    tokenizer_dir.mkdir(parents=True, exist_ok=True)

    for split, filename in SPLIT_FILES.items():
        input_path = args.input_dir / filename
        output_path = tokenizer_dir / f"{split}.bin"

        with input_path.open("r", encoding="utf-8") as input_file, output_path.open("wb") as output_file:
            total_chunks = total_chunks_in_file(input_file)
            chunks = read_chunks(input_file)
            for text in tqdm(chunks, desc=f"{split}", total=total_chunks):
                token_ids = sentencepiece_tokenizer.encode(text)
                np.asarray(token_ids, dtype=np.uint16).tofile(
                    output_file
                )


if __name__ == "__main__":
    main()
