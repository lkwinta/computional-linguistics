import argparse
import os
from pathlib import Path

from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()

DATASET_NAME_WIKITEXT = "Salesforce/wikitext"
DATASET_CONFIG_WIKITEXT = "wikitext-103-raw-v1"

DATASET_NAME_TINYSTORIES = "roneneldan/TinyStories"

SPLIT_FILES = {
    "train": "train.txt",
    "validation": "val.txt",
    "test": "test.txt",
}


def export_split(dataset, path: Path, limit_mb: float | None = None) -> tuple[int, int]:
    limit_bytes = int(limit_mb * 1024 * 1024) if limit_mb is not None else None
    written_bytes = 0
    written_rows = 0

    with path.open("wb") as f:
        for row in dataset:
            encoded_line = (row["text"] + "\n").encode("utf-8")
            if limit_bytes is not None and written_bytes + len(encoded_line) > limit_bytes:
                break
            f.write(encoded_line)
            written_bytes += len(encoded_line)
            written_rows += 1

    return written_rows, written_bytes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exports Hugging Face Dataset to train.txt, val.txt and test.txt files.",
    )
    parser.add_argument(
        "--dataset-name", type=str, choices=["wikitext", "tinystories"], default="wikitext",
        help="Dataset name (default: wikitext).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("corpus"),
        help="Output directory (default: corpus).",
    )
    parser.add_argument(
        "--train-mb", type=float, default=None,
        help="Optional limit for train.txt in MiB; val and test remain full.",
    )
    args = parser.parse_args()

    if args.train_mb is not None and args.train_mb <= 0:
        parser.error("--train-mb must be greater than 0")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.dataset_name == "wikitext":
        print(f"Downloading {DATASET_NAME_WIKITEXT} ({DATASET_CONFIG_WIKITEXT})...")
        dataset = load_dataset(DATASET_NAME_WIKITEXT, DATASET_CONFIG_WIKITEXT, token=os.environ.get("HF_TOKEN"))
    else:
        print(f"Downloading {DATASET_NAME_TINYSTORIES}...")
        dataset = load_dataset(DATASET_NAME_TINYSTORIES, token=os.environ.get("HF_TOKEN"))

    for split, filename in SPLIT_FILES.items():
        if split not in dataset:
            print(f"Warning: {split} split not found in dataset, skipping.")
            continue

        path = args.output_dir / filename
        limit_mb = args.train_mb if split == "train" else None
        rows, size = export_split(dataset[split], path, limit_mb)
        print(f"{path}: {rows:,} rows, {size / 1024**2:.2f} MiB")

    print("Done. Use only train.txt for tokenizer training.")


if __name__ == "__main__":
    main()
