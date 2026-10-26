
import argparse
import hashlib
from contextlib import ExitStack
from pathlib import Path

from speakleash import Speakleash

BASE_DIR = Path(__file__).resolve().parent
DATASET_NAME = "wolne_lektury_corpus"

SPLIT_FILES = {
    "train": "train.txt",
    "validation": "val.txt",
    "test": "test.txt",
}


def mib_to_bytes(value: float | None) -> int | None:
    return None if value is None else int(value * 1024 * 1024)


def choose_split(document: bytes, seed: int) -> str:
    digest = hashlib.blake2b(
        document,
        digest_size=8,
        key=str(seed).encode("utf-8"),
    ).digest()

    p = int.from_bytes(digest, "big") / 2**64

    if p < 0.90:
        return "train"
    if p < 0.95:
        return "validation"
    return "test"


def export_corpus(
    documents,
    output_dir: Path,
    seed: int,
    limits: dict[str, int | None],
) -> dict[str, tuple[int, int]]:
    rows = {split: 0 for split in SPLIT_FILES}
    sizes = {split: 0 for split in SPLIT_FILES}

    with ExitStack() as stack:
        files = {
            split: stack.enter_context(
                (output_dir / filename).open("wb")
            )
            for split, filename in SPLIT_FILES.items()
        }

        for text in documents:
            if not isinstance(text, str):
                raise TypeError("Expected text from SpeakLeash")
            if not text.strip():
                continue

            document = (
                text.rstrip("\r\n") + "\n\n"
            ).encode("utf-8")

            split = choose_split(document, seed)
            limit = limits[split]

            # Never split a book between files
            if (
                limit is not None
                and sizes[split] + len(document) > limit
            ):
                continue

            files[split].write(document)
            sizes[split] += len(document)
            rows[split] += 1

    return {
        split: (rows[split], sizes[split])
        for split in SPLIT_FILES
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export SpeakLeash corpus to train/val/test."
    )

    parser.add_argument(
        "--dataset-name",
        type=str,
        default=DATASET_NAME,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("corpus"),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("datasets"),
    )
    parser.add_argument("--train-mb", type=float, default=None)
    parser.add_argument("--val-mb", type=float, default=None)
    parser.add_argument("--test-mb", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    for name in ("train_mb", "val_mb", "test_mb"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(
                f"--{name.replace('_', '-')} must be greater than 0"
            )

    (BASE_DIR / args.output_dir).mkdir(parents=True, exist_ok=True)
    (BASE_DIR / args.cache_dir).mkdir(parents=True, exist_ok=True)

    print(f"Loading SpeakLeash dataset: {args.dataset_name}")

    sl = Speakleash(str(BASE_DIR / args.cache_dir))
    dataset = sl.get(args.dataset_name)

    if dataset is None:
        parser.error(f"Dataset not found: {args.dataset_name}")

    limits = {
        "train": mib_to_bytes(args.train_mb),
        "validation": mib_to_bytes(args.val_mb),
        "test": mib_to_bytes(args.test_mb),
    }

    stats = export_corpus(
        dataset.data,
        BASE_DIR / args.output_dir,
        args.seed,
        limits,
    )

    for split, filename in SPLIT_FILES.items():
        rows, size = stats[split]
        path = BASE_DIR / args.output_dir / filename

        print(
            f"{path}: {rows:,} documents, "
            f"{size / 1024**2:.2f} MiB"
        )

        if rows == 0:
            print(f"Warning: {split} split is empty.")

    print("Done. Train tokenizers on train.txt only.")


if __name__ == "__main__":
    main()
