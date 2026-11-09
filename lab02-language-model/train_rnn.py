import json
import argparse

from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from models.rnn import RNN, LSTM
from custom_tokenizers import SentencePieceBPETokenizer


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "encoded_dataset"
TOKENIZER_PATH = BASE_DIR / "output/sp_bpe.model"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"

BLOCK_SIZE = 256
BATCH_SIZE = 16
STEPS = 20000
EVAL_INTERVAL = 100
EVAL_BATCHES = 20
LEARNING_RATE = 3e-4

EMBEDDING_DIM = 384
HIDDEN_SIZE = 512
NUM_LAYERS = 2
GRAD_CLIP = 1.0


def get_batch(data, device):
    starts = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([
        torch.from_numpy(data[i:i + BLOCK_SIZE].astype(np.int64))
        for i in starts
    ])
    y = torch.stack([
        torch.from_numpy(data[i + 1:i + BLOCK_SIZE + 1].astype(np.int64))
        for i in starts
    ])
    return x.to(device), y.to(device)


@torch.no_grad()
def evaluate(model, data, device):
    model.eval()
    losses = []
    for _ in range(EVAL_BATCHES):
        x, y = get_batch(data, device)
        logits, _ = model(x)
        loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", choices=["rnn", "lstm"], default="rnn")
    parser.add_argument(
        "--steps",
        type=int,
        default=STEPS
    )
    args = parser.parse_args()
    
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = SentencePieceBPETokenizer()
    tokenizer.load(str(TOKENIZER_PATH))

    train_data = np.memmap(
        DATA_DIR / "train.bin",
        dtype=np.uint16,
        mode="r",
    )
    val_data = np.memmap(
        DATA_DIR / "val.bin",
        dtype=np.uint16,
        mode="r",
    )

    config = {
        "vocab_size": tokenizer.vocab_size,
        "embedding_dim": EMBEDDING_DIM,
        "hidden_size": HIDDEN_SIZE,
        "num_layers": NUM_LAYERS,
    }
    if args.type == "rnn":
        model = RNN(**config).to(device)
    else:
        model = LSTM(**config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    print(f"device: {device}")
    print(f"train tokens: {len(train_data):,}, validation tokens: {len(val_data):,}")

    history = {"train": [], "validation": []}
    progress = tqdm(range(1, args.steps + 1), desc="Training")
    if device == "cuda":
        torch.cuda.synchronize()
    start_time = perf_counter()
    for step in progress:
        x, y = get_batch(train_data, device)
        logits, _ = model(x)
        loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        train_loss = loss.item()
        history["train"].append({"step": step, "loss": train_loss})
        progress.set_postfix(loss=f"{train_loss:.4f}")

        if step % EVAL_INTERVAL == 0 or step == args.steps:
            val_loss = evaluate(model, val_data, device)
            history["validation"].append({"step": step, "loss": val_loss})
            progress.write(f"step {step}: train {train_loss:.4f}, val {val_loss:.4f}")

    if device == "cuda":
        torch.cuda.synchronize()
    history["training_time_seconds"] = perf_counter() - start_time
    print(f"Training time (including validation): {history['training_time_seconds']:.2f} s")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "config": config,
            "model_type": args.type,
            "tokenizer": "sentencepiece",
            "tokenizer_model": str(TOKENIZER_PATH),
            "block_size": BLOCK_SIZE,
            "val_loss": val_loss,
            "step": args.steps,
        },
        CHECKPOINT_DIR / f"{args.type}_sentencepiece.pt",
    )
    (CHECKPOINT_DIR / f"{args.type}_sentencepiece_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    print(f"Saved model and training history to {CHECKPOINT_DIR}")


if __name__ == "__main__":
    main()
