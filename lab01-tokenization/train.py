import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from nano_gpt import GPT, GPTConfig


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "encoded_dataset"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"

VOCAB_SIZE = 50257
BLOCK_SIZE = 256
BATCH_SIZE = 16
STEPS = 20000
EVAL_INTERVAL = 100
EVAL_BATCHES = 20
LEARNING_RATE = 3e-4

N_LAYER = 6
N_HEAD = 6
N_EMBD = 384
DROPOUT = 0.1


def get_batch(data, device):
    starts = torch.randint(
        len(data) - BLOCK_SIZE - 1,
        (BATCH_SIZE,),
    )
    x = torch.stack([
        torch.from_numpy(
            data[i:i + BLOCK_SIZE].astype(np.int64)
        )
        for i in starts
    ])
    y = torch.stack([
        torch.from_numpy(
            data[i + 1:i + BLOCK_SIZE + 1].astype(np.int64)
        )
        for i in starts
    ])
    return x.to(device), y.to(device)


@torch.no_grad()
def evaluate(model, data, device):
    model.eval()
    losses = []

    for _ in range(EVAL_BATCHES):
        x, y = get_batch(data, device)
        _, loss = model(x, y)
        losses.append(loss.item())

    model.train()
    return sum(losses) / len(losses)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "tokenizer",
        choices=["gpt2", "whitespace", "sentencepiece"],
    )
    args = parser.parse_args()

    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer_dir = DATA_DIR / args.tokenizer

    train_data = np.memmap(
        tokenizer_dir / "train.bin",
        dtype=np.uint16,
        mode="r",
    )
    val_data = np.memmap(
        tokenizer_dir / "val.bin",
        dtype=np.uint16,
        mode="r",
    )

    config = GPTConfig(
        vocab_size=VOCAB_SIZE,
        block_size=BLOCK_SIZE,
        n_layer=N_LAYER,
        n_head=N_HEAD,
        n_embd=N_EMBD,
        dropout=DROPOUT,
        bias=False,
    )
    model = GPT(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    print(f"device: {device}")
    print(f"tokenizer: {args.tokenizer}")
    print(f"train tokens: {len(train_data):,}")
    print(f"validation tokens: {len(val_data):,}")

    history = {
        "train": [],
        "validation": [],
    }
    progress = tqdm(range(1, STEPS + 1), desc="Training")
    for step in progress:
        x, y = get_batch(train_data, device)
        _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        train_loss = loss.item()
        history["train"].append({"step": step, "loss": train_loss})
        progress.set_postfix(loss=f"{train_loss:.4f}")

        if step % EVAL_INTERVAL == 0:
            val_loss = evaluate(model, val_data, device)
            history["validation"].append(
                {"step": step, "loss": val_loss}
            )
            progress.write(
                f"step {step}: "
                f"train loss {train_loss:.4f}, "
                f"val loss {val_loss:.4f}"
            )

    val_loss = evaluate(model, val_data, device)
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    checkpoint_path = CHECKPOINT_DIR / f"{args.tokenizer}.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "config": config.__dict__,
            "tokenizer": args.tokenizer,
            "val_loss": val_loss,
        },
        checkpoint_path,
    )
    history_path = CHECKPOINT_DIR / f"{args.tokenizer}_history.json"
    with history_path.open("w", encoding="utf-8") as history_file:
        json.dump(history, history_file, indent=2)

    print(f"saved checkpoint: {checkpoint_path}")
    print(f"saved training history: {history_path}")


if __name__ == "__main__":
    main()
