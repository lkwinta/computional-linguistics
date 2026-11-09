
import json
import argparse

from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from models.transformer import LanguageTransformer
from custom_tokenizers import SentencePieceBPETokenizer


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "encoded_dataset"
TOKENIZER_PATH = BASE_DIR / "output" / "sp_bpe.model"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"

BLOCK_SIZE = 256
BATCH_SIZE = 16
STEPS = 20000
EVAL_INTERVAL = 100
EVAL_BATCHES = 20
LEARNING_RATE = 3e-4
GRAD_CLIP = 1.0

EMBEDDING_DIM = 384
NUM_HEADS = 6
NUM_LAYERS = 6


def get_batch(data, device):
    starts = torch.randint(
        0, len(data) - BLOCK_SIZE, (BATCH_SIZE,)
    )

    # B fragmentów po T+1 tokenów
    sequences = np.stack([
        data[i:i + BLOCK_SIZE + 1]
        for i in starts.tolist()
    ]).astype(np.int64)

    batch = torch.from_numpy(sequences).to(device)

    x = batch[:, :-1]
    y = batch[:, 1:]

    return x, y


@torch.no_grad()
def evaluate(model, data, device):
    model.eval()
    losses = []

    for _ in range(EVAL_BATCHES):
        x, y = get_batch(data, device)

        logits = model(x)

        loss = F.cross_entropy(
            logits.flatten(0, 1),
            y.flatten()
        )

        losses.append(loss.item())

    model.train()
    return sum(losses) / len(losses)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--use-rope",
        action="store_true",
        help="Use RoPE instead of learned positional embeddings"
    )
    parser.add_argument(
        "--feed-forward",
        choices=["mlp", "swiglu"],
        default="mlp",
        help="Type of feed-forward network to use"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=STEPS,
        help="Number of training steps"
    )

    args = parser.parse_args()

    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Tokenizer
    tokenizer = SentencePieceBPETokenizer()
    tokenizer.load(str(TOKENIZER_PATH))

    # Encoded datasets
    train_data = np.memmap(
        DATA_DIR / "train.bin",
        dtype=np.uint16,
        mode="r"
    )

    val_data = np.memmap(
        DATA_DIR / "val.bin",
        dtype=np.uint16,
        mode="r"
    )

    if min(len(train_data), len(val_data)) <= BLOCK_SIZE:
        raise ValueError("Dataset is shorter than block_size + 1")

    # Similar FFN parameter budget for MLP and SwiGLU
    ff_hidden_dim = (
        4 * EMBEDDING_DIM
        if args.feed_forward == "mlp"
        else 8 * EMBEDDING_DIM // 3
    )

    config = {
        "vocab_size": tokenizer.vocab_size,
        "dim": EMBEDDING_DIM,
        "num_heads": NUM_HEADS,
        "ff_hidden_dim": ff_hidden_dim,
        "num_layers": NUM_LAYERS,
        "block_size": BLOCK_SIZE,
        "feed_forward": args.feed_forward,
        "use_rope": args.use_rope,
        "weight_tying": True,
    }

    model = LanguageTransformer(**config)
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.1
    )

    num_params = sum(p.numel() for p in model.parameters())

    print(f"device: {device}")
    print(f"parameters: {num_params:,} ({num_params / 1e6:.2f}M)")
    print(f"use_rope: {args.use_rope}")
    print(f"feed_forward: {args.feed_forward}")
    print(f"train tokens: {len(train_data):,}")
    print(f"validation tokens: {len(val_data):,}")

    history = {"train": [], "validation": []}

    progress = tqdm(
        range(1, args.steps + 1),
        desc="Training Transformer"
    )

    if device == "cuda":
        torch.cuda.synchronize()

    start_time = perf_counter()

    for step in progress:
        model.train()

        x, y = get_batch(train_data, device)

        logits = model(x)

        loss = F.cross_entropy(
            logits.flatten(0, 1),
            y.flatten()
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRAD_CLIP
        )

        optimizer.step()

        train_loss = loss.item()

        history["train"].append({
            "step": step,
            "loss": train_loss
        })

        progress.set_postfix(loss=f"{train_loss:.4f}")

        if step % EVAL_INTERVAL == 0 or step == args.steps:
            val_loss = evaluate(model, val_data, device)

            history["validation"].append({
                "step": step,
                "loss": val_loss
            })

            progress.write(
                f"step {step}: "
                f"train {train_loss:.4f}, "
                f"val {val_loss:.4f}"
            )

    if device == "cuda":
        torch.cuda.synchronize()

    history["training_time_seconds"] = (
        perf_counter() - start_time
    )

    print(
        f"Training time (including validation): "
        f"{history['training_time_seconds']:.2f} s"
    )

    # Save checkpoint and history
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    position_type = "rope" if args.use_rope else "learned"
    name = f"transformer_{position_type}_{args.feed_forward}"

    torch.save(
        {
            "model": model.state_dict(),
            "config": config,
            "model_type": "transformer",
            "tokenizer": "sentencepiece",
            "tokenizer_model": str(TOKENIZER_PATH),
            "block_size": BLOCK_SIZE,
            "val_loss": val_loss,
            "step": args.steps,
        },
        CHECKPOINT_DIR / f"{name}.pt"
    )

    (CHECKPOINT_DIR / f"{name}_history.json").write_text(
        json.dumps(history, indent=2),
        encoding="utf-8"
    )

    print(f"Saved model and history to {CHECKPOINT_DIR}")


if __name__ == "__main__":
    main()
