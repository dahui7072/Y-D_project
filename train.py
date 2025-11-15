import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import CFG, seed_everything, UniDSet, collate_fn, CrossAttnVLM


# ---------------------------
#  Train Loop
# ---------------------------

def train_one_epoch(model, loader, optim, device):
    model.train()
    total_loss = 0

    for imgs, ids, lens, targets, meta in loader:
        imgs = imgs.to(device)
        ids = ids.to(device)
        lens = lens.to(device)
        targets = targets.to(device)

        optim.zero_grad()
        preds = model(imgs, ids, lens)

        loss = nn.functional.l1_loss(preds, targets)
        loss.backward()
        optim.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# ---------------------------
#  Main
# ---------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_dir", type=str, required=True)
    parser.add_argument("--jpg_dir", type=str, required=True)
    parser.add_argument("--save_ckpt", type=str, default="./outputs/ckpt/cross_attn_vlm.pth")
    args = parser.parse_args()

    # Device (CUDA → MPS → CPU)
    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print("Using device:", device)

    seed_everything(CFG.SEED)

    print("Loading dataset...")
    train_ds = UniDSet(args.json_dir, args.jpg_dir, vocab=None, build_vocab=True)

    vocab = train_ds.vocab
    vocab_size = len(vocab.itos)

    loader = DataLoader(
        train_ds,
        batch_size=CFG.BATCH_SIZE,
        shuffle=True,
        num_workers=CFG.NUM_WORKERS,
        collate_fn=collate_fn
    )

    # Model
    model = CrossAttnVLM(vocab_size).to(device)

    # Optimizer
    optim = torch.optim.Adam(model.parameters(), lr=CFG.LR)

    # Train
    for epoch in range(1, CFG.EPOCHS + 1):
        loss = train_one_epoch(model, loader, optim, device)
        print(f"[Epoch {epoch}/{CFG.EPOCHS}] Loss: {loss:.4f}")

    # Save
    os.makedirs(os.path.dirname(args.save_ckpt), exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "vocab": vocab.itos
    }, args.save_ckpt)

    print("\nTraining completed.")
    print(f"Model saved at: {args.save_ckpt}")


if __name__ == "__main__":
    main()
