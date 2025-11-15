import os
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import CFG, seed_everything, UniDSet, collate_fn, CrossAttnVLM


def train_one_epoch(model, loader, optim, device, epoch, total_epochs):
    model.train()
    total_loss = 0
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs}", ncols=120)

    for imgs, ids, lens, targets, meta in pbar:
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
        pbar.set_postfix({
            "batch_loss": loss.item(),
            "avg_loss": total_loss / (len(pbar) + 1)
        })

    return total_loss / len(loader)


def main():
    CFG.EPOCHS = 2
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_dir", type=str, required=True)
    parser.add_argument("--jpg_dir", type=str, required=True)
    parser.add_argument("--save_ckpt", type=str, default="./outputs/ckpt/cross_attn_vlm.pth")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("Using device:", device)

    seed_everything(CFG.SEED)

    train_ds = UniDSet(
        args.json_dir,
        args.jpg_dir,
        vocab=None,
        build_vocab=True,
        test_mode=False
    )

    if len(train_ds.items) > 5000:
        random.shuffle(train_ds.items)
        train_ds.items = train_ds.items[:5000]
        print(f"Random sampled 5000 items.")
    else:
        print(f"Dataset has only {len(train_ds.items)} samples → using all.")

    vocab = train_ds.vocab
    vocab_size = len(vocab.itos)

    loader = DataLoader(
        train_ds,
        batch_size=CFG.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )

    model = CrossAttnVLM(vocab_size).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=CFG.LR)

    for epoch in range(1, CFG.EPOCHS + 1):
        loss = train_one_epoch(model, loader, optim, device, epoch, CFG.EPOCHS)
        print(f"[Epoch {epoch}/{CFG.EPOCHS}] Loss: {loss:.4f}")

    os.makedirs(os.path.dirname(args.save_ckpt), exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "vocab": vocab.itos
    }, args.save_ckpt)

    print("\nTraining completed.")
    print(f"Model saved at: {args.save_ckpt}")


if __name__ == "__main__":
    main()
