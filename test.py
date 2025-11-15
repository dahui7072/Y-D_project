import os
import csv
import random
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import CFG, seed_everything, UniDSet, collate_fn, CrossAttnVLM


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_dir", type=str, required=True)
    parser.add_argument("--jpg_dir", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--save_csv", type=str, required=True)
    parser.add_argument("--sample_size", type=int, default=5000)
    args = parser.parse_args()

    seed_everything()

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("Using device:", device)
    print("Loading dataset...")

    # Test dataset
    test_ds = UniDSet(
        args.json_dir,
        args.jpg_dir,
        vocab=None,
        build_vocab=False,
        test_mode=True
    )

    TOTAL = len(test_ds.items)
    SAMPLE = min(args.sample_size, TOTAL)

    print(f"Random sampled {SAMPLE} items (from {TOTAL}).")

    # random sampling
    indices = random.sample(range(TOTAL), SAMPLE)
    test_ds.items = [test_ds.items[i] for i in indices]

    loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    # Model Load
    ckpt = torch.load(args.ckpt, map_location="cpu")
    vocab_itos = ckpt["vocab"]
    vocab_size = len(vocab_itos)

    model = CrossAttnVLM(vocab_size).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    print("\nStart Inference...\n")

    results = []

    for imgs, ids, lens, targets, meta in tqdm(loader, desc="Inference", ncols=100):
        imgs = imgs.to(device)
        ids = ids.to(device)
        lens = lens.to(device)

        with torch.no_grad():
            pred = model(imgs, ids, lens)[0].cpu().tolist()

        item = meta[0]
        qid = item["query_id"]
        qtext = item["query"]

        results.append([qid, qtext] + pred)

    # Save CSV (submission format)
    with open(args.save_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "query_text", "pred_x", "pred_y", "pred_w", "pred_h"])
        writer.writerows(results)

    print(f"\nSaved submission file: {args.save_csv}")


if __name__ == "__main__":
    main()
