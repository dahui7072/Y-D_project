import os
import torch
from torch.utils.data import DataLoader

from model import CFG, seed_everything, UniDSet, collate_fn, CrossAttnVLM


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_dir", type=str, required=True)
    parser.add_argument("--jpg_dir", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    args = parser.parse_args()

    seed_everything()

    print("Loading dataset...")
    test_ds = UniDSet(args.json_dir, args.jpg_dir, vocab=None, build_vocab=False)

    loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    ckpt = torch.load(args.ckpt, map_location="cpu")
    vocab_itos = ckpt["vocab"]

    vocab_size = len(vocab_itos)
    model = CrossAttnVLM(vocab_size).cuda()
    model.load_state_dict(ckpt["model"])
    model.eval()

    print("\nStart Inference...\n")

    for imgs, ids, lens, targets, meta in loader:
        imgs = imgs.cuda()
        ids = ids.cuda()
        lens = lens.cuda()

        with torch.no_grad():
            pred = model(imgs, ids, lens)[0].cpu().tolist()

        item = meta[0]
        print("--------------------------------------------------")
        print(f"Query: {item['query']}")
        print(f"Class: {item['class_name']}")
        print(f"Prediction (cx, cy, w, h): {pred}")
        print("--------------------------------------------------\n")


if __name__ == "__main__":
    main()
