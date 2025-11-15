import os
import csv
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import CFG, seed_everything, UniDSet, collate_fn, CrossAttnVLM



def clean_text(s: str) -> str:
    if s is None:
        return ""
    s = s.replace("\n", " ").replace("\r", " ")
    s = s.replace(",", " ")      
    s = s.replace('"', ' ')  
    s = s.replace("'", " ")
    s = s.strip()
    return s




def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_dir", type=str, required=True)
    parser.add_argument("--jpg_dir", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--save_csv", type=str, required=True)
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

   
    test_ds = UniDSet(
        args.json_dir,
        args.jpg_dir,
        vocab=None,
        build_vocab=False,
        test_mode=True
    )

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

        # float clean
        pred = [float(f"{v:.6f}") for v in pred]

        item = meta[0]
        qid = item["query_id"]
        qtext = clean_text(item["query"])

        results.append([qid, qtext] + pred)

   
    with open(args.save_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "query_text", "pred_x", "pred_y", "pred_w", "pred_h"])
        writer.writerows(results)
        writer = csv.writer(f, quoting=csv.QUOTE_NONE, escapechar=' ')


    print(f"\nSaved submission file: {args.save_csv}")


if __name__ == "__main__":
    main()
