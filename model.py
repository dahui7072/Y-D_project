import os
import json
import math
import random
import numpy as np
from glob import glob
from typing import List
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset


class CFG:
    IMG_SIZE = 512
    BATCH_SIZE = 8
    EPOCHS = 2
    LR = 1e-4
    NUM_WORKERS = 2
    SEED = 42
    DIM = 256
    NO_PRETRAIN = False


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def find_jsons(json_dir: str) -> List[str]:
    return sorted(glob(os.path.join(json_dir, "*.json")))


def read_json_safe(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return None


def simple_tokenize(s: str) -> List[str]:
    if not s:
        return []
    for ch in [",", "(", ")", ":", "?", "!", ";"]:
        s = s.replace(ch, " ")
    return [t for t in s.split() if t]


class Vocab:
    def __init__(self, min_freq=1):
        self.min_freq = min_freq
        self.freq = {}
        self.itos = ["<pad>", "<unk>"]
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    def build(self, texts):
        for s in texts:
            for tok in simple_tokenize(s):
                self.freq[tok] = self.freq.get(tok, 0) + 1

        for tok, f in self.freq.items():
            if f >= self.min_freq and tok not in self.stoi:
                self.stoi[tok] = len(self.itos)
                self.itos.append(tok)

    def encode(self, s: str, max_len=40):
        toks = simple_tokenize(s)[:max_len]
        if not toks:
            return [1]
        return [self.stoi.get(t, 1) for t in toks]


class UniDSet(Dataset):
    def __init__(self, json_dir, jpg_dir, vocab=None, build_vocab=False, test_mode=False):

        json_files = find_jsons(json_dir)
        self.items = []
        self.test_mode = test_mode

        for jf in json_files:
            data = read_json_safe(jf)
            if data is None:
                continue

            if test_mode:
                if "image_id" not in data or "query" not in data:
                    continue
                bbox = [0, 0, 1, 1]
            else:
                if "image_id" not in data or "query" not in data or "bbox" not in data:
                    continue
                bbox = data["bbox"]
                x, y, w, h = bbox
                if w <= 0 or h <= 0:
                    continue

            img_path = os.path.join(jpg_dir, data["image_id"])
            if not os.path.exists(img_path):
                continue

            self.items.append({
                "json": jf,
                "img": img_path,
                "query": data["query"],
                "class_name": data.get("class_name", "unknown"),
                "bbox": bbox,
                "query_id": os.path.splitext(os.path.basename(jf))[0]
            })

        if len(self.items) == 0:
            raise RuntimeError("ERROR: No samples found in dataset folder.")

        self.vocab = vocab if vocab else Vocab()
        if build_vocab:
            self.vocab.build([it["query"] for it in self.items])

        from torchvision import transforms
        self.tf = transforms.Compose([
            transforms.Resize((CFG.IMG_SIZE, CFG.IMG_SIZE)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]

        img = Image.open(it["img"]).convert("RGB")
        W, H = img.size
        img_t = self.tf(img)

        ids = torch.tensor(self.vocab.encode(it["query"]), dtype=torch.long)
        length = torch.tensor(len(ids), dtype=torch.long)

        x, y, w, h = it["bbox"]
        cx = (x + w/2) / W
        cy = (y + h/2) / H
        nw = w / W
        nh = h / H

        return {
            "image": img_t,
            "query_ids": ids,
            "length": length,
            "target": torch.tensor([cx, cy, nw, nh], dtype=torch.float),
            "meta": it
        }


def collate_fn(batch):
    B = len(batch)
    max_len = max(len(b["query_ids"]) for b in batch)
    ids = torch.zeros(B, max_len, dtype=torch.long)
    lens = torch.zeros(B, dtype=torch.long)
    imgs = torch.stack([b["image"] for b in batch])
    targets = torch.stack([b["target"] for b in batch])
    meta = [b["meta"] for b in batch]

    for i, b in enumerate(batch):
        l = len(b["query_ids"])
        ids[i, :l] = b["query_ids"]
        lens[i] = l

    return imgs, ids, lens, targets, meta


class TextEncoder(nn.Module):
    def __init__(self, vocab_size, dim=CFG.DIM):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.gru = nn.GRU(dim, dim, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(dim * 2, dim)

    def forward(self, tokens, lengths):
        x = self.emb(tokens)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h = self.gru(packed)
        h = torch.cat([h[-2], h[-1]], dim=-1)
        return self.proj(h)


class ImageEncoder(nn.Module):
    def __init__(self, dim=CFG.DIM):
        super().__init__()
        from torchvision.models import resnet18, ResNet18_Weights
        m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        layers = list(m.children())[:-2]
        self.backbone = nn.Sequential(*layers)
        self.proj = nn.Conv2d(512, dim, 1)

    def forward(self, x):
        f = self.backbone(x)
        return self.proj(f)


class CrossAttentionBBox(nn.Module):
    def __init__(self, dim=CFG.DIM):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.kv = nn.Conv2d(dim, dim * 2, 1)
        self.head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 4)
        )

    def forward(self, q, fmap):
        B, D, H, W = fmap.shape

        q = self.q_proj(q).unsqueeze(1)
        kv = self.kv(fmap)
        K, V = kv.chunk(2, dim=1)

        Kf = K.flatten(2).transpose(1, 2)
        Vf = V.flatten(2).transpose(1, 2)

        attn = torch.matmul(q, Kf.transpose(1, 2)) / math.sqrt(D)
        attn = torch.softmax(attn, dim=-1)

        ctx = torch.matmul(attn, Vf).squeeze(1)
        return torch.sigmoid(self.head(ctx))


class CrossAttnVLM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.txt = TextEncoder(vocab_size)
        self.img = ImageEncoder()
        self.head = CrossAttentionBBox()

    def forward(self, imgs, ids, lens):
        q = self.txt(ids, lens)
        fmap = self.img(imgs)
        return self.head(q, fmap)
