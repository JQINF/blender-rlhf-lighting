"""train_rm.py — reward model training: frozen image encoder + scoring head on pairwise preferences.

Architecture: frozen image encoder (--encoder: clip_b16 = local open_clip ViT-B/16 with weights from
data/weights/open_clip_model.safetensors, loaded offline; pickscore = HF PickScore CLIP-H /
clip_l336 = CLIP-L/14@336 / siglip_so400m = SigLIP SO400M@384,
all HF-family weights come from the HF cache — pull the repo once on a new machine via
export HF_ENDPOINT=https://hf-mirror.com, then pass HF_HUB_OFFLINE=1) +
scoring head Linear(emb,128)+ReLU+Dropout+Linear(128,1) (~66k params,
the only trained part; --hidden 0 degrades to a linear probe); weighted Bradley-Terry pairwise loss
+ tie supervision (s_a - s_b)^2; output is z-scored (mean/std stored with the checkpoint).

Data interface:
  --pairs / --labels  silver labels (joined by pair_id; silver ties are dropped — only 6 rows, not worth a class)
  --user-labels       user-label jsonl {"pair_id","a","b","winner":"a"|"b"|"tie"} (blind-label export;
                      non-tie rows enter BT with --user-weight, ties enter tie supervision at the same weight; user labels are train-only)
Approach: dedup images, run the frozen encoder once, store one embedding each for forward + horizontal flip
(flip augmentation off by default — measured to hurt; enable with --flip); afterwards only the head trains each epoch, seconds-level.

Internal eval: silver labels split 90/10 by pair into train/val for early stopping and reference
(same-(sku,view) pairs can straddle the split, so this is optimistically biased);
the real acceptance gate is the user's own blind labeling (see make_blind200.py / score_blind_eval.py).
--user-cv: 4-fold pre-check — train on the silver+user mix, hold out 1/4 of user non-tie labels per fold, report mean+-std
to estimate whether user taste is learnable (below 0.70, don't spend another blind-label round).

--smoke: skips real data; synthesizes a learnable preference (images shared across pairs, so val images appeared in train —
tests true generalization, not memorization), runs the loop, asserts.

Usage: .venv/bin/python scripts/train/train_rm.py [--user-labels data/pairs/user_labels.jsonl --user-cv]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]

BACKBONE = "ViT-B-16"
EMB_DIM = 512  # CLIP ViT-B/16 image embedding dim
DEFAULT_WEIGHTS = ROOT / "data/weights/open_clip_model.safetensors"

    # Encoder registry: selected via --encoder; emb_dim sets the scoring-head input width.
    # pickscore / clip_l336 / siglip_so400m / siglip2_so400m / dinov2_l all load through the
    # transformers pipeline with weights from the HF cache (never embedded in ckpts).
    # Set HF_HUB_OFFLINE=1 (and HF_ENDPOINT=https://hf-mirror.com if needed) for offline runs.
ENCODERS = {
    "clip_b16": {"emb_dim": 512},
    "pickscore": {"emb_dim": 1024, "hf_repo": "yuvalkirstain/PickScore_v1",
                  "proc_repo": "openai/clip-vit-large-patch14"},
    "clip_l336": {"emb_dim": 768, "hf_repo": "openai/clip-vit-large-patch14-336"},
    "siglip_so400m": {"emb_dim": 1152, "hf_repo": "google/siglip-so400m-patch14-384",
                      "kind": "siglip"},
    "siglip2_so400m": {"emb_dim": 1152, "hf_repo": "google/siglip2-so400m-patch14-384",
                       "kind": "siglip"},
    "dinov2_l": {"emb_dim": 1024, "hf_repo": "facebook/dinov2-large", "kind": "dinov2"},
    "dinov2_l518": {"emb_dim": 1024, "hf_repo": "facebook/dinov2-large", "kind": "dinov2",
                    "img_size": 518},
    "dinov2_siglip2": {"emb_dim": 2176, "hf_repo": "facebook/dinov2-large", "kind": "combo"},
}


class ScoreHead(nn.Module):
    """Scoring head Linear(512,h)+ReLU+Dropout+Linear(h,1); h=0 -> linear probe Linear(512,1)."""

    def __init__(self, emb_dim=EMB_DIM, hidden=128, p_drop=0.2):
        super().__init__()
        if hidden > 0:
            self.net = nn.Sequential(
                nn.Linear(emb_dim, hidden), nn.ReLU(), nn.Dropout(p_drop), nn.Linear(hidden, 1))
        else:
            self.net = nn.Linear(emb_dim, 1)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def load_pairs(pairs_path, labels_path, weight=1.0):
    """pairs.jsonl join labels.jsonl -> [{"pair_id","kind","a","b","winner","w"}]; silver ties dropped.
    When weight != 1 (--silver-weight, e.g. 0 = silver only kept as val domain, out of training gradients),
    the per-row w becomes weight."""
    pairs = [json.loads(l) for l in Path(pairs_path).read_text().splitlines() if l.strip()]
    labels = {json.loads(l)["pair_id"]: json.loads(l) for l in Path(labels_path).read_text().splitlines() if l.strip()}
    rows, n_tie, missing = [], 0, []
    for p in pairs:
        lab = labels.get(p["pair_id"])
        if lab is None:
            missing.append(p["pair_id"])
            continue
        w = lab["winner"]
        if w == "tie":
            n_tie += 1
            continue
        assert w in ("a", "b"), f"pair {p['pair_id']} has invalid winner: {w}"
        rows.append({"pair_id": p["pair_id"], "kind": p["kind"], "a": p["a"], "b": p["b"],
                     "winner": w, "w": float(weight)})
    if missing:
        sys.exit(f"error: {len(missing)} pairs lack labels (first 5: {missing[:5]}) — label them or check labels.jsonl")
    if not rows:
        sys.exit("error: no valid pairs after joining pairs/labels")
    print(f"silver pairs {len(pairs)}: dropped tie {n_tie}, valid {len(rows)}"
          f"（hard {sum(1 for r in rows if r['kind'] == 'hard')} / anchor {sum(1 for r in rows if r['kind'] == 'anchor')}）")
    return rows


def load_user_labels(path, weight):
    """User blind-label jsonl -> (non-tie rows with w=weight kind='user', tie rows with w=weight); paths validated relative to project root.
    Optional per-row "w" field overrides the uniform weight (recipe experiments, e.g. downweighting a sweep batch x1)."""
    rows, ties = [], []
    for l in Path(path).read_text().splitlines():
        if not l.strip():
            continue
        d = json.loads(l)
        assert d["winner"] in ("a", "b", "tie"), f"user pair {d.get('pair_id')} has invalid winner: {d['winner']}"
        for key in ("a", "b"):
            assert (ROOT / d[key]).is_file(), f"user pair {d.get('pair_id')} image not found: {d[key]}"
        row = {"pair_id": d["pair_id"], "kind": "user", "a": d["a"], "b": d["b"],
               "winner": d["winner"], "w": float(d.get("w", weight))}
        (ties if d["winner"] == "tie" else rows).append(row)
    n_override = sum(1 for r in rows + ties if r["w"] != weight)
    print(f"human labels: {len(rows)} non-tie (weight {weight}) + {len(ties)} tie rows into tie supervision"
          + (f", {n_override} per-row w overrides" if n_override else ""))
    return rows, ties


def load_auto_labels(path, weight):
    """Exploration-arm auto anchor-pair jsonl -> rows (kind kept as-is, e.g. anchor_low, for per-kind diagnostics; no ties)."""
    rows = []
    for l in Path(path).read_text().splitlines():
        if not l.strip():
            continue
        d = json.loads(l)
        assert d["winner"] in ("a", "b"), f"auto pair {d.get('pair_id')} has invalid winner: {d['winner']}"
        for key in ("a", "b"):
            assert (ROOT / d[key]).is_file(), f"auto pair {d.get('pair_id')} image not found: {d[key]}"
        rows.append({"pair_id": d["pair_id"], "kind": d.get("kind", "auto"),
                     "a": d["a"], "b": d["b"], "winner": d["winner"], "w": weight})
    print(f"auto anchor pairs: {len(rows)} (weight {weight}, train-only, excluded from val)")
    return rows


def encode_images(img_paths, device, batch_size=64, encoder="clip_b16"):
    """Encode deduped images with the frozen backbone, one embedding each for forward + horizontal flip -> dict[path] = (2,emb_dim) unit-norm numpy.
    Returns (feats, backbone_state_dict) — clip_b16's backbone_state is stored with the checkpoint (self-contained at inference);
    HF-family encoders are too large to embed, so None is returned and inference rebuilds from the HF cache per the registry."""
    from PIL import Image

    if encoder == "clip_b16":
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(BACKBONE, pretrained=str(DEFAULT_WEIGHTS))
        backbone_state = model.visual.state_dict()

        def encode(xs):
            return model.encode_image(xs)

        def load_batch(paths):
            return torch.stack([preprocess(Image.open(ROOT / p).convert("RGB")) for p in paths])

    elif encoder in ENCODERS and "hf_repo" in ENCODERS[encoder]:
        cfg = ENCODERS[encoder]
        if cfg.get("kind") == "siglip":
            from transformers import AutoProcessor, SiglipModel
            model = SiglipModel.from_pretrained(cfg["hf_repo"])
            processor = AutoProcessor.from_pretrained(cfg["hf_repo"])

            def encode(xs):
                out = model.get_image_features(pixel_values=xs)
                return getattr(out, "pooler_output", out)  # transformers v5+ returns BaseModelOutputWithPooling
        elif cfg.get("kind") == "dinov2":
            from transformers import AutoImageProcessor, AutoModel
            model = AutoModel.from_pretrained(cfg["hf_repo"])
            sz = cfg.get("img_size")
            processor = AutoImageProcessor.from_pretrained(
                cfg["hf_repo"],
                **({"size": {"shortest_edge": sz},
                    "crop_size": {"height": sz, "width": sz}} if sz else {}))

            def encode(xs):
                return model(pixel_values=xs).last_hidden_state[:, 0]  # CLS token
        else:
            from transformers import CLIPModel, CLIPProcessor
            model = CLIPModel.from_pretrained(cfg["hf_repo"])
            processor = CLIPProcessor.from_pretrained(cfg.get("proc_repo", cfg["hf_repo"]))

            def encode(xs):
                out = model.get_image_features(pixel_values=xs)
                return getattr(out, "pooler_output", out)  # transformers v5+ returns BaseModelOutputWithPooling
        backbone_state = None

        def load_batch(paths):
            return processor(images=[Image.open(ROOT / p).convert("RGB") for p in paths],
                             return_tensors="pt")["pixel_values"]
    else:
        sys.exit(f"error: unknown encoder {encoder} (options: {list(ENCODERS)})")

    model.requires_grad_(False)
    model = model.eval().to(device)

    feats = {}
    paths = list(img_paths)
    for i in range(0, len(paths), batch_size):
        batch = paths[i:i + batch_size]
        xs = load_batch(batch).to(device)
        with torch.no_grad():
            out = []
            for x in (xs, torch.flip(xs, dims=[-1])):  # forward / horizontal flip
                f = encode(x)
                out.append(f / f.norm(dim=-1, keepdim=True))
            f = torch.stack(out, dim=1)  # (B,2,emb_dim)
        for p, v in zip(batch, f.cpu().numpy().astype(np.float32)):
            feats[p] = v
        if (i // batch_size) % 10 == 0:
            print(f"  {encoder} encoding {min(i + batch_size, len(paths))}/{len(paths)}", flush=True)
    del model
    torch.cuda.empty_cache()
    return feats, backbone_state


def pair_tensors(rows, feats, flip_rng=None):
    """rows + embedding cache -> (Xa, Xb, y, wt) torch tensors; y=1 means a wins; wt = per-sample loss weight.
    If flip_rng is not None, each image independently picks forward/flipped (train augmentation); None = all forward (eval)."""
    def pick(path):
        v = feats[path]
        if flip_rng is None:
            return v[0]
        return v[int(flip_rng.integers(2))]
    Xa = torch.as_tensor(np.stack([pick(r["a"]) for r in rows]))
    Xb = torch.as_tensor(np.stack([pick(r["b"]) for r in rows]))
    y = torch.as_tensor([1.0 if r["winner"] == "a" else 0.0 for r in rows])
    wt = torch.as_tensor([r.get("w", 1.0) for r in rows])
    return Xa, Xb, y, wt


def bt_loss(head, xa, xb, y, wt):
    """Weighted Bradley-Terry: y=1 -> softplus(-(sa-sb)); y=0 -> softplus(-(sb-sa)), weighted-mean by wt."""
    sa, sb = head(xa), head(xb)
    s_win = y * sa + (1 - y) * sb
    s_lose = (1 - y) * sa + y * sb
    per = F.softplus(-(s_win - s_lose))
    return (per * wt).sum() / wt.sum()


def tie_loss(head, xa, xb, wt):
    """Tie supervision: both equally good -> push the score diff to 0, weighted (sa-sb)^2."""
    sa, sb = head(xa), head(xb)
    return (((sa - sb) ** 2) * wt).sum() / wt.sum()


def pair_acc(head, rows, feats, device):
    xa, xb, y, _wt = pair_tensors(rows, feats)
    with torch.no_grad():
        sa, sb = head(xa.to(device)), head(xb.to(device))
    return ((sa > sb).float().cpu() == y).float().mean().item()


def make_smoke_data(n=512, pool=64, seed=0):
    """Synthetic learnable preference: a pool of shared "images", true score = fixed weighted sum of the first 8 embedding dims,
    winner by score + small noise. Images are reused across pairs, so val images appeared in train — tests true generalization, not memorization; always learnable."""
    rng = np.random.default_rng(seed)
    emb = rng.standard_normal((pool, EMB_DIM)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    w = rng.standard_normal(8).astype(np.float32)
    score = emb[:, :8] @ w * 20.0  # amplify the signal: unit-norm squeezes scores to ~0.1 magnitude; without amplification label noise explodes
    rows, feats = [], {}
    for i in range(n):
        a, b = rng.choice(pool, size=2, replace=False)
        noisy = score[[a, b]] + rng.standard_normal(2) * 0.1
        rows.append({"pair_id": i, "kind": "hard" if i % 5 else "anchor",
                     "a": f"img{a}", "b": f"img{b}", "winner": "a" if noisy[0] > noisy[1] else "b",
                     "w": 1.0})
    for i in range(pool):
        feats[f"img{i}"] = np.stack([emb[i], emb[i]])  # (2,512), both "orientations" identical
    return rows, feats, None


def train_once(tr, va, tie_rows, feats, args, device, tag="", va_report=None):
    """One training run: BT on tr/va split (val for early stopping), tie_rows enter tie supervision.
    va_report = read-only report set (e.g. silver val, ignored by early stopping). Returns (head, history, best)."""
    head = ScoreHead(emb_dim=args.emb_dim, hidden=args.hidden, p_drop=args.dropout).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    n = len(tr)
    flip_rng = np.random.default_rng(args.seed + 1) if args.flip else None
    has_tie = bool(tie_rows) and args.tie_coef > 0

    history = []
    best = {"val": -1.0, "epoch": -1, "state": None}
    for epoch in range(args.epochs):
        head.train()
        Xa, Xb, y, wt = pair_tensors(tr, feats, flip_rng)
        Xa, Xb, y, wt = Xa.to(device), Xb.to(device), y.to(device), wt.to(device)
        if has_tie:
            Ta, Tb, _ty, Tw = pair_tensors(tie_rows, feats, flip_rng)
            Ta, Tb, Tw = Ta.to(device), Tb.to(device), Tw.to(device)
        perm = torch.randperm(n, device=device)
        tot, steps = 0.0, 0
        for s in range(0, n - args.batch_size + 1, args.batch_size):
            ii = perm[s:s + args.batch_size]
            loss = bt_loss(head, Xa[ii], Xb[ii], y[ii], wt[ii])
            if has_tie:
                loss = loss + args.tie_coef * tie_loss(head, Ta, Tb, Tw)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
            steps += 1
        history.append(tot / steps)
        head.eval()
        acc_va = pair_acc(head, va, feats, device) if va else 0.0
        if not va or (epoch + 1 >= args.min_epochs and acc_va > best["val"]):
            best.update(val=acc_va, epoch=epoch + 1,
                        state={k: v.detach().cpu().clone() for k, v in head.state_dict().items()})
        if (epoch + 1) % 25 == 0 or epoch == 0:
            acc_tr = pair_acc(head, tr, feats, device)
            extra = f"  val_silver(read-only)={pair_acc(head, va_report, feats, device):.3f}" if va_report else ""
            print(f"{tag}epoch {epoch + 1:3d}/{args.epochs}  loss={history[-1]:.4f}  "
                  f"train_acc={acc_tr:.3f}  val_acc={acc_va:.3f}  (best {best['val']:.3f} @ep{best['epoch']}){extra}",
                  flush=True)

    if args.early_stop and best["state"] is not None:
        head.load_state_dict(best["state"])
        print(f"{tag}early stop: rolled back to best val epoch {best['epoch']} (val_acc={best['val']:.3f})")
    head.eval()
    return head, history, best


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pairs", type=Path, default=ROOT / "data/pairs/pairs.jsonl", help="silver-label pair list jsonl")
    p.add_argument("--labels", type=Path, default=ROOT / "data/pairs/labels.jsonl", help="silver-label annotations jsonl")
    p.add_argument("--user-labels", type=Path, default=None, help="user-label jsonl (blind-label export, mixed retrain)")
    p.add_argument("--auto-labels", type=Path, default=None,
                   help="exploration-arm auto anchor-pair jsonl (gen_explore low-z anchors, winner embedded, train-only not val)")
    p.add_argument("--auto-weight", type=float, default=1.0, help="auto anchor-pair weight")
    p.add_argument("--encoder", choices=list(ENCODERS), default="clip_b16",
                   help="image encoder: clip_b16 = local open_clip ViT-B-16; pickscore = HF PickScore; "
                        "clip_l336 = CLIP-L/14@336; siglip_so400m = SigLIP SO400M@384 (the last three's weights come from the HF cache)")
    p.add_argument("--user-weight", type=float, default=8.0, help="user-label weight in BT/tie loss (silver = 1)")
    p.add_argument("--silver-weight", type=float, default=1.0,
                   help="silver-label row weight (0 = silver only kept as val domain, out of training gradients; pure-human-label recipe)")
    p.add_argument("--user-cv", action="store_true", help="4-fold pre-check: hold out 1/4 of user labels to test taste learnability, no ckpt saved")
    p.add_argument("--tie-coef", type=float, default=1.0, help="tie-supervision coefficient (0 = off)")
    p.add_argument("--out", type=Path, default=None, help="checkpoint path (default logs/ckpt/rm[_smoke].pt)")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3, help="scoring-head learning rate (backbone frozen)")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--hidden", type=int, default=128, help="scoring-head hidden dim; 0 = linear probe")
    p.add_argument("--val-frac", type=float, default=0.1, help="silver internal val ratio (random split by pair)")
    p.add_argument("--human-val-frac", type=float, default=0.0,
                   help="if > 0, carve a fixed fraction of user labels as the early-stop val (silver val degrades to "
                        "read-only report) — silver val is semi-adversarial to the training objective under "
                        "--silver-weight 0, while human val aligns with it; the split is decoupled from --seed and stable across generations")
    p.add_argument("--human-val-ids", type=Path, default=ROOT / "data/pairs/human_val_ids.json",
                   help="human-val split file: reused if it exists, else created with fixed seed 7; "
                        "once created it is frozen — later batches (pair_ids not in the file) always enter training, never val")
    p.add_argument("--flip", action="store_true", help="enable horizontal-flip augmentation (measured to hurt, default off)")
    p.add_argument("--no-early-stop", action="store_true", help="disable early stop (default: roll back to val peak)")
    p.add_argument("--min-epochs", type=int, default=0,
                   help="minimum epochs before early stop updates best (guards against premature stops on small-val noise "
                        "and against underfitting the human val)")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--cpu", action="store_true", help="force CPU (default: use CUDA if available)")
    p.add_argument("--smoke", action="store_true", help="synthetic-data self-test: asserts loss decreases + val agreement >0.9")
    return p.parse_args()


def main():
    args = parse_args()
    args.early_stop = not args.no_early_stop
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    user_rows, tie_rows, auto_rows = [], [], []
    human_val = []  # fixed human-label val (carved out when --human-val-frac is on; early-stop/report only, never trains)
    if args.smoke:
        rows, feats, backbone_state = make_smoke_data(seed=args.seed)
        args.epochs, args.lr = 60, 3e-3
        args.emb_dim = EMB_DIM  # smoke uses fixed 512-dim synthetic embeddings regardless of --encoder
    else:
        cfg = ENCODERS[args.encoder]
        args.emb_dim = cfg["emb_dim"]
        if args.encoder == "clip_b16" and not DEFAULT_WEIGHTS.is_file():
            sys.exit(f"error: local CLIP weights missing {DEFAULT_WEIGHTS} (offline machine; download them first)")
        rows = load_pairs(args.pairs, args.labels, args.silver_weight)
        if args.user_labels:
            user_rows, tie_rows = load_user_labels(args.user_labels, args.user_weight)
            if args.human_val_frac > 0:
                split_path = args.human_val_ids
                if split_path.is_file():
                    val_ids = set(json.loads(split_path.read_text())["pair_ids"])
                    stale = val_ids - {r["pair_id"] for r in user_rows}
                    if stale:  # pair_ids of suspended batches dropped by view rebuilds: drop them from the val set too, and say so
                        print(f"human-val split: {len(stale)} pids no longer in user_labels, dropped from the val set")
                        val_ids -= stale
                else:  # create with fixed seed 7: decoupled from --seed, stable across generations; frozen once created
                    rng_hv = np.random.default_rng(7)
                    n_hv = max(1, int(round(len(user_rows) * args.human_val_frac)))
                    ids_arr = np.array([r["pair_id"] for r in user_rows])
                    val_ids = set(ids_arr[rng_hv.choice(len(user_rows), size=n_hv, replace=False)].tolist())
                    split_path.parent.mkdir(parents=True, exist_ok=True)
                    split_path.write_text(json.dumps({"seed": 7, "frac": args.human_val_frac,
                                                      "pair_ids": sorted(val_ids)}, ensure_ascii=False))
                    print(f"human-val split created (fixed seed 7, frozen afterwards): {len(val_ids)} pairs -> {split_path}")
                human_val = [r for r in user_rows if r["pair_id"] in val_ids]
                user_rows = [r for r in user_rows if r["pair_id"] not in val_ids]
            rows = rows + user_rows  # user labels are train-only (split below only slices silver); human val already carved out
        if args.auto_labels:
            auto_rows = load_auto_labels(args.auto_labels, args.auto_weight)
            rows = rows + auto_rows  # auto anchor pairs behave like user labels: train-only
        uniq = sorted({r["a"] for r in rows} | {r["b"] for r in rows} |
                      {r["a"] for r in tie_rows} | {r["b"] for r in tie_rows} |
                      {r["a"] for r in human_val} | {r["b"] for r in human_val})
        print(f"{len(uniq)} unique images; encoding with frozen {args.encoder} (device {device})…")
        feats, backbone_state = encode_images(uniq, device, encoder=args.encoder)

    # silver split into train/val by pair (user rows / auto anchor pairs never enter val); smoke has no user rows, plain split
    auto_ids = {id(r) for r in auto_rows}
    silver = [r for r in rows if r["kind"] != "user" and id(r) not in auto_ids]
    rng = np.random.default_rng(args.seed)
    idx = np.arange(len(silver))
    rng.shuffle(idx)
    n_val = max(1, int(round(len(silver) * args.val_frac)))
    va = [silver[i] for i in idx[:n_val]]
    silver_tr = [silver[i] for i in idx[n_val:]]
    tr = silver_tr + user_rows + auto_rows
    print(f"train {len(tr)} (silver {len(tr) - len(user_rows) - len(auto_rows)} + user {len(user_rows)}"
          f" + auto {len(auto_rows)}) / val {len(va)} (silver) + human val {len(human_val)} / tie {len(tie_rows)}, "
          f"head hidden={args.hidden} dropout={args.dropout} wd={args.weight_decay} flip={'on' if args.flip else 'off'}, device {device}")

    if args.user_cv:
        assert user_rows, "--user-cv requires --user-labels"
        k = 4
        order = np.random.default_rng(args.seed + 7).permutation(len(user_rows))
        folds = np.array_split(order, k)
        accs = []
        for fi, hold in enumerate(folds):
            hold_set = set(hold.tolist())
            tr_u = [r for i, r in enumerate(user_rows) if i not in hold_set]
            va_u = [r for i, r in enumerate(user_rows) if i in hold_set]
            head, _h, _b = train_once(silver_tr + tr_u, va, tie_rows, feats, args, device,
                                      tag=f"[fold {fi + 1}/{k}] ")
            acc = pair_acc(head, va_u, feats, device)
            accs.append(acc)
            print(f"[fold {fi + 1}/{k}] held-out user agreement = {acc:.3f} ({len(va_u)} pairs)", flush=True)
        print(f"\nuser-taste learnability (4-fold mean±std): {np.mean(accs):.3f} ± {np.std(accs):.3f}"
              f"  -> {'>=0.70: full retrain + new gate' if np.mean(accs) >= 0.70 else '<0.70: report first'}")
        return

    # early-stop val: use human val if present (aligned with taste; avoids the silver-val early-stop failure), silver val degrades to read-only report
    es_va = human_val if human_val else va
    head, history, best = train_once(tr, es_va, tie_rows, feats, args, device,
                                     va_report=va if human_val else None)

    final = {"train": pair_acc(head, tr, feats, device), "val": pair_acc(head, va, feats, device)}
    if human_val:  # human_val rows are kind='user' but sit outside tr/va, so the per-kind loop below misses them — report separately
        final["val_user"] = pair_acc(head, human_val, feats, device)
    for name, sel in (("train", tr), ("val", va)):
        for kind in ("hard", "anchor", "user", "anchor_low"):
            sub = [r for r in sel if r["kind"] == kind]
            if sub:
                final[f"{name}_{kind}"] = pair_acc(head, sub, feats, device)
    if tie_rows:
        Ta, Tb, _ty, _tw = pair_tensors(tie_rows, feats)
        with torch.no_grad():
            td = (head(Ta.to(device)) - head(Tb.to(device))).abs().mean().item()
        final["tie_mean_absdiff"] = td
    print("pairwise agreement:", {k: round(v, 3) for k, v in final.items()})

    # z-score statistics: score distribution over the full image set (forward only); at inference score_z = (score-mean)/std
    all_rows = rows + tie_rows
    uniq = sorted({r["a"] for r in all_rows} | {r["b"] for r in all_rows})
    with torch.no_grad():
        scores = head(torch.as_tensor(np.stack([feats[p][0] for p in uniq])).to(device))
    s_mean, s_std = float(scores.mean()), float(max(float(scores.std()), 1e-6))

    out = args.out or ROOT / "logs/ckpt" / ("rm_smoke.pt" if args.smoke else "rm.pt")
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "head": head.state_dict(),
        "backbone_state": backbone_state,      # self-contained: no weights file needed at inference (None for smoke/pickscore)
        "backbone": BACKBONE,
        "encoder": "smoke" if args.smoke else args.encoder,
        "weights": str(DEFAULT_WEIGHTS),
        "emb_dim": args.emb_dim,
        "hidden": args.hidden,
        "score_mean": s_mean,
        "score_std": s_std,
        "pair_acc": {k: float(v) for k, v in final.items()},
        "best_val": {"val": float(best["val"]), "epoch": best["epoch"],
                     "metric": "user" if human_val else "silver"},
        "n_pairs": {"train": len(tr), "val": len(va), "human_val": len(human_val),
                    "user": len(user_rows), "tie": len(tie_rows),
                    "auto": len(auto_rows)},
        "user_weight": args.user_weight if user_rows else None,
        "auto_weight": args.auto_weight if auto_rows else None,
        "epochs": args.epochs,
        "final_loss": float(history[-1]),
        "smoke": bool(args.smoke),
    }, out)
    print(f"checkpoint saved {out} (score z-score: mean={s_mean:.4f} std={s_std:.4f})")

    if args.smoke:
        assert all(np.isfinite(history)), "loss is NaN/Inf"
        assert history[-1] < history[0], f"loss did not decrease: {history[0]:.4f} -> {history[-1]:.4f}"
        assert final["val"] > 0.9, f"synthetic-data val agreement should be >0.9, got {final['val']:.3f}"
        print(f"RM SMOKE: PASS（bt_loss {history[0]:.4f} → {history[-1]:.4f}，val_acc={final['val']:.3f}）")


if __name__ == "__main__":
    main()
