"""rank_candidates.py — score and rank best-of-N candidate renders (deployment pipeline scoring step).

Loads an rm checkpoint (train_rm.py output: frozen image encoder + scoring head + z-score statistics;
clip_b16 embeds the backbone weights so no download is needed at inference; HF-family encoders are rebuilt
offline from the HF cache per HF_REPOS), scores candidate images (z-score) from a directory (--dir) or a list
file (--list), sorts descending, and writes a ranking json + winner path.
At deployment this is the scoring step of per-view best-of-16 (step-2 state sampled 16 candidates, scored at the same render tier).

--smoke: builds 8 random 224² images in a temp dir + an rm checkpoint (default logs/ckpt/rm_smoke.pt;
built on the fly with random weights in train_rm's smoke style if missing) and runs the full ranking,
writing logs/ckpt/rank_smoke.json.
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]

BACKBONE = "ViT-B-16"
EMB_DIM = 512
IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp"}

# HF-family encoder rebuild table (mirrors train_rm.ENCODERS; this file stays standalone, no training-script imports):
# encoder name -> (model_repo, processor_repo, model kind[, input resolution]); weights come from the HF cache (HF_HUB_OFFLINE=1)
HF_REPOS = {
    "pickscore": ("yuvalkirstain/PickScore_v1", "openai/clip-vit-large-patch14", "clip"),
    "clip_l336": ("openai/clip-vit-large-patch14-336", "openai/clip-vit-large-patch14-336", "clip"),
    "siglip_so400m": ("google/siglip-so400m-patch14-384", "google/siglip-so400m-patch14-384", "siglip"),
    "siglip2_so400m": ("google/siglip2-so400m-patch14-384", "google/siglip2-so400m-patch14-384", "siglip"),
    "dinov2_l": ("facebook/dinov2-large", "facebook/dinov2-large", "dinov2"),
    "dinov2_l518": ("facebook/dinov2-large", "facebook/dinov2-large", "dinov2", 518),
}


class ScoreHead(nn.Module):
    """Same structure as train_rm.py's ScoreHead (Linear(emb,h)+ReLU+Dropout(0.2)+Linear(h,1); hidden=0 degrades to a linear probe);
    kept here so this file stays standalone with no training-script imports."""

    def __init__(self, emb_dim=EMB_DIM, hidden=128):
        super().__init__()
        self.net = (nn.Sequential(nn.Linear(emb_dim, hidden), nn.ReLU(), nn.Dropout(0.2),
                                  nn.Linear(hidden, 1)) if hidden > 0 else nn.Linear(emb_dim, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def build_rm(ckpt_path, device, allow_random_fallback=False):
    """Restore an RM from a checkpoint -> (encode_fn, head, mean, std, src).
    encode_fn(paths) encodes a batch of image paths into a (B, emb_dim) feature tensor on device.
    clip_b16 ckpts embed the backbone weights -> pretrained=None, no download; only old ckpts without backbone_state
    fall back to the pretrained tag. HF-family encoder ckpts do not embed the backbone (too large); rebuilt from the
    HF cache per HF_REPOS (WSL needs HF_HUB_OFFLINE=1). If the ckpt is missing and allow_random_fallback is set,
    builds a random-weight RM on the fly (= train_rm --smoke's random backbone + random head, z-score stats as 0/1 placeholders)."""
    if ckpt_path is not None and Path(ckpt_path).is_file():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        enc = ckpt.get("encoder")
        if enc in HF_REPOS:
            repo = HF_REPOS[enc]
            model_repo, proc_repo, kind = repo[0], repo[1], repo[2]
            img_size = repo[3] if len(repo) > 3 else None
            if kind == "siglip":
                from transformers import AutoProcessor, SiglipModel
                model = SiglipModel.from_pretrained(model_repo)
                processor = AutoProcessor.from_pretrained(proc_repo)
            elif kind == "dinov2":
                from transformers import AutoImageProcessor, AutoModel
                model = AutoModel.from_pretrained(model_repo)
                processor = AutoImageProcessor.from_pretrained(
                    proc_repo,
                    **({"size": {"shortest_edge": img_size},
                        "crop_size": {"height": img_size, "width": img_size}} if img_size else {}))
            else:
                from transformers import CLIPModel, CLIPProcessor
                model = CLIPModel.from_pretrained(model_repo)
                processor = CLIPProcessor.from_pretrained(proc_repo)

            if kind == "dinov2":
                def encode_fn(paths):
                    x = processor(images=[Image.open(p).convert("RGB") for p in paths],
                                  return_tensors="pt")["pixel_values"].to(device)
                    return model(pixel_values=x).last_hidden_state[:, 0]  # CLS token
            else:
                def encode_fn(paths):
                    x = processor(images=[Image.open(p).convert("RGB") for p in paths],
                                  return_tensors="pt")["pixel_values"].to(device)
                    out = model.get_image_features(pixel_values=x)
                    return getattr(out, "pooler_output", out)  # transformers v5+ returns BaseModelOutputWithPooling
        else:
            import open_clip
            pretrained = None if ckpt.get("backbone_state") is not None else ckpt.get("pretrained")
            model, _, preprocess = open_clip.create_model_and_transforms(
                ckpt.get("backbone", BACKBONE), pretrained=pretrained)
            if ckpt.get("backbone_state") is not None:
                model.visual.load_state_dict(ckpt["backbone_state"])

            def encode_fn(paths):
                x = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in paths]).to(device)
                return model.encode_image(x)

        head = ScoreHead(ckpt.get("emb_dim", EMB_DIM), hidden=ckpt.get("hidden", 128))
        head.load_state_dict(ckpt["head"])
        mean, std = float(ckpt.get("score_mean", 0.0)), float(ckpt.get("score_std", 1.0))
        src = str(ckpt_path)
    else:
        if not allow_random_fallback:
            sys.exit(f"error: RM checkpoint not found: {ckpt_path} (run train_rm.py first; use --smoke for self-test)")
        print(f"note: {ckpt_path} not found, building a random-weight RM on the spot (smoke only, scores meaningless)")
        import open_clip
        torch.manual_seed(0)
        model, _, preprocess = open_clip.create_model_and_transforms(BACKBONE, pretrained=None)

        def encode_fn(paths):
            x = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in paths]).to(device)
            return model.encode_image(x)

        head = ScoreHead()
        mean, std = 0.0, 1.0
        src = "<random-fallback>"
    model.requires_grad_(False).eval().to(device)
    head.eval().to(device)
    return encode_fn, head, mean, std, src


def collect_candidates(img_dir=None, list_file=None):
    """--dir scans common image extensions (sorted for reproducibility), or --list reads paths line by line."""
    if list_file is not None:
        paths = [Path(l.strip()) for l in Path(list_file).read_text().splitlines() if l.strip()]
    else:
        d = Path(img_dir)
        paths = sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not paths:
        sys.exit(f"error: no candidate images (dir={img_dir} list={list_file})")
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        sys.exit(f"error: {len(missing)} candidate images missing, e.g. {missing[0]}")
    return paths


def score_images(encode_fn, head, paths, device, batch_size=16):
    """Score raw in batches -> 1-D numpy (caller applies mean/std for z-score).
    Features are L2-normalized first — same protocol as train_rm.encode_images
    (the head was trained on unit-norm features; skipping normalization inflates scores by the feature norm, over 10x)."""
    out = []
    with torch.no_grad():
        for i in range(0, len(paths), batch_size):
            f = encode_fn(paths[i:i + batch_size])
            f = f / f.norm(dim=-1, keepdim=True)
            out.append(head(f.float()).cpu())
    return torch.cat(out).numpy()


def score_images_mc(encode_fn, head, paths, device, mc_k=16, batch_size=16):
    """MC-dropout uncertainty scoring (exploration arm): encode features once (L2-normalized, same protocol as score_images),
    then run K stochastic forward passes through the scoring head's Dropout -> (mean, std) 1-D numpy arrays (raw scale; caller applies z).
    Uncertainty comes only from head Dropout (backbone frozen in eval) — a head-level epistemic-uncertainty approximation;
    p is read from head.net[2], not hardcoded; hidden=0 linear-probe heads have no Dropout, std is identically 0.
    Functional dropout (training=True) does not touch the module's eval state — zero impact on existing callers."""
    import torch.nn.functional as F
    means, stds = [], []
    with torch.no_grad():
        for i in range(0, len(paths), batch_size):
            f = encode_fn(paths[i:i + batch_size])
            f = (f / f.norm(dim=-1, keepdim=True)).float()
            if isinstance(head.net, nn.Sequential) and len(head.net) == 4:
                h = head.net[1](head.net[0](f))  # ReLU(Linear(f))
                samples = torch.stack([
                    head.net[3](F.dropout(h, p=head.net[2].p, training=True)).squeeze(-1)
                    for _ in range(mc_k)])  # (K, B)
            else:  # linear probe: no stochastic source
                samples = head(f).unsqueeze(0).repeat(mc_k, 1)
            means.append(samples.mean(0).cpu())
            stds.append(samples.std(0).cpu() if mc_k > 1 else torch.zeros(f.shape[0]))
    return torch.cat(means).numpy(), torch.cat(stds).numpy()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group()
    src.add_argument("--dir", type=Path, default=None, help="candidate render image directory")
    src.add_argument("--list", type=Path, default=None, help="candidate image path list (one per line)")
    p.add_argument("--ckpt", type=Path, default=None, help="rm checkpoint (default logs/ckpt/rm[_smoke].pt)")
    p.add_argument("--out", type=Path, default=None, help="ranking json output path")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--cpu", action="store_true", help="force CPU (default: use CUDA if available)")
    p.add_argument("--smoke", action="store_true", help="random images + rm_smoke.pt (built on the fly if missing) to run the full ranking")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    tmp = None
    if args.smoke:
        # 8 random 224² images into a temp dir, exercising the real disk-IO path
        tmp = tempfile.TemporaryDirectory(prefix="rank_smoke_")
        rng = np.random.default_rng(0)
        for i in range(8):
            Image.fromarray(rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)).save(
                Path(tmp.name) / f"cand_{i:02d}.png")
        args.dir = Path(tmp.name)
        args.ckpt = args.ckpt or ROOT / "logs/ckpt/rm_smoke.pt"
        args.out = args.out or ROOT / "logs/ckpt/rank_smoke.json"
    elif args.dir is None and args.list is None:
        sys.exit("error: provide --dir or --list (use --smoke for self-test)")

    ckpt = args.ckpt or ROOT / "logs/ckpt/rm.pt"
    encode_fn, head, mean, std, src = build_rm(ckpt, device, allow_random_fallback=args.smoke)
    paths = collect_candidates(args.dir, args.list)
    raw = score_images(encode_fn, head, paths, device, args.batch_size)
    z = (raw - mean) / max(std, 1e-6)

    order = np.argsort(-z)  # descending
    ranking = [{"rank": r + 1, "path": str(paths[i]), "score_z": float(z[i]), "score_raw": float(raw[i])}
               for r, i in enumerate(order)]
    result = {"ckpt": src, "score_zscore": {"mean": mean, "std": std},
              "n": len(paths), "winner": ranking[0]["path"], "ranking": ranking}

    out = args.out or Path("ranking.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1))
    print(f"{len(paths)} candidates (RM: {src}), ranking written to {out}")
    for r in ranking[:3]:
        print(f"  #{r['rank']}  z={r['score_z']:+.3f}  {r['path']}")
    print(f"winner: {result['winner']}")

    if args.smoke:
        assert len(ranking) == 8 and Path(result["winner"]).is_file()
        assert all(np.isfinite(r["score_z"]) for r in ranking)
        tmp.cleanup()
        print("RANK SMOKE: PASS")


if __name__ == "__main__":
    main()
