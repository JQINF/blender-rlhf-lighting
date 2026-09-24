"""Static embedding cache.

Default ResNet18 (ImageNet frozen): clay 3 views 3×512 + neutral-light photo 512 + bbox 3 + is_grounded 1 (constant 0, placeholder)
= 2052-dim static cache → data/embeddings/<sku>.npy.
--backbone dinov2_l: all four images switch to DINOv2-L/14 (1024-dim CLS per image, same convention as train_rm dinov2_l: AutoModel +
AutoImageProcessor default 224, facebook/dinov2-large, offline from the HF cache) → 4×1024+4 = 4100 dims
→ by default written to data/embeddings_dinov2l/<sku>.npy (the live ResNet18 cache is left untouched).
(The view one-hot 2 + camera-jitter sin/cos 2 are per-episode quantities, appended at runtime → full static state 2056 / 4104 dims.)

Usage (venv python, not Blender):
  .venv/bin/python scripts/extract_embeddings.py 13 [01 02 ...]   # specific skus
  .venv/bin/python scripts/extract_embeddings.py --all            # everything under model/
  --skip-render   reuse already-rendered pngs (rerun embedding extraction only; mandatory when switching backbone — rendering is encoder-independent)
  --random-init   ResNet18 loads no ImageNet weights (fallback when there is no network; embeddings are meaningless, pipeline test only)

Rendering is done by a Blender subprocess (scripts/env/embed_views.py + the hand-frozen rig scene/embed_stage.blend,
Cycles 512²@100spp+OIDN, ~8s per product).
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BLENDER = os.environ.get("BLENDER", "blender")
IMG_DIR = ROOT / "data" / "embeddings" / "img"
OUT_DIR = ROOT / "data" / "embeddings"

VIEWS = ("clay_front", "clay_side", "clay_top", "neutral")


def render_views(sku):
    cmd = [BLENDER, "-b", str(ROOT / "scene" / "embed_stage.blend"),
           "--python", str(ROOT / "scripts" / "env" / "embed_views.py"), "--", sku]
    # GPU passthrough for headless WSL2 (ignored elsewhere); see the env override below.
    env = dict(os.environ, GALLIUM_DRIVER="d3d12")
    print(f"[{sku}] rendering 4 views (embed tier Cycles 512²)...")
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if r.returncode != 0:
        print(r.stdout[-3000:])
        print(r.stderr[-2000:], file=sys.stderr)
        raise RuntimeError(f"embed_views rendering failed: {sku}")
    for v in VIEWS:
        p = IMG_DIR / f"{sku}_{v}.png"
        if not p.is_file():
            raise RuntimeError(f"render output missing: {p}")


def build_encoder(random_init=False, backbone="resnet18"):
    """Return (encode, load_batch): load_batch([PIL.Image]) → batch tensor; encode(batch) → (N, emb_dim).
    encode handles device placement internally. dinov2_l = same convention as train_rm dinov2_l (CLS token, processor default 224)."""
    import torch
    if backbone == "resnet18":
        from torchvision.models import resnet18, ResNet18_Weights
        weights = None if random_init else ResNet18_Weights.IMAGENET1K_V1
        model = resnet18(weights=weights)
        model.fc = torch.nn.Identity()
        model.eval()
        if torch.cuda.is_available():
            model.cuda()
        preprocess = ResNet18_Weights.IMAGENET1K_V1.transforms()  # preprocessing stays decoupled from the loaded weights, always identical

        def load_batch(imgs):
            return torch.stack([preprocess(im) for im in imgs])

        def encode(batch):
            return model(batch.to(next(model.parameters()).device))

    elif backbone == "dinov2_l":
        if random_init:
            raise ValueError("dinov2_l has no random-init fallback (weights come from the HF cache; HF_HUB_OFFLINE=1)")
        from transformers import AutoImageProcessor, AutoModel
        model = AutoModel.from_pretrained("facebook/dinov2-large")
        model.eval()
        if torch.cuda.is_available():
            model.cuda()
        processor = AutoImageProcessor.from_pretrained("facebook/dinov2-large")

        def load_batch(imgs):
            return processor(images=imgs, return_tensors="pt")["pixel_values"]

        def encode(batch):
            return model(pixel_values=batch.to(next(model.parameters()).device)).last_hidden_state[:, 0]

    else:
        raise ValueError(f"unknown backbone: {backbone}")
    return encode, load_batch


def extract(sku, encode, load_batch):
    import torch
    from PIL import Image
    imgs = [Image.open(IMG_DIR / f"{sku}_{v}.png").convert("RGB") for v in VIEWS]
    with torch.no_grad():
        feats = encode(load_batch(imgs)).float().cpu().numpy()  # (4, emb_dim)
    assert feats.shape[0] == len(VIEWS)
    bbox = json.loads((IMG_DIR / f"{sku}_bbox.json").read_text())["bbox_size"]
    vec = np.concatenate([feats[0], feats[1], feats[2], feats[3],
                          np.asarray(bbox, dtype=np.float32),
                          np.zeros(1, dtype=np.float32)])  # is_grounded constant 0 (placeholder for suspended mode)
    assert vec.shape == (4 * feats.shape[1] + 4,), vec.shape
    assert np.isfinite(vec).all()
    return vec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("skus", nargs="*", help="product sku(s) (e.g. 13); use --all for every product")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--skip-render", action="store_true")
    ap.add_argument("--random-init", action="store_true")
    ap.add_argument("--backbone", choices=["resnet18", "dinov2_l"], default="resnet18",
                    help="dinov2_l = all four images switch to DINOv2-L/14 (4100-dim cache, by default written to embeddings_dinov2l)")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="cache output directory (default resnet18→data/embeddings, dinov2_l→data/embeddings_dinov2l)")
    args = ap.parse_args()

    if args.all:
        skus = sorted(p.stem for p in (ROOT / "model").glob("*.blend"))
    else:
        skus = [s.removesuffix(".blend") for s in args.skus]
    if not skus:
        ap.error("no sku given (or --all)")

    out_dir = args.out_dir or (OUT_DIR if args.backbone == "resnet18"
                               else ROOT / "data" / "embeddings_dinov2l")
    model = preprocess = None
    ok = 0
    for sku in skus:
        out = out_dir / f"{sku}.npy"
        try:
            if not args.skip_render:
                render_views(sku)
            if model is None:
                model, preprocess = build_encoder(args.random_init, args.backbone)
            vec = extract(sku, model, preprocess)
            out_dir.mkdir(parents=True, exist_ok=True)
            np.save(out, vec)
            ok += 1
            print(f"[{sku}] -> {out}  shape={vec.shape}  "
                  f"|emb|={np.linalg.norm(vec[:-4]):.1f} bbox={vec[-4:-1].round(3)}")
        except Exception as e:
            print(f"[{sku}] FAIL: {type(e).__name__}: {e}", file=sys.stderr)
    print(f"done {ok}/{len(skus)} (backbone={args.backbone})")
    sys.exit(0 if ok == len(skus) else 1)


if __name__ == "__main__":
    main()
