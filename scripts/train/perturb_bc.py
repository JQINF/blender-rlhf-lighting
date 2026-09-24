"""perturb_bc.py — generate noisy correction pairs for BC data augmentation.

Anchors = before/final snapshots per (sku, view); 20 noisy variants of each, target always final:
  noisy before -> final (learn "repair a broken inheritance"); noisy final -> final (learn "refine near-optimal" = step-2 behavior).
Noise is applied in physical space with per-dim semantics (dict edits on the anchor config; pure numeric, no bpy):
  pos   each component += U(-1, 1) m        (±20% x POS_RANGE 5m, additive)
  size  x= U(0.8, 1.25)                     (±20% multiplicative, clamp [SIZE_MIN, SIZE_MAX])
  energy x= 10^U(-0.114, 0.114) dex         (~x0.77-1.30, clamp [ENERGY_MIN, ENERGY_MAX])
  aim/roll untouched; soft-off slots (energy <= 0.1+eps) keep whole-slot POSE_OFF constants (so BC never learns noise).
Post-noise simplified env clamping (pos radius < 0.65 pushed radially out / z >= -1.2) — approximates the clamp_pos in apply
(exact R is per-product and needs bpy; a few residual illegal points fall within BC noise tolerance; deployment input is always a legal post-apply state).

Output = same snapshot format {"sku","view","kind","config"} to data/snapshots/perturb/
  <sku>_<view>_perturb_{b|f}<NN>.json (b=noisy before anchor / f=noisy final anchor) —
  Main consumer in this repo: the training env, which perturbs inherited start configs with this noise model.

Usage: .venv/bin/python scripts/train/perturb_bc.py [--per-anchor 20] [--seed 42] [--clean] [--skus 38,39,...]
  --skus: generate only these products (for adding new products; never rerun the full set — the RNG consumes in sorted(pairs) order,
          so inserting a product into the sort shifts all later perturbations and desyncs them from the old renders/256/perturb/ images)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SNAP_DIR = ROOT / "data" / "snapshots"
OUT_DIR = SNAP_DIR / "perturb"

sys.path.insert(0, str(ROOT / "addons"))
from lighting_rl import core  # noqa: E402

N_PER_ANCHOR = 20
POS_JITTER = 0.2 * core.POS_RANGE      # ±1m additive
SIZE_JITTER = (0.8, 1.25)              # multiplicative ±20%
ENERGY_JITTER_DEX = 0.114              # ±dex on log axis
FLOOR_Z = -1.2                         # matches core.FLOOR_Z
MIN_R = 0.65                           # simplified bounding-sphere radius (normalized products R ∈ ~[0.5, 0.87], low-biased conservative value)


def perturb_config(base, rng):
    """Deep-copy base and add per-slot noise (soft-off slots skipped entirely)."""
    cfg = json.loads(json.dumps(base))
    for s in cfg["slots"].values():
        if s["energy"] <= core.ENERGY_MIN + core.OFF_EPS:
            continue  # soft-off slot keeps canonical constants
        s["pos"] = [float(p + rng.uniform(-POS_JITTER, POS_JITTER)) for p in s["pos"]]
        s["size"] = [float(np.clip(x * rng.uniform(*SIZE_JITTER), core.SIZE_MIN, core.SIZE_MAX))
                     for x in s["size"]]
        s["energy"] = float(np.clip(s["energy"] * 10.0 ** rng.uniform(-ENERGY_JITTER_DEX, ENERGY_JITTER_DEX),
                                    core.ENERGY_MIN, core.ENERGY_MAX))
        # simplified env clamping
        p = np.asarray(s["pos"])
        if np.linalg.norm(p) < MIN_R:
            s["pos"] = (p * (MIN_R / max(np.linalg.norm(p), 1e-6))).tolist()
        s["pos"][2] = max(s["pos"][2], FLOOR_Z)
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-anchor", type=int, default=N_PER_ANCHOR)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--clean", action="store_true", help="wipe the perturb dir before generating")
    ap.add_argument("--skus", type=str, default=None,
                    help="comma-separated skus; generate only these products (for adding new products; old perturbations stay matched to old renders)")
    args = ap.parse_args()

    pairs = {}
    for p in sorted(SNAP_DIR.glob("*.json")):
        d = json.loads(p.read_text())
        pairs.setdefault((d["sku"], d["view"]), {})[d["kind"]] = d["config"]

    if args.skus:
        keep = set(args.skus.split(","))
        pairs = {k: v for k, v in pairs.items() if k[0] in keep}

    if args.clean and OUT_DIR.is_dir():
        for old in OUT_DIR.glob("*.json"):
            old.unlink()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    n = 0
    for (sku, view), kinds in sorted(pairs.items()):
        if "before" not in kinds or "final" not in kinds:
            print(f"  skipped {sku}_{view} (missing before or final)")
            continue
        for anchor, tag in (("before", "b"), ("final", "f")):
            for i in range(args.per_anchor):
                cfg = perturb_config(kinds[anchor], rng)
                out = OUT_DIR / f"{sku}_{view}_perturb_{tag}{i:02d}.json"
                out.write_text(json.dumps({"sku": sku, "view": view, "kind": "perturb",
                                           "config": cfg}, ensure_ascii=False, indent=1))
                n += 1
    print(f"generated {n} perturbed configs -> {OUT_DIR} ({len(pairs)} views x 2 anchors x {args.per_anchor})")
    sys.exit(0 if n else 1)


if __name__ == "__main__":
    main()
