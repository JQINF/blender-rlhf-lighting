"""NN retrieval: embedding distance -> path of the most similar hand-tuned seed.

Usage (venv python):
  .venv/bin/python scripts/nn_retrieve.py --sku 13 [--exclude-self] [--json]

The addon shell calls retrieve() in-process via importlib instead of subprocess (pure numpy, ships with Blender).

Candidate = product having both data/embeddings/<sku>.npy and seeds/<sku>.json.
Distance = L2 on the embedding segment (first 2048 dims); bbox/pose dims are excluded (their magnitudes
would dominate the ranking; neighbor semantics come from the appearance/shape embedding).
The no-neighbor fallback to the generic template is the caller's decision via a distance threshold
(this script only reports the distance). Subprocess callers parse the last stdout line with --json.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EMB_DIR = ROOT / "data" / "embeddings"
SEED_DIR = ROOT / "seeds"
EMB_DIM = 2048  # first 2048 dims of the cached vector = clay 3x512 + neutral-light 512; trailing bbox/is_grounded dims stay out of the distance


def retrieve(sku, exclude_self=False):
    """Return {"seed": Path, "sku": str, "distance": float}; None if no candidates; FileNotFoundError if the query embedding is missing."""
    q_path = EMB_DIR / f"{sku}.npy"
    if not q_path.is_file():
        raise FileNotFoundError(f"query embedding missing: {q_path} (run extract_embeddings.py first)")
    q = np.load(q_path)[:EMB_DIM].astype(np.float64)

    candidates = []
    for seed_path in sorted(SEED_DIR.glob("*.json")):
        cand_sku = seed_path.stem
        if cand_sku == "template_generic":  # template is the no-neighbor fallback; not ranked as a neighbor
            continue
        if exclude_self and cand_sku == sku:
            continue
        emb_path = EMB_DIR / f"{cand_sku}.npy"
        if emb_path.is_file():
            candidates.append((cand_sku, seed_path, np.load(emb_path)[:EMB_DIM].astype(np.float64)))
    if not candidates:
        return None
    d, cand_sku, p = min((float(np.linalg.norm(q - emb)), c, p) for c, p, emb in candidates)
    return {"seed": p, "sku": cand_sku, "distance": d}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sku", required=True)
    ap.add_argument("--exclude-self", action="store_true",
                    help="leave-self-out for training products (rollout start-point sampling)")
    ap.add_argument("--json", action="store_true", help="print machine-readable JSON on the last line")
    args = ap.parse_args()

    try:
        hit = retrieve(args.sku, args.exclude_self)
    except FileNotFoundError as e:
        sys.exit(str(e))
    if hit is None:
        sys.exit(f"no candidates: no seed with an embedding under seeds/ ({SEED_DIR})")
    print(f"nearest neighbour: {hit['sku']}  distance={hit['distance']:.3f} -> {hit['seed']}")
    if args.json:
        print(json.dumps({"seed": str(hit["seed"]), "sku": hit["sku"],
                          "distance": hit["distance"]}))


if __name__ == "__main__":
    main()
