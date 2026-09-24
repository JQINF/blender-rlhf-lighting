"""make_blind_eval.py — build a blind-labeling batch directly from the three-arm final frames dumped by eval_policy (human-scale recheck of eval numbers for new-product pools).

Purpose: when the RM numbers from eval need a human-scale recheck, reuse the existing {ep}_{sku}_{view}_{arm}.png
frames under logs/rollout/<eval_dir>/ — no re-rendering — and assemble N (sku, view) pairs (default policy vs bc).

Outputs (same layout as make_blind200; labeling page / manifest / export fully compatible):
  blind200.jsonl  {"blind_id","pair_id"(synthesized from 900000),"sku","view","kind","a","b","side_<arm>"}
                  side_* records which arm is on which side (kept hidden from the page; used at scoring)
  manifest.json / index.html  same as make_blind200

Usage: .venv/bin/python scripts/train/make_blind_eval.py --eval-dir logs/rollout/<eval dir> --n 24 --out-dir data/pairs/blind24_repro
Score: .venv/bin/python scripts/train/score_blind_eval.py --blind-dir <same> --answers <export json> [--ckpt logs/ckpt/rm_v19c.pt]
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_blind200 import INDEX_HTML  # noqa: E402  reuse the labeling-page template


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-dir", type=Path, required=True, help="eval_policy --out-dir (source of arm_a)")
    ap.add_argument("--eval-dir-b", type=Path, default=None,
                    help="second eval dir (if given: cross-ckpt duel — arm_b images come from here, episode prefixes must match)")
    ap.add_argument("--arms", nargs=2, default=["policy", "bc"], help="the two arms to pair (default policy bc)")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--seed", type=int, default=207)
    ap.add_argument("--answers-name", default="blind_eval_answers.json")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    arm_a, arm_b = args.arms
    tag_a, tag_b = (arm_a, arm_b) if arm_a != arm_b else (f"{arm_a}A", f"{arm_b}B")

    def scan(d):
        eps = defaultdict(dict)
        for p in d.glob("*.png"):
            m = re.match(r"(.+_(\w+?)_(front|high))_(\w+)\.png$", p.name)
            if m:
                eps[(m.group(2), m.group(3))][m.group(4)] = p
        return eps

    eps_a = scan(args.eval_dir)
    if args.eval_dir_b:  # cross-ckpt duel: each arm's images come from its own dir, paired by (sku, view) prefix
        eps_b = scan(args.eval_dir_b)
        keys = sorted(k for k in eps_a if arm_a in eps_a[k] and arm_b in eps_b.get(k, {}))
        triples = {k: {tag_a: eps_a[k][arm_a], tag_b: eps_b[k][arm_b]} for k in keys}
        assert triples, f"episode prefixes do not match between the two dirs ({len(eps_a)} vs {len(eps_b)})"
    else:
        triples = {k: {tag_a: v[arm_a], tag_b: v[arm_b]}
                   for k, v in eps_a.items() if arm_a in v and arm_b in v}
        assert triples, f"cannot assemble {arm_a}/{arm_b} frames from {args.eval_dir} (saw {len(eps_a)} episode prefixes)"

    rng = np.random.default_rng(args.seed)
    keys = sorted(triples)
    take = min(args.n, len(keys))
    if take < args.n:
        print(f"{len(keys)} views total, fewer than --n {args.n}, sampling {take}")
    picked = [keys[i] for i in rng.choice(len(keys), size=take, replace=False)]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows, manifest = [], []
    for i, (sku, view) in enumerate(picked):
        fa = triples[(sku, view)][tag_a].resolve().relative_to(ROOT)
        fb = triples[(sku, view)][tag_b].resolve().relative_to(ROOT)
        flip = bool(rng.integers(2))  # randomize left/right; side_ records the truth
        a, b = (str(fb), str(fa)) if flip else (str(fa), str(fb))
        rows.append({"blind_id": i, "pair_id": 900000 + i, "sku": sku, "view": view,
                     "kind": f"eval_{tag_a}_vs_{tag_b}", "a": a, "b": b,
                     f"side_{tag_a}": "b" if flip else "a", f"side_{tag_b}": "a" if flip else "b"})
        manifest.append({"blind_id": i, "a": a, "b": b})
    (args.out_dir / "blind200.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest))
    html = INDEX_HTML.replace("blind200_answers.json", args.answers_name).replace(
        "Blind labeling (200 pairs)", f"Blind labeling ({len(rows)} pairs)")
    (args.out_dir / "index.html").write_text(html)
    print(f"blind set {len(rows)} pairs ({arm_a} vs {arm_b}, covering {len(picked)} views) -> {args.out_dir}")
    try:
        rel = args.out_dir.resolve().relative_to(ROOT)
        print(f"serve: `.venv/bin/python scripts/label_server.py` (static + answers to disk), "
              f"open http://localhost:8000/{rel}/")
    except ValueError:
        print(f"output dir is outside the project ({args.out_dir}); http.server cannot reach it")


if __name__ == "__main__":
    main()
