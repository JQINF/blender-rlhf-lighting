"""score_blind_eval.py — score a make_blind_eval batch: human-scale per-arm win rates + RM agreement.

Blind rows carry the side_<arm> truth (which side is which arm), so on top of RM-vs-human agreement
the tool reports human-scale win rates per arm (ties excluded) — the human recheck protocol for eval numbers.

Usage: HF_HUB_OFFLINE=1 .venv/bin/python scripts/train/score_blind_eval.py \
  --blind-dir data/pairs/<batch> --answers <export json> --ckpt logs/ckpt/rm_v19c.pt
"""

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rank_candidates import build_rm  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--blind-dir", type=Path, required=True)
    ap.add_argument("--answers", type=Path, required=True, help="JSON exported from the labeling page")
    ap.add_argument("--ckpt", type=Path, default=None, help="RM ckpt (if given, also reports RM agreement / per-arm win rates)")
    args = ap.parse_args()

    answers = {int(k): v for k, v in json.loads(args.answers.read_text()).items()}
    rows = [json.loads(l) for l in (args.blind_dir / "blind200.jsonl").read_text().splitlines() if l.strip()]
    arms = [k[5:] for k in rows[0] if k.startswith("side_")]
    assert len(arms) == 2, f"bad side_ field: {arms}"
    arm_a, arm_b = arms

    n_tie = sum(1 for v in answers.values() if v == "tie")
    print(f"human labels {len(answers)}/{len(rows)} (tie {n_tie} excluded)")

    score = None
    if args.ckpt:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        score = build_rm(args.ckpt, device)

    stat = {s: {"human_win": 0, "rm_win": 0} for s in arms}
    n_eval, agree_rm = 0, 0
    disagree = []
    for r in rows:
        u = answers.get(r["blind_id"])
        if u in (None, "tie"):
            continue
        n_eval += 1
        for s in arms:
            stat[s]["human_win"] += int(r[f"side_{s}"] == u)
        if score:
            sa, sb = score(ROOT / r["a"]), score(ROOT / r["b"])
            rm_w = "a" if sa > sb else "b"
            agree_rm += int(rm_w == u)
            for s in arms:
                stat[s]["rm_win"] += int(r[f"side_{s}"] == rm_w)
            if rm_w != u:
                disagree.append({"blind_id": r["blind_id"], "sku": r["sku"], "view": r["view"],
                                 "user": u, "rm": rm_w})

    ha, hb = stat[arm_a]["human_win"], stat[arm_b]["human_win"]
    print(f"\nhuman win rate (ties excluded, n={n_eval}): {arm_a} {ha}/{n_eval} = {ha / n_eval:.3f} | "
          f"{arm_b} {hb}/{n_eval} = {hb / n_eval:.3f}")
    if score:
        ra, rb = stat[arm_a]["rm_win"], stat[arm_b]["rm_win"]
        print(f"RM win rate (same pairs): {arm_a} {ra}/{n_eval} = {ra / n_eval:.3f} | "
              f"{arm_b} {rb}/{n_eval} = {rb / n_eval:.3f}")
        print(f"pairwise agreement (RM vs human): {agree_rm}/{n_eval} = {agree_rm / n_eval:.3f}")
        out = args.blind_dir / f"eval_gate_{Path(args.ckpt).stem}.json"
        out.write_text(json.dumps({"n_eval": n_eval, "arms": arms,
                                   "human": {s: stat[s]["human_win"] for s in arms},
                                   "rm": {s: stat[s]["rm_win"] for s in arms},
                                   "agree_rm": agree_rm, "disagree": disagree},
                                  ensure_ascii=False, indent=1))
        print(f"details saved {out}")


if __name__ == "__main__":
    main()
