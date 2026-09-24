"""answers_to_labels.py — convert a labeling-page answers export into user_labels rows for train_rm.py and append to the master table.

The labeling page exports {"blind_id": "a"|"b"|"tie"}; this script looks up
(pair_id, a, b) from blind200.jsonl in the same directory and appends
{"pair_id","a","b","winner"} rows to user_labels.jsonl.
--pid-offset shifts pair_id to avoid cross-batch collisions (pair_id is only used for logging; train_rm joins on a/b paths).

Usage: .venv/bin/python scripts/train/answers_to_labels.py \
  --label-dir data/pairs/ppo_v1/label --answers ppo_v1_answers.json --pid-offset 100000
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
USER_LABELS = ROOT / "data/pairs/user_labels.jsonl"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label-dir", type=Path, required=True, help="labeling-page directory (contains blind200.jsonl)")
    ap.add_argument("--answers", default="blind200_answers.json", help="exported answers filename")
    ap.add_argument("--pid-offset", type=int, default=0)
    ap.add_argument("--out", type=Path, default=USER_LABELS)
    args = ap.parse_args()

    rows = {r["blind_id"]: r for r in
            (json.loads(l) for l in (args.label_dir / "blind200.jsonl").read_text().splitlines() if l.strip())}
    answers = {int(k): v for k, v in json.loads((args.label_dir / args.answers).read_text()).items()}

    out, skipped = [], 0
    for blind_id, w in sorted(answers.items()):
        r = rows.get(blind_id)
        if r is None:
            skipped += 1
            continue
        for key in ("a", "b"):
            assert (ROOT / r[key]).is_file(), f"blind_id {blind_id} image not found: {r[key]}"
        out.append({"pair_id": r["pair_id"] + args.pid_offset, "a": r["a"], "b": r["b"], "winner": w})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a") as f:
        for d in out:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    n_tie = sum(1 for d in out if d["winner"] == "tie")
    print(f"appended {len(out)} rows (tie {n_tie}) -> {args.out}" + (f", skipped {skipped} unknown blind_id" if skipped else ""))


if __name__ == "__main__":
    main()
