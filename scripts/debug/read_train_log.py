"""read_train_log.py — RLOO training-log quick read.

logs/rloo_run*.jsonl has two line types: type=update (one per update: rew/entropy/kl/grp_std/warnings…)
and type=episode (one per rollout: sku/view/inherit_from/rew). This script turns them into a readable digest:

  [1] update table: one row per update (rew_mean±rew_std, entropy and its σ equivalent, approx_kl,
      clipfrac, kl_stop, grp_std, zero/low_var, collapse_warn, sps)
  [2] sliding-window stop line: midpoints of the last few 10-update windows + a "three consecutive
      trendless windows" verdict (Δ < 0.05 counts as flat)
  [3] per-pool episode flow: mean episode rew by sku pool — train(1-37)/batch2(38-49)/batch3(50-54)/batch4(55-134)/holdout(test_*)
      (see whether the new-product pool is learning)
  [4] --follow: live tail (poll for new lines every 5s; prints updates + a window summary every 10 updates)
  [5] same-file restart segmentation: an update-number rollback starts a new segment (typical trace of a
      same-file resume); reads only the latest segment by default (old segments collapsed with a notice); --all shows the full mosaic

Usage (CPU only):
  .venv/bin/python scripts/debug/read_train_log.py logs/rloo_repro.jsonl            # full read
  .venv/bin/python scripts/debug/read_train_log.py logs/rloo_repro.jsonl --follow   # live tail
  .venv/bin/python scripts/debug/read_train_log.py logs/rloo_repro.jsonl --last 15  # only the last 15 updates
"""

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

# Per-pool episode split. Defaults match the released dataset's product ranges
# (seeded core / three synthetic batches / the test_* holdout); unmatched skus land in "other",
# so editing these bands is optional when running on your own product pool.
POOLS = [("pool 1-37", lambda s: s.isdigit() and 1 <= int(s) <= 37),
         ("pool 38-49", lambda s: s.isdigit() and 38 <= int(s) <= 49),
         ("pool 50-54", lambda s: s.isdigit() and 50 <= int(s) <= 54),
         ("pool 55-134", lambda s: s.isdigit() and 55 <= int(s) <= 134),
         ("holdout test_*", lambda s: s.startswith("test_"))]


def sigma_of(ent):
    """Differential entropy of the 72-dim Gaussian → per-dim σ (troubleshooting appendix: σ = e^(ent/72 − 1.419))。"""
    return math.exp(ent / 72 - 1.4189)


def read(path):
    """Read the jsonl and split by training restarts: an update-number rollback = a new segment (typical trace of a same-file resume).
    Returns [(ups, eps), ...] in file order; length 1 when there was no restart. Episodes are assigned to the current segment by file position."""
    segs = [[[], []]]
    prev_up = 0
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("type") == "update":
            if r["update"] < prev_up:
                segs.append([[], []])
            prev_up = r["update"]
        ups, eps = segs[-1]
        (ups if r.get("type") == "update" else eps).append(r)
    return segs


def flatten(segs):
    ups = [r for u, _ in segs for r in u]
    eps = [r for _, e in segs for r in e]
    return ups, eps


def fmt_update(r):
    ent = r.get("entropy", float("nan"))
    seg = (f"  δp50{r.get('delta_p50', 0):.3f} p95{r['delta_p95']:.3f} sat{r.get('delta_sat', 0):.2f}"
           if r.get("delta_p95") is not None else "")  # residual-mode δ monitor (absent in old logs → not printed)
    anc = (f"  anchor_d{r['anchor_d']:.3f} λ{r['anchor_lam']:.2f}"
           if r.get("anchor_d") is not None else "")  # direct-anchor regularizer (--anchor-d-star)
    rep = (f"  repel_h{r['rep_hinge']:.3f} λ{r['rep_lam']:.2f} keep{r.get('rep_ratio', 0):.2f}"
           if r.get("rep_hinge") is not None else "")  # repel regularizer (--repel-alpha)
    return (f"u{r['update']:>3d} step{r['global_step']:>6d}  grp{r.get('grp_std', 0):.2f}"
            f" adv{r.get('adv_std', 0):.2f}"
            f"  ent{ent:+6.1f}(σ{sigma_of(ent):.3f})  kl{r.get('approx_kl', 0):.3f}"
            f"  clip{r.get('clipfrac', 0):.2f}  klstop{'Y' if r.get('kl_stop') else '-'}"
            f"  zero{r.get('zero_var_frac', 0):.2f}  low{r.get('low_var_frac', 0):.2f}"
            f"  warn{'⚠' if r.get('collapse_warn') else '-'}"
            f"  rew{r['rew_mean']:+.3f}±{r.get('rew_std', 0):.2f}(context){seg}{anc}{rep}  sps{r.get('sps', 0):.1f}")


def anchor_line(eps):
    """Anchor penalty reading (non-zero only under --anchor-lambda; skipped in old logs without anchor_pen)."""
    pens = [e.get("anchor_pen") for e in eps[-300:] if e.get("anchor_pen") is not None]
    if not pens:
        return None
    return (f"  anchor (last 300): mean anchor_pen {sum(pens) / len(pens):.3f}"
            f"(should be 0 when λ=0; = rm_score − episodic_return)")


def signal_check(ups):
    """RLOO signal health check: trend of within-group contrast (grp_std/adv_std) and zero-information-group share.
    RLOO optimizes the within-group LOO advantage — absolute rew is descriptive only, not the objective."""
    if len(ups) < 6:
        return
    g = [u.get("grp_std", 0) for u in ups]
    a = [u.get("adv_std", 0) for u in ups]
    z = [u.get("zero_var_frac", 0) for u in ups]
    lo = [u.get("low_var_frac", 0) for u in ups]
    n = min(10, len(ups) // 2)
    print(f"  [RLOO signal] grp_std median first {n} rounds {sorted(g[:n])[n // 2]:.3f} -> last {n} rounds "
          f"{sorted(g[-n:])[n // 2]:.3f} (halved = collapse warning; stable = signal alive) | "
          f"adv_std median {sorted(a)[len(a) // 2]:.2f} | zero {sum(z) / len(z):.2f} / low {sum(lo) / len(lo):.2f}")


def windows(ups, size=10):
    out = []
    for i in range(0, len(ups) - size + 1):
        chunk = ups[i:i + size]
        out.append((chunk[0]["update"], chunk[-1]["update"],
                    sum(c["rew_mean"] for c in chunk) / size))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", type=Path)
    ap.add_argument("--last", type=int, default=0, help="show only the last N updates (0 = all)")
    ap.add_argument("--follow", action="store_true", help="live tail (poll for new lines every 5s)")
    ap.add_argument("--window", type=int, default=10, help="sliding window size (default: 10 updates)")
    ap.add_argument("--all", action="store_true", help="multi-segment log (same-file restart): show all segments; default reads only the latest segment")
    args = ap.parse_args()
    if not args.log.is_file():
        raise SystemExit(f"log file not found: {args.log} (pass the jsonl written by --log-file during training)")
    def pick(segs):
        if args.all:
            return flatten(segs)
        return segs[-1]

    def snapshot(full=True):
        segs = read(args.log)
        ups, eps = pick(segs)
        if not args.all and len(segs) > 1:
            old_n = sum(len(u) for u, _ in segs[:-1])
            print(f"  (detected {len(segs)} segments = same-file restarts; older {old_n} updates folded, "
                  f"only the latest segment is shown below. --all shows the full mix)", flush=True)
        if not ups:
            print("(no update records yet — training just started?)", flush=True)
            return len(segs), 0
        show = ups[-args.last:] if args.last else ups
        if full and not args.last:
            print(f"=== {args.log}: {len(ups)} updates / {len(eps)} episodes ===")
        for r in show:
            print("  " + fmt_update(r), flush=True)
        ws = windows(ups, args.window)
        if len(ws) >= 3:
            tail = ws[-3:]
            trend = max(w[2] for w in tail) - min(w[2] for w in tail)
            print(f"  sliding window ({args.window} rounds), last three centres (rew context reading, not the RLOO objective): "
                  + " / ".join(f"u{a}-{b}: {m:+.3f}" for a, b, m in tail)
                  + f"  -> Δ={trend:.3f} {'(no trend = plateau)' if trend < 0.05 else ''}", flush=True)
        signal_check(ups)
        if full and eps:
            line = anchor_line(eps)
            if line:
                print(line, flush=True)
            print("  per-pool episode rew (last 300):")
            recent = eps[-300:]
            for name, pred in POOLS:
                sub = [e["episodic_return"] for e in recent if pred(str(e.get("sku", "")))]
                if sub:
                    print(f"    {name:16s} n={len(sub):4d}  mean {sum(sub) / len(sub):+.3f}")
            matched = sum(1 for e in recent if any(pred(str(e.get("sku", ""))) for _, pred in POOLS))
            rest = [e["episodic_return"] for e in recent if not any(pred(str(e.get("sku", ""))) for _, pred in POOLS)]
            if rest:
                print(f"    {'other':16s} n={len(rest):4d}  mean {sum(rest) / len(rest):+.3f}")
        return len(segs), len(ups)

    n_seg, seen = snapshot()
    if not args.follow:
        return
    print("\n-- live follow (Ctrl-C to exit; prints updates and a window summary every 10 rounds) --", flush=True)
    while True:
        time.sleep(5)
        try:
            segs = read(args.log)
        except FileNotFoundError:
            continue
        if not args.all and len(segs) != n_seg:
            n_seg = len(segs)
            seen = 0
            print(f"  [training restart detected: segment {n_seg} begins, readings below are from the new segment]", flush=True)
        ups, eps = pick(segs)
        for r in ups[seen:]:
            print("  " + fmt_update(r), flush=True)
        if len(ups) > seen and len(ups) % 10 == 0 and ups:
            ws = windows(ups, args.window)
            if len(ws) >= 3:
                tail = ws[-3:]
                trend = max(w[2] for w in tail) - min(w[2] for w in tail)
                print(f"    [window·rew context] " + " / ".join(f"u{a}-{b}: {m:+.3f}" for a, b, m in tail)
                      + f"  Δ={trend:.3f}" + ("  <- plateau" if trend < 0.05 else ""), flush=True)
                signal_check(ups)
                line = anchor_line(eps)
                if line:
                    print(line, flush=True)
        seen = len(ups)


if __name__ == "__main__":
    main()
