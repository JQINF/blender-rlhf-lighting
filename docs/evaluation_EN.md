# Evaluation Protocol & Numbers

> 中文版本：[evaluation.md](evaluation.md)

## The three evaluation pools

| Pool | Composition | What it answers |
|---|---|---|
| Training pool (seeded) | 46 seeded products × 2 views, 32 views sampled | In-manifold accuracy (human endpoints live in this pool) |
| Holdout pool | 6 products that never took part in training or labeling (test_*) | Conservatism on unseen geometry |
| New-product pool | 85 products with no seed (skus 50-134), 1 view each | The deployment target: transfer |

## RM z-score reconciliation (deterministic policy, mean-arm convention)

Scoring RM = the same-generation DINOv2-L human-preference model (z normalised against its own
training set). Anchors = the historical champion model, same RM, same protocol.

| Pool | Δ(policy−baseline), final | win cells | champion anchor Δ | verdict |
|---|---|---|---|---|
| Seeded, 32 cells | +1.674 | 31/32 | +1.282 (31/32) | exceeds |
| Holdout, 12 cells | +0.793 | 12/12 | +0.428 (10/12) | exceeds |
| New products, 85 cells | +1.200 | 79/85 | +1.185 (75/85) | tie → exceeds |

Δ(policy−BC supervised baseline) on new products: **+0.780** (policy +0.710 vs BC −0.070) —
the RL refinement is not a repackaged supervised regression (BC sits at the baseline level on new
products).

**Scale discipline**: every number above uses the same RM; z-scores are not comparable across RM
generations.

## Human blind evaluation (final court)

Protocol: same (product, view) pairs (zero confound), randomised left/right, the rater does not know
which image comes from which arm; ties ("indistinguishable / both ugly / same lighting") are recorded
separately and excluded from the win-rate denominator; binomial significance plus same-batch RM
agreement are reported. The final batch = all 85 new-product pairs.

| Metric | Value |
|---|---|
| Human win rate (ties excluded, n=36) | final 17/36 = 0.472 vs champion 19/36 = 0.528 (p ≈ 0.90, a tie) |
| **Tie rate** | **49/85 = 58%** (historically 7-9% when the same champion frames faced other challengers) |
| RM on the same batch | 18/36 vs 18/36 (50/50); RM-human pairwise agreement 0.361 |

Reading: the final model and the champion are **statistically indistinguishable** on new products,
and on nearly six in ten pairs the rater could not tell them apart at all — a controlled replication.
Single-rater caveat: the conclusion is scoped to that expert's taste.

## Where the final model sits among historical new-product human results

| Model | New-product human win rate vs champion |
|---|---|
| Historical champion | baseline (0.658 in an independent batch) |
| **Final model (this repo)** | **0.472 (statistical-tie band)** |
| Supervised regression BC_v4 | 0.352 |
| Repulsion + product-fit-RM RL | 0.373 / 0.342 / 0.156 (three generations) |
| Interactive RL + render-embedding compass | 0.220 |

## Tooling shipped in this repo

- `scripts/train/eval_policy.py` — three-arm (policy/baseline, optional BC) paired evaluation on any
  product pool; `--rm-ckpt` is mandatory (never silently falls back to an old RM)
- `scripts/label_server.py` — static blind-labeling server (browser pairwise judging, answers written
  straight to disk)
- `scripts/train/make_blind200.py` + `make_blind_eval.py` + `score_blind_eval.py` — batch construction
  (pair_id management, randomised sides, ground truth kept in the manifest) and scoring (human win
  rate / same-batch RM / agreement / binomial p)
- `scripts/debug/read_train_log.py` — training-log reader and live follower (z window / grp_std /
  collapse warning)
- `scripts/debug/deploy_light_check.py` — deploy-pipeline smoke test (contract keys, gate
  self-consistency, same-args rerun reproducibility)
