<h1 align="center">blender-rlhf-lighting</h1>

<p align="center"><b>Automatic product lighting inside a Blender studio</b><br>
embedding inheritance → 2-step RLOO refinement → human-preference RM arbitrating best-of-16</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License: MIT"></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/python-3.13-blue.svg" alt="Python 3.13"></a>
  <a href="https://www.blender.org/"><img src="https://img.shields.io/badge/blender-5.x-orange.svg" alt="Blender 5.x"></a>
</p>

<p align="center">中文版：<a href="README_CN.md">README_CN.md</a> · Docs: <a href="docs/method_EN.md">method</a> · <a href="docs/evaluation_EN.md">evaluation</a> · <a href="docs/reproduction_EN.md">reproduction</a></p>

---

## Results

RM z-scores, deterministic policy, DINOv2-L preference model (same scale within a row):

| Pool | Δ vs baseline | wins | Δ vs supervised |
|---|---|---|---|
| Seed products (32) | **+1.674** | 31/32 | +0.446 |
| Holdout (12) | **+0.793** | 12/12 | +0.379 |
| New products (85) | **+1.200** | 79/85 | +0.780 |

- **baseline** = noisy inheritance delivered as-is · **supervised** = regression fitted on seeds
- **Human blind evaluation** (85 new-product pairs, double-blind): **0.472 vs 0.528** against the
  strongest historical model — a statistical tie (p ≈ 0.90), with **58% of pairs indistinguishable**
  to the rater

<table>
<tr>
<td><img src="images/new_products_grid.png" width="100%" alt="New products: inherit baseline / supervised regression / this repo / strongest baseline"></td>
<td><img src="images/seed_products_grid.png" width="100%" alt="Seed products: hand-tuned seed / inherit baseline / this repo"></td>
</tr>
<tr>
<td align="center"><sub><b>New products</b> — inherit baseline · supervised · <b>this repo</b> · strongest baseline</sub></td>
<td align="center"><sub><b>Seed products</b> — hand-tuned seed · inherit baseline · <b>this repo</b></sub></td>
</tr>
</table>

---

## Method at a glance

```
new product ─▶ embedding nearest-neighbour inheritance (the start)
            ─▶ policy 2-step refinement  (step1: preview render → obs2; step2: final render)
            ─▶ RM-arbitrated best-of-16  (deploy)   /   RM z-score (training reward)
```

**The policy trainer is RLOO** (`scripts/train/rloo_continuous_action.py`):

| Aspect | Design |
|---|---|
| Group sampling | `reset()` draws a start (product, view, jitter, inherited config + noise); `reset_replay()` replays it verbatim K=10 times, so a group differs only by the policy's own σ-sampling |
| Advantage | exact leave-one-out: `A_i = R_i − mean(R_{j≠i})`. **No critic, no GAE, no bootstrapping** — with a fixed 2-step episode and a terminal RM reward (r1 ≡ 0) a value function only adds an exploitable error surface |
| Why not off-policy | the SAC predecessor took five consecutive fixes to move reward −0.93 → −0.49, always asymptoting to the do-nothing baseline: replay + bootstrapped Q + an actor maximising Q lets the actor exploit the critic's extrapolation error (measured Q 9–19 vs real returns ≈ −2) |
| Stability | brakes rather than bonuses: clipped surrogate 0.2 · target-KL 0.2 early stop · grad-norm 0.5 · collapse-warning chain · checkpoints every 5 updates; σ is learned and self-anneals (no entropy bonus) |
| Throughput | two persistent Blender render workers; 46 starts × K10 × 2 steps = 920 env steps per update (300k steps ≈ 326 updates) |

Full recipe, observation layout and thresholds: [docs/method_EN.md](docs/method_EN.md)

---

## Validate in 10 minutes

1. Download the four assets from [Releases](https://github.com/JQINF/blender-rlhf-lighting/releases)

   | Asset | Goes to |
   |---|---|
   | `run38e_final.pt` (policy), `rm_v19c.pt` (RM) | `logs/ckpt/` |
   | `stage_pack.zip` (studio scene) | unzip into `scene/` |
   | `samples.zip` (products + embeddings + seed) | unzip into `model/`, `data/embeddings/`, `seeds/` |

2. Install dependencies: `uv sync` (see [docs/reproduction_EN.md](docs/reproduction_EN.md))
3. Run it:

```bash
HF_HUB_OFFLINE=1 .venv/bin/python scripts/deploy_light.py --sku 52 --view front \
  --actor logs/ckpt/run38e_final.pt --rm-ckpt logs/ckpt/rm_v19c.pt \
  --out logs/rollout/result.json
```

Or in Blender: install `addons/lighting_deploy`, open `scene/stage.blend`, press **Auto Light**
(inherit → refine → best-of-16 → winner applied). If the addon cannot find the project root, set
`LRL_PROJECT_ROOT=<repo path>`.

---

## Repository layout

```
scripts/    env/ (render tiers, resident worker, camera, intake) · train/ (rloo trainer, RM training,
            evaluation, blind-label tooling) · debug/ (log reader, deploy smoke test)
addons/     lighting_rl (interactive recording) + lighting_deploy (one-click deploy)
seeds/      46 hand-tuned lighting seeds (72-dim, front/high) + a generic template
docs/       method · evaluation · reproduction   (Chinese and English)
images/     the two comparison grids above
```

---

## Key findings

1. **Coverage is generalization** — 94.8% of the variance in hand-tuned lightings is
   product-specific, and that residual is not predictable from any product observation
   (embeddings / per-light contribution maps, linear or nonlinear: R² ≤ 0, 0/46 products).
   Lighting a new product well therefore comes from storing one human-tuned seed per product.
2. **Arbitration power ≠ reward power** — a preference model that improves as a judge does not
   necessarily improve policies when used as an RL reward (measured: arbitration +6.3pt while the
   policy's human win rate fell 0.373 → 0.156).
3. **Off-policy is structurally doomed here** — see the SAC account in the table above; dropping the
   critic (RLOO) fixes it.

---

## Assets & License

- Code: MIT ([LICENSE](LICENSE)). Weights and seeds: MIT, provided as-is.
- Sample products (`samples.zip`): demonstration use only.
- **Training data is not distributed**: no pair annotations or render sets are included. The repo
  ships the labeling and training tooling — label your own pairs with `label_server` to train an RM
  for your own taste.
- **Bring your own models**: drop your `.blend` files in per the intake contract (`product`
  collection, longest edge 1.0 m, bbox centred, transforms applied) and run the full
  inherit → refine → arbitrate pipeline with the shipped weights.

## Citation

```bibtex
@misc{blender-rlhf-lighting,
  title  = {blender-rlhf-lighting: RL-refined product lighting for Blender studios},
  year   = {2026},
  note   = {https://github.com/JQINF/blender-rlhf-lighting}
}
```
