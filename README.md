# blender-rlhf-lighting

**Automatic product lighting inside a Blender studio: NN inheritance → 2-step RL refinement → RM-arbitrated best-of-16.**

中文版：[README_CN.md](README_CN.md)

Given any product model in a Blender stage scene, the pipeline produces
e-commerce-grade white-background lighting: an embedding nearest-neighbour
search inherits a hand-tuned lighting from the closest known product as the
starting point → an RLOO policy performs a 2-step interactive refinement
(step 1 renders a preview into the observation, step 2 renders the final
image) → a human-preference reward model (RM) arbitrates 16 sampled
candidates and picks the winner. The repo ships the full training pipeline
(RM training / RL training / blind-evaluation infrastructure).

## Method at a glance

```
new product ─▶ embedding nearest-neighbour inheritance (the start)
            ─▶ policy 2-step refinement  (step1: preview render → obs2; step2: final render)
            ─▶ RM-arbitrated best-of-16  (deploy)   /   RM z-score (training reward)
```

**The policy trainer is RLOO** (`scripts/train/rloo_continuous_action.py`):

- **Same-start group sampling** — `reset()` draws a start (product, view, camera and product
  jitter, inherited config + noise); `reset_replay()` then replays that start verbatim K=10 times,
  so a group differs only by the policy's own σ-sampling.
- **Exact leave-one-out baseline** — `advantage_i = R_i − mean(R_{j≠i})`. No critic, no GAE, no
  bootstrapping: with a fixed 2-step episode and a terminal RM reward (r1 ≡ 0) a value function
  adds nothing but an exploitable error surface.
- **Why not off-policy** — the SAC predecessor went through five consecutive fixes that improved
  reward monotonically (−0.93 → −0.49) yet asymptoted toward the do-nothing baseline: replay +
  bootstrapped Q + an actor maximising Q lets the actor exploit the critic's extrapolation error
  (measured Q 9–19 against real returns ≈ −2). RLOO stays on-policy and drops the critic entirely.
- **Brakes, not bonuses** — clipped surrogate (0.2), target-KL 0.2 early stop, grad-norm 0.5,
  collapse-warning chain, segment checkpoints every 5 updates. σ is learned and self-anneals; no
  entropy bonus.
- **Throughput** — two persistent Blender render workers in parallel; 46 starts × K10 × 2 steps =
  920 env steps per update (300k steps ≈ 326 updates).

Full recipe table, observation layout, thresholds and the evaluation protocol:
[docs/method_EN.md](docs/method_EN.md) · [docs/reproduction_EN.md](docs/reproduction_EN.md)

## Results

**New products** (85 products with no hand-tuned seed at training time),
deterministic-policy evaluation, RM scores (DINOv2-L backbone preference
model, z-scale):

| Pool | Δ(policy − noisy-inherit baseline) | win cells | Δ(policy − BC supervised baseline) |
|---|---|---|---|
| Seed products, 32 cells | **+1.674** | 31/32 | +0.446 |
| Holdout products, 12 cells | **+0.793** | 12/12 | +0.379 |
| New products, 85 cells | **+1.200** | 79/85 | +0.780 |

**Human blind evaluation** (85 same-pose pairs, double-blind): final policy
vs. the strongest historical model **17/36 = 0.472 vs 0.528, a statistical
tie (p≈0.90)**; on **49/85 (58%) of pairs the rater judged the two images
indistinguishable**.

### New products — inherit baseline / supervised regression / this repo / the strongest baseline

<img src="images/new_products_grid.png" width="620" alt="New-product comparison grid: inherit baseline, supervised regression, this repo, strongest baseline">

### Seed products — hand-tuned seed / inherit baseline / this repo

<img src="images/seed_products_grid.png" width="600" alt="Seed-product comparison grid: hand-tuned seed, inherit baseline, this repo">

Docs: [method](docs/method_EN.md) · [evaluation](docs/evaluation_EN.md) · [reproduction](docs/reproduction_EN.md)

## Validate in 10 minutes (no training)

1. Download the four release assets: `run38e_final.pt` (policy), `rm_v19c.pt`
   (scoring RM), `stage_pack.zip` (stage scene), `samples.zip` (sample
   products + embeddings)
2. Unpack into `scene/`, `model/`, and `logs/ckpt/` respectively
3. Install dependencies (see [docs/reproduction_EN.md](docs/reproduction_EN.md))
4. One-shot CLI:

```bash
HF_HUB_OFFLINE=1 python scripts/deploy_light.py --sku <sample_id> --view front \
  --ckpt logs/ckpt/run38e_final.pt --rm-ckpt logs/ckpt/rm_v19c.pt
```

Or install the `addons/lighting_deploy` addon in Blender, open
`scene/stage.blend`, press "Auto Light" (inherit → refine → best-of-16 →
winner applied to the scene).

> If the addon cannot locate the project root, set `LRL_PROJECT_ROOT=<repo path>`.

## Key findings (TL;DR)

Full method and the "era" training recipe in [docs/method_EN.md](docs/method_EN.md).

1. **Coverage is generalization**: per-product lighting preference is not
   predictable from any product observation (embeddings / per-light
   contribution maps / linear / nonlinear all fail; 94.8% of the variance in
   hand-tuned lightings is product-specific). The only proven mechanism for
   lighting a new product well is human-tuning one seed per product.
2. **Arbitration power ≠ reward power**: preference-model improvements as a
   judge do not transfer to the gradient side when used as an RL reward.
3. **Off-policy is structurally doomed here**: five consecutive SAC fixes
   improved reward monotonically (−0.93→−0.49) yet asymptoted toward the
   do-nothing baseline; dropping the critic entirely (RLOO with a
   leave-one-out group baseline) fixed it.

## Assets & License

- Code: MIT (see LICENSE)
- Weights (`run38e_final.pt`, `rm_v19c.pt`) and seeds: MIT, provided as-is
- Sample product models (samples.zip): demonstration use only; your own
  product data never needs to leave your machine
- **Training data is not distributed**: no pair annotations or render sets are
  included. The repo ships the labeling and training tooling — label your own
  pairs with `label_server` to train an RM for your own taste
- **Bring your own models**: drop your `.blend` files in per the intake contract
  (`product` collection, longest edge 1.0m, bbox centred, transforms applied) and
  run the full inherit → refine → best-of-16 pipeline with the shipped weights

## Citation

```bibtex
@misc{blender-rlhf-lighting,
  title  = {blender-rlhf-lighting: RL-refined product lighting for Blender studios},
  year   = {2026},
  note   = {https://github.com/JQINF/blender-rlhf-lighting}
}
```
