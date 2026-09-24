# Method & Key Findings

> 中文版本：[method.md](method.md)

## Task

The studio is a fixed Blender scene (`scene/stage.blend`): 8 light slots
(`light_01`-`light_08`), a `product` collection and a single camera. A lighting setup is a
**72-dim config** (8 slots × 9 params: position 3 / aim 2 / roll 1 / energy 1 / size 2, in tanh
space). A task instance is (product, view); the two views are front and high-45°, each with its
own config.

## Inference pipeline

```
new product ─▶ embeddings (frozen ResNet18 over 4 auxiliary renders) ─▶ NN inheritance:
               borrow the nearest product's hand-tuned 72-dim config as the start
            ─▶ policy 2-step refinement: step1 action → preview render → embedding into obs;
               step2 action → final 72-dim → train-tier render
            ─▶ best-of-16: sample 16 candidates from the same obs → RM arbitration →
               winner applied to the scene
```

Observation = 2640 dims = [static 2056 (product embedding 2052 + view/jitter 4) | current config 72
| preview embedding 512]. Reward = RM z-score of the final frame. Deployment also has a
do-no-harm gate (keep the inherited lighting when the improvement Δz ≤ τ).

## RM (human-preference model)

DINOv2-L/14 frozen backbone + a small scoring head (1024→128→1). Trained on human blind pairwise
preferences over rendered images (pairwise hinge loss, human-label-only recipe);
z-score = (raw − train mean) / std. At deploy time the RM only **arbitrates** candidates.

## Training recipe (the final policy)

| Item | Value |
|---|---|
| Engine | RLOO: K=10 replays per start, advantage = R_i − mean(R_≠i) (exact leave-one-out baseline, **no critic**) |
| Init | hot start from the previous policy of the same pipeline (including its learned σ) |
| Reward | same-generation RM z-score, **terminal only** (2-step episode, r1 ≡ 0) |
| Extras | none (no entropy bonus / no anchor / no repulsion; instruments only: min-grp-std, collapse warning, segment ckpts) |
| Start mechanism | NN inheritance + perturbation over 46 seeded products (leave-self-out) |
| Brakes | clip 0.2 + target-KL 0.2 early stop + grad-norm 0.5 |
| Budget | 300k env steps = 326 updates (46 starts × K10 × 2 steps = 920 steps/update), two parallel workers |

Key design: no critic (with a 2-step terminal reward V(s) adds nothing, and an off-policy actor
would exploit the critic's extrapolation error); the brakes plus σ self-annealing make the policy
grind slowly on the RM's fine-grained landscape (~+0.1z per 40-60 updates).

## Generalization: what transfers, what does not

**Transfers — the refinement policy works on unseen products**

| Pool | Δ(policy − noisy-inherit baseline) | win cells |
|---|---|---|
| Seeded products, 32 cells | **+1.674** | 31/32 |
| Holdout, 6 products / 12 cells (never trained or labeled) | **+0.793** | 12/12 |
| New products, 85 cells (no seed at training time) | **+1.200** | 79/85 |

Human blind evaluation (85 new-product pairs, double-blind): **0.472** vs the strongest historical
baseline 0.528 (p ≈ 0.90, a tie), with **58% of pairs indistinguishable** to the rater. In other
words, the refinement behaviour transfers to unseen products — it beats the
"deliver the inherited lighting" baseline and matches the strongest baseline on human preference.

**Does not transfer — product-specific taste cannot be predicted**

Across 46 seeds (92 views), **94.8% of the lighting variance is product-specific** (cross-product
shared component 4.4-5.2%). Predicting the product-specific residual from product observations
(embeddings in two spaces / per-light contribution maps; linear and nonlinear predictors) gives
**R² ≤ 0 under strict leave-one-product-out, 0 of 46 products positive** — the residual is
**aesthetically arbitrary, not physically determined**: two products with identical light response
may receive completely different lightings. New-product quality therefore comes from **coverage**
(hand-tune one seed per product), not from inference over observations.

**Recipe-level reproducibility**: retraining the same recipe (same-generation RM as reward +
champion-parent hot start + zero extras) on the modern chassis reaches or exceeds the original
champion anchors on all three pools (+1.582/+0.832/+1.137 vs +1.282/+0.428/+1.185) and ties it in
the new-product human evaluation — the champion lineage can be re-derived from one RM checkpoint
plus one training command.

## Boundaries

- Human-evaluation results come from a **single expert rater** (the deployment target is that
  operator's taste)
- All numbers come from one white-background e-commerce category; external validity is untested
- Cross-product diversity ≈ 0.5× the human level (inherent to this recipe; not an acceptance
  metric here, disclosed as-is)
- z-scores from different RM generations are not comparable (each is normalised against its own
  training set)
