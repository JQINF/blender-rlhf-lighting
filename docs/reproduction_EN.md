# Reproduction Guide

> 中文版本：[reproduction.md](reproduction.md)

## Environment

- **OS**: Linux (Ubuntu family); GPU optional (Cycles falls back OPTIX → CUDA → CPU; CPU works,
  just slowly)
- **Blender 5.x**: `blender` on PATH (snap or official build)
- **Python 3.13 venv**: `uv sync` (creates the venv from pyproject.toml + uv.lock and installs
  torch / torchvision / open_clip / transformers / gymnasium / numpy / Pillow / matplotlib).
  Development used a CUDA build of torch (cu126); CPU also works (rendering goes through Cycles, slower)
- **HF models**: the RM and embedding backbones are not distributed with the repo (everything
  except the tiny scoring head is too large); first use needs access to HuggingFace (or hf-mirror):
  - `rm_v19c.pt`'s backbone = **`facebook/dinov2-large` (DINOv2-L/14, frozen, CLS token, ~1.1 GB)**;
    the scoring head (1024 → 128 → 1, ~130k params) is inside the ckpt
  - the embedding backbone (ResNet18) uses torchvision pretrained weights
  - after one download, everything runs offline with `HF_HUB_OFFLINE=1`
  - exception: `train_rm.py --encoder clip_b16` needs your own
    `data/weights/open_clip_model.safetensors` (not distributed); every other backbone
    (including the default dinov2_l) is fetched from HF

## Assets (the four release files)

```
stage_pack.zip ─▶ stage.blend / embed_stage.blend → scene/
samples.zip    ─▶ <sku>.blend → model/ ; <sku>.npy → data/embeddings/
run38e_final.pt / rm_v19c.pt ─▶ logs/ckpt/
```

## Validating the final model (10 minutes)

```bash
# Deploy semantics, one shot (NN inherit → 2-step refine → best-of-16 → winner applied)
HF_HUB_OFFLINE=1 .venv/bin/python scripts/deploy_light.py --sku <sample_id> --view front \
  --actor logs/ckpt/run38e_final.pt --rm-ckpt logs/ckpt/rm_v19c.pt --out logs/rollout/result.json
```

Or through Blender: install `addons/lighting_deploy`, open `scene/stage.blend`, press
"Auto Light" in the N panel. If the addon cannot locate the project root, set
`LRL_PROJECT_ROOT=<repo path>`.

## Data spec (bringing your own products)

- Product file: `<sku>.blend` containing a collection named `product`; longest edge 1.0 m; bbox
  centred at the origin; rotation/scale applied
- Embeddings: `HF_HUB_OFFLINE=1 .venv/bin/python scripts/extract_embeddings.py --all`
  (4 auxiliary 512² renders per product incl. clay views; re-extract after changing a product).
  Rendering is device-nondeterministic (OPTIX/OIDN pixel jitter), so re-extracting on another
  device shifts embeddings by ~0.2% — harmless for retrieval and for the policy observation
  (same-machine reproducibility tolerances live in `scripts/debug/deploy_light_check.py`)
- Seeds (optional but strongly recommended): `seeds/<sku>.json` = {front, high} 72-dim configs.
  **A seeded product starts from its own seed at deploy time** (exclude-self retrieval) and benefits
  immediately from "coverage is generalization" — the only proven new-product quality mechanism
  (see docs/method.md)
- Render conventions: training/eval 256²@25spp+OIDN; embedding extraction 512²; final delivery 2048²

## Reproducing the training (era recipe)

Prerequisites: a product pool ≥10 (embeddings in place; seeds optional but recommended) and a
same-generation RM (`scripts/train/train_rm.py`, pure-human-pair recipe). **Note**: RM pair
annotations are not distributed — label your own pairs with `label_server.py`
(`make_blind200.py` builds a batch → label in the browser → `answers_to_labels.py` ingests), then
train an RM for your own taste and reuse the same recipe.

```bash
HF_HUB_OFFLINE=1 .venv/bin/python scripts/train/rloo_continuous_action.py \
  --env blender --rm-ckpt logs/ckpt/rm_v19c.pt \
  --episodes-per-product 1 --num-envs 2 --num-starts 46 --rloo-k 10 \
  --total-timesteps 300000 --ent-coef 0.0 --min-grp-std 0.05 \
  --stop-after-warns 6 --ckpt-every 5 \
  --resume-actor logs/ckpt/run38e_final.pt \
  --out logs/ckpt/rloo_actor_repro.pt --log-file logs/rloo_repro.jsonl
```

- 920 steps per update (46 starts × K10 × 2 steps); 300k ≈ 326 updates; 2 workers in parallel at
  SPS 2-3, ~30 h end to end; `--resume-actor` also serves for crash resume (use a fresh `--out` and
  log filename)
- Set `--num-starts` to your pool size (starts cover the pool through a no-replacement deck)
- Health monitor: `.venv/bin/python scripts/debug/read_train_log.py logs/rloo_repro.jsonl --follow`
  (the z window should grind from +0.3~0.5 up to a +0.8 plateau; grp_std must not halve; zero_var
  must not rise)
- **Two starting points**: ① continue training on the released policy (as above, hot start —
  smallest change, fastest climb); ② train your own style from scratch: drop `--resume-actor`,
  use random init plus `--init-std 0.06` (the σ-management convention established with the
  historical champion)

## Evaluation and blind labeling

```bash
# Three-pool z reconciliation (example: the 85-cell new-product pool; the BC arm is optional)
HF_HUB_OFFLINE=1 .venv/bin/python scripts/train/eval_policy.py \
  --ckpt logs/ckpt/run38e_final.pt --rm-ckpt logs/ckpt/rm_v19c.pt \
  --skus 50,51,...,134 --episodes 85 --out-dir logs/rollout/repro_new

# Human blind evaluation: start label_server.py, label data/pairs/<batch>/ in the browser, then
HF_HUB_OFFLINE=1 .venv/bin/python scripts/train/score_blind_eval.py \
  --blind-dir data/pairs/<batch> --answers data/pairs/<batch>/blind_eval_answers.json \
  --ckpt logs/ckpt/rm_v19c.pt
```

Scale discipline: every number must carry the `--rm-ckpt` it was produced with; cross-RM
comparisons are invalid.
