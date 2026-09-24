"""eval_policy.py — evaluate the SAC actor's mean policy (pilot analysis).

Motivation: the episodic_return in the sac_continuous_action.py training log is the return of **sampled actions**
(mean + std·ε); the auto-tuned alpha entropy term keeps pushing exploration std up, so the return can be drowned
by exploration noise. This script runs the closed loop with the **deterministic action tanh(mean)** (third return
value of get_action), paired **same-start** against a baseline of "render the noisy-inheritance start directly"
(the baseline/BC arms replay the policy arm's exact start via reset_replay: product/view/jitter/inheritance noise all identical):

  SAC mean > baseline  -> learning is effective; the -1.9 plateau in the training log is just exploration noise
  SAC mean <= baseline -> true drift / reward hacking; tighten the BC regularization tail or the alpha floor

Final frames land in --out-dir for visual inspection ({seed}_{sku}_{view}_{policy|base|bc}.png, same-start three-arm pairs).

Usage:
  HF_HUB_OFFLINE=1 .venv/bin/python scripts/train/eval_policy.py \
      --ckpt logs/ckpt/sac_actor_pilot.pt --episodes 16
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))  # same-dir import, same as refine_env

from sac_continuous_action import CONFIG_DIM, OBS_DIM, Actor, BCActorRef  # noqa: E402
from refine_env import BlenderRefineEnv  # noqa: E402

STATIC_DIM = OBS_DIM - CONFIG_DIM - 512  # 2056: offset of config within obs


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=str(ROOT / "logs/ckpt/sac_actor_pilot.pt"))
    p.add_argument("--bc-ckpt", default=None,
                   help="optional: BC ckpt (e.g. logs/ckpt/mvp_actor_bc.pt) to also run a third BC reference arm")
    p.add_argument("--episodes", type=int, default=16)
    p.add_argument("--port", type=int, default=17871,
                   help="render worker port; when training is running, use 17872 for rechecks to avoid stealing the training worker's connection")
    p.add_argument("--seed-base", type=int, default=9100)
    p.add_argument("--holdout", action="store_true",
                   help="evaluate on the 6 holdout products (test_*) — generalization acceptance, excludes memorization")
    p.add_argument("--skus", default=None,
                   help="comma-separated arbitrary product pool (e.g. 38,39,40 or test_01): the pool is fully specified by it"
                        "(only requires embeddings to exist; seedless products automatically get the NN-inherit start); mutually exclusive with --holdout")
    p.add_argument("--rm-ckpt", type=Path, required=True,
                   help="RM checkpoint path (required; no default, so a stale RM can never be used silently)")
    p.add_argument("--out-dir", default=str(ROOT / "logs/rollout/mean_eval"))
    return p.parse_args()


def load_actor(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    actor = Actor(ckpt.get("obs_dim", OBS_DIM), ckpt.get("action_dim", CONFIG_DIM),
                  ckpt.get("hidden_dim", 256), residual=ckpt.get("residual", False)).to(device)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()
    return actor


@torch.no_grad()
def mean_action(actor, obs, device):
    x = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    _, _, mean = actor.get_action(x)  # third return value = tanh(mean), deterministic policy
    return mean.squeeze(0).cpu().numpy()


_GOAL_VEC = None  # injected goal for goal-conditioned ckpts (median-goal r18 embedding)


def _with_goal(obs):
    return np.concatenate([obs, _GOAL_VEC]).astype(np.float32) if _GOAL_VEC is not None else obs


def run_episode(env, actor, seed, out_dir, tag, device):
    """One episode with tanh(mean) on both steps; returns (rm_z, rm_raw, sku, view, final-frame path).
    Goal-conditioned ckpt (obs_dim > 2640): inject the median-goal embedding (run34 eval protocol = zero-click auto mode)."""
    obs, info = env.reset(seed=seed)
    obs = _with_goal(obs)
    a1 = mean_action(actor, obs, device)
    obs2, _, term, _, _ = env.step(a1)
    assert not term
    obs2 = _with_goal(obs2)
    a2 = mean_action(actor, obs2, device)
    _, r2, term, _, info2 = env.step(a2)
    assert term
    sku, view = info["sku"], info["view"]
    dst = out_dir / f"{seed}_{sku}_{view}_{tag}.png"
    shutil.copy(env._tmp / f"{os.getpid()}_{env.port}_s2.png", dst)  # filename is the Python process PID (refine_env.step)
    return r2, info2["rm_score_raw"], sku, view, dst


def run_baseline(env, seed, out_dir):
    """Same-start baseline: reset_replay replays the policy arm's exact start (product/view/jitter/inheritance noise all identical);
    both steps feed the inherited config back (the config segment of obs), so the final frame is the start rendered directly. seed only names the file.
    Note: after shuffled iteration the deck persists across resets, so calling reset(seed) three times would draw three
    different cards and silently degrade "same-seed pairing" into unpaired cross-product — hence the reset_replay path below."""
    obs, info = env.reset_replay()
    a_inherit = obs[STATIC_DIM:STATIC_DIM + CONFIG_DIM]
    obs2, _, term, _, _ = env.step(a_inherit)
    assert not term
    _, r2, term, _, info2 = env.step(a_inherit)
    assert term
    sku, view = info["sku"], info["view"]
    dst = out_dir / f"{seed}_{sku}_{view}_base.png"
    shutil.copy(env._tmp / f"{os.getpid()}_{env.port}_s2.png", dst)
    return r2, info2["rm_score_raw"], sku, view, dst


def load_bc(bc_ckpt, device):
    ckpt = torch.load(bc_ckpt, map_location="cpu", weights_only=True)
    ref = BCActorRef(ckpt.get("obs_dim", OBS_DIM), ckpt.get("hidden", 256), CONFIG_DIM,
                     residual=ckpt.get("residual", False))
    ref.load_state_dict(ckpt["state_dict"])
    return ref.requires_grad_(False).eval().to(device)


@torch.no_grad()
def bc_action(ref, obs, device):
    x = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    return ref(x).squeeze(0).clamp(-1.0, 1.0).cpu().numpy()  # BCActorRef has no tanh head; clip into action space


def run_bc(env, ref, seed, out_dir, device):
    """BC anchor arm: reset_replay same-start replay (same pairing fix as run_baseline);
    both steps run the BC forward pass (clip into [-1,1]), same protocol as run_episode. seed only names the file."""
    obs, info = env.reset_replay()
    obs2, _, term, _, _ = env.step(bc_action(ref, obs, device))
    assert not term
    _, r2, term, _, info2 = env.step(bc_action(ref, obs2, device))
    assert term
    sku, view = info["sku"], info["view"]
    dst = out_dir / f"{seed}_{sku}_{view}_bc.png"
    shutil.copy(env._tmp / f"{os.getpid()}_{env.port}_s2.png", dst)
    return r2, info2["rm_score_raw"], sku, view, dst


def main():
    args = parse_args()
    if not args.rm_ckpt.is_file():
        raise SystemExit(f"--rm-ckpt not found: {args.rm_ckpt} (hard fail, no silent fallback — write the explicit RM path)")
    if args.skus and args.holdout:
        raise SystemExit("--skus and --holdout are mutually exclusive: use --skus for an explicit pool, --holdout for the 6 holdout products")
    sku_list = [s.strip() for s in args.skus.split(",") if s.strip()] if args.skus else None
    if sku_list is not None and not sku_list:
        raise SystemExit("--skus parsed to an empty list (comma separated, e.g. 38,39,40)")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global _GOAL_VEC
    actor = load_actor(args.ckpt, device)
    _ad = actor.fc1.in_features
    if _ad > OBS_DIM:  # goal-conditioned ckpt: inject the median goal (19_high) embedding
        z = np.load(ROOT / "data/raft/run30v2/seed_ref_embeds.npz")
        _GOAL_VEC = z["19_high"].astype(np.float32)
        print(f"goal-cond ckpt (obs {_ad} = {OBS_DIM}+{ _ad - OBS_DIM}) injecting the median goal 19_high")
    bc_ref = load_bc(args.bc_ckpt, device) if args.bc_ckpt else None
    env = BlenderRefineEnv(rm_ckpt=args.rm_ckpt, episodes_per_product=1,
                           port=args.port, holdout=args.holdout, skus=sku_list)  # new product every ep for more diverse samples
    arms = "policy|baseline" + ("|BC" if bc_ref else "")
    pool = (f"skus[{','.join(sku_list)}]" if sku_list
            else ("holdout(test_*)" if args.holdout else "training pool"))
    print(f"mean-policy eval: ckpt={Path(args.ckpt).name}, RM={args.rm_ckpt.name}, pool={pool}, "
          f"{args.episodes} ep x ({arms}) paired", flush=True)
    zs_sac, zs_base, zs_bc = [], [], []
    try:
        for i in range(args.episodes):
            seed = args.seed_base + i
            z_s, raw_s, sku, view, _ = run_episode(env, actor, seed, out_dir, "policy", device)
            z_b, raw_b, _, _, _ = run_baseline(env, seed, out_dir)
            zs_sac.append(z_s)
            zs_base.append(z_b)
            line = (f"[{i + 1:02d}] {sku}_{view}  policy {z_s:+.3f} (raw {raw_s:+.3f}) | "
                    f"baseline {z_b:+.3f} (raw {raw_b:+.3f})")
            if bc_ref is not None:
                z_c, raw_c, _, _, _ = run_bc(env, bc_ref, seed, out_dir, device)
                zs_bc.append(z_c)
                line += f" | BC {z_c:+.3f} (raw {raw_c:+.3f})"
            print(line + f" | Δ(policy-baseline) {z_s - z_b:+.3f}", flush=True)
    finally:
        env.close()
    zs_sac, zs_base = np.asarray(zs_sac), np.asarray(zs_base)
    d = zs_sac - zs_base
    summary = (f"\npolicy mean {zs_sac.mean():+.3f} | baseline mean {zs_base.mean():+.3f} | "
               f"paired Δ mean {d.mean():+.3f} ({int((d > 0).sum())}/{len(d)} wins)")
    if zs_bc:
        zs_bc = np.asarray(zs_bc)
        summary += (f"\nBC mean {zs_bc.mean():+.3f} | "
                    f"Δ(BC-baseline) {(zs_bc - zs_base).mean():+.3f} | "
                    f"Δ(policy-BC) {(zs_sac - zs_bc).mean():+.3f}")
    print(summary)
    print(f"final frames written {out_dir}")
    print("MEAN EVAL:", "PASS-learning works" if d.mean() > 0 else "WARN-suspected drift")


if __name__ == "__main__":
    main()
