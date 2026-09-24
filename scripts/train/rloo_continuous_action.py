# RLOO trainer — the project's only active RL engine. Era recipe: rm reward mode + zero extras.
# Based on CleanRL ppo_rloo_envpool.py; paper: Ahmadian et al. 2024, "Back to Basics: Revisiting
# REINFORCE Style Optimization for Learning from Human Feedback in LLMs".
#
# No critic: episodes are fixed at 2 steps with a terminal RM reward (r1 == 0), so V(s)/GAE add
# nothing. K episodes per start (reset_replay = verbatim replay: same product/view/jitter/inherit
# noise), advantage_i = R_i - mean(R_j, j != i) (leave-one-out in-group baseline, no network).
# PPO brakes kept: clip 0.2 + target-kl early stop + max-grad-norm + BC/init-std initialisation.
#
# Rollout is grouped by start: starts x K replays x 2 steps per update; envs are held directly
# (one per --num-envs, threads write disjoint slots); ckpts are named {stem}_u{update}.pt so
# segment evals never overwrite each other; ckpt layout matches eval_policy.py (ckpt["actor"]).
import argparse
import json
import random
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))  # import from same dir, same as eval_policy

from sac_continuous_action import (  # noqa: E402
    CONFIG_DIM, OBS_DIM, STATIC_DIM, Actor, PixelActor, load_bc_ckpt, make_env,
)


def get_logprob_entropy(actor, obs, action, return_det=False):
    """log_prob (with tanh Jacobian, same convention as Actor.get_action) and pre-tanh Normal
    entropy (no analytic entropy after squash; this approximates it, ent_coef ~= 0 so accuracy
    doesn't matter) for replayed actions under the current policy. return_det=True adds a third
    return = deterministic action (same convention as Actor.get_action's third return; used by
    the direct-anchor penalty to skip a second forward)."""
    mean, log_std = actor(obs)
    std = log_std.exp()
    normal = torch.distributions.Normal(mean, std)
    if getattr(actor, "residual", False):
    # residual mode: correction = squash-free Gaussian (mean clamped to +/-2); exact log_prob.
        mean = mean.clamp(-2.0, 2.0)
        normal = torch.distributions.Normal(mean, std)
        cfg = obs[:, STATIC_DIM:STATIC_DIM + action.shape[1]]
        det = (cfg + mean).clamp(-1.0, 1.0)
        delta = action - cfg
        out = (normal.log_prob(delta).sum(1), normal.entropy().sum(1))
        return (*out, det) if return_det else out
    a = (action - actor.action_bias) / actor.action_scale  # identity mapping; keep the formula in case it changes
    a = a.clamp(-1.0 + 1e-6, 1.0 - 1e-6)  # atanh numerical guard
    x_t = torch.atanh(a)
    log_prob = normal.log_prob(x_t)
    # tanh Jacobian (same as Actor.get_action; cancels in ratio, kept so both ends share convention)
    log_prob -= torch.log(actor.action_scale * (1 - torch.tanh(x_t).pow(2)) + 1e-6)
    out = (log_prob.sum(1), normal.entropy().sum(1))
    if return_det:
        det = torch.tanh(mean) * actor.action_scale + actor.action_bias
        return (*out, det)
    return out


def parse_args():
    p = argparse.ArgumentParser(description="RLOO 2-step correction loop (K samples per start + LOO baseline; see file header)",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--torch-deterministic", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--cuda", action=argparse.BooleanOptionalAction, default=True)
    # env (same protocol as ppo/sac: smoke synthetic self-test / blender real loop)
    p.add_argument("--env", type=str, default="smoke",
                   help="env name: smoke=synthetic self-test; blender=real loop (BlenderRefineEnv; port increments when num-envs>1, RM shared in-process)")
    p.add_argument("--rm-ckpt", type=Path, default=ROOT / "logs/ckpt/rm_v19c.pt",
                   help="RM checkpoint (used by --env blender; effective when --reward-mode rm, not loaded in seed mode)")
    p.add_argument("--reward-mode", choices=["rm", "seed", "render_emb", "render_emb_dd", "render_emb_goal", "pixel_goal"], default="rm",
                   help="rm = standard (RM score); seed = seed-distance compass (reward = -tanh-RMS(a2, own seed), "
                        "no RM, skips step-2 train render); render_emb = perceptual volumetric goal "
                        "(-||final render embed - seed render embed||, wide-basin version); "
                        "render_emb_dd/render_emb_goal/pixel_goal = archived variants")
    p.add_argument("--goal-cond", action="store_true",
                   help="goal conditioning (obs + 512-dim target-image embed; requires --reward-mode render_emb_dd)")
    p.add_argument("--neutral-ref", type=Path, default=None, help="npz of neutral-image embeds for render_emb_dd")
    p.add_argument("--pixel-obs", action="store_true",
                   help="pure-pixel 3-stack obs (config+clay+preview+goal, each 64x64 gray; PixelActor CNN trunk)")
    p.add_argument("--goal-images", type=Path, default=None,
                   help="directory of target render images for pixel_goal (ref_*.png; key = filename minus ref_ prefix)")
    p.add_argument("--reward-embedder", choices=["r18", "vgg"], default="r18",
                   help="embedder for the reward path (vgg = LPIPS backend family; obs representation unchanged)")
    p.add_argument("--seed-ref", type=Path, default=None,
                   help="npz of seed reference embeds for render_emb mode (precomputed, same convention as bc7s_gate)")
    p.add_argument("--port", type=int, default=17871, help="render worker TCP port (--env blender; increments when num-envs>1)")
    p.add_argument("--episodes-per-product", type=int, default=8,
                   help="episodes per product in a row, amortizing BVH warm-up on product switch (--env blender)")
    p.add_argument("--include-seedless", action="store_true",
                   help="merge products that have embeddings but no seeds into the training pool (reset via NN-inherit start, --env blender)")
    p.add_argument("--deck-weight-skus", default=None,
                   help="comma-separated sku list: repeat each --deck-weight times per deck (new-product oversampling; "
                        "seen products stay at 1 as anti-forgetting anchors; --env blender; skus must already be in the pool)")
    p.add_argument("--deck-weight", type=int, default=1,
                   help="repeat count per listed sku per deck (1 = no weighting; batch2 x2 -> 24/61 new products, x3 -> 36/73)")
    # RLOO grouped sampling
    p.add_argument("--num-starts", type=int, default=50,
                   help="starts per update; a start = fresh reset() draw (product/view/jitter/inherit noise)")
    p.add_argument("--rloo-k", type=int, default=8,
                   help="episodes replayed per start via reset_replay (>=2; LOO baseline group size)")
    p.add_argument("--total-timesteps", type=int, default=60000,
                   help="env-step budget; each update consumes num-starts*rloo-k*2 steps (default 50*8*2=800 steps/update -> 75 updates)")
    p.add_argument("--num-envs", type=int, default=1,
                   help="number of env instances (starts round-robin; num-envs>1 = real parallel rendering, "
                        "one resident worker per env, ports increment from --port)")
    # algorithm hyperparams (brakes identical to PPO)
    p.add_argument("--update-epochs", type=int, default=4)
    p.add_argument("--minibatch-size", type=int, default=100)
    p.add_argument("--clip-coef", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.001,
                   help="entropy bonus coefficient (0.01 climbs too fast late; planned 0.003)")
    p.add_argument("--policy-lr", type=float, default=1e-4, help="policy lr; the only lr once the critic is gone")
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--target-kl", type=float, default=0.2,
                   help="early-stop remaining epochs this update once approx_kl exceeds it; None disables")
    p.add_argument("--norm-adv", action=argparse.BooleanOptionalAction, default=False,
                   help="renormalize advantage across the batch; RLOO advantage is already group-centered, off by default")
    p.add_argument("--min-grp-std", type=float, default=0.05,
                   help="group reward std below this = no-information group (RM can't tell actions apart): "
                        "advantage zeroed, monitored only, no gradient (DAPO dynamic-sampling idea). "
                        "Monitor zero_var_frac (< threshold) vs low_var_frac (threshold..0.1, count-only): "
                        "low-up is the leading indicator of zero-up -> grp_std median halving")
    p.add_argument("--stop-after-warns", type=int, default=3,
                   help="soft-stop training after N consecutive collapse-warn updates (save final ckpt, exit); 0=off. "
                        "Segment ckpts are kept every 10 updates, so resuming after collapse wastes GPU")
    p.add_argument("--hidden-dim", type=int, default=256)
    # init / hot start
    p.add_argument("--bc-ckpt", type=Path, default=None,
                   help="BC checkpoint: init log_std per-dim from BC residuals + warm start (cold start)")
    p.add_argument("--resume-actor", type=Path, default=None,
                   help="PPO/RLOO ckpt: hot-start full actor weights (incl fc_logstd exploration radius), wins over --bc-ckpt")
    p.add_argument("--init-std", type=float, default=0.0,
                   help="if >0, override fc_logstd with this sigma (0.06 is the established starting value)")
    p.add_argument("--anchor-lambda", type=float, default=0.0,
                   help="reward anchor (taxes cross-product common drift): reward = RM z - lambda*d(a2, clean inherit start), "
                        "d = per-dim RMS in tanh space. lambda=0 = old convention; typical drift d~=0.63, lambda=0.25 -> ~0.16z penalty")
    p.add_argument("--anchor-d-star", type=float, default=0.0,
                   help="direct-anchor penalty (distinct mechanism from --anchor-lambda): not in reward, not in LOO "
                        "baseline; differentiable penalty lambda*d(tanh mu, clean inherit start) on the step-2 "
                        "deterministic mean action — level not absorbed by the baseline, zero sampling variance. "
                        "lambda via dual ascent (SAC-temperature-style controller): raise when measured d > d*, "
                        "relax otherwise. 0=off. d* scale: human-human LSO 0.183 / inherit->human final 0.271 / collapsed 0.67")
    p.add_argument("--anchor-lr", type=float, default=0.1, help="dual-ascent step size for log lambda (per update)")
    p.add_argument("--anchor-lambda-init", type=float, default=0.1, help="initial lambda")
    p.add_argument("--anchor-lambda-max", type=float, default=10.0, help="lambda cap (fuse)")
    p.add_argument("--residual", action="store_true",
                   help="residual parameterization (structural fix for fixed-lighting collapse): network outputs delta only "
                        "(squash-free Gaussian), action = obs config segment + delta. Requires a BC ckpt retrained with "
                        "a residual BC; weights same-shape-but-different-meaning vs absolute ckpts, mixing via "
                        "--resume-actor errors out")
    p.add_argument("--repel-alpha", type=float, default=0.0,
                   help="repulsion regularizer (global median hinge vs cross-product fixed-lighting collapse): not in "
                        "reward/baseline; penalizes median pairwise det distance across distinct-task start pairs in each "
                        "(view x step) subset, lambda*relu(alpha*M_base - M_det) — only cloud flattening taxed, per-product "
                        "free, common drift zero-penalty (M_base = median pairwise distance of clean inherit lights, "
                        "~0.94-1.0x human level). lambda dual ascent: log lambda += repel-lr*mean_hinge. 0=off. "
                        "Scale: collapsed 0.47x human, gate line 0.8x. Orthogonal to --anchor-d-star, not recommended together")
    p.add_argument("--repel-lr", type=float, default=0.1, help="repulsion lambda dual-ascent step (per update)")
    p.add_argument("--repel-lambda-init", type=float, default=0.1, help="repulsion lambda initial value")
    p.add_argument("--repel-lambda-max", type=float, default=10.0, help="repulsion lambda cap (fuse)")
    # logging / ckpt
    p.add_argument("--log-file", type=Path, default=None, help="jsonl log path (print only by default)")
    p.add_argument("--out", type=Path, default=None, help="save checkpoint at end (default logs/ckpt/rloo_actor[_smoke].pt)")
    p.add_argument("--log-every", type=int, default=1, help="log every N updates (default: every update)")
    p.add_argument("--ckpt-every", type=int, default=10,
                   help="save a separate {out_stem}_u{update}.pt every N updates (0 = only at end; segment re-eval snapshots, no overwrite)")
    p.add_argument("--smoke", action="store_true",
                   help="self-test: 4 small updates on SmokeRefineEnv; asserts loss has no NaN, grouped advantage correct, ckpt eval_policy-compatible")
    return p.parse_args()


def main():
    args = parse_args()
    assert args.rloo_k >= 2, "RLOO needs K>=2 for the leave-one-out baseline"
    assert args.deck_weight >= 1, "--deck-weight must be >=1 (1 = no weighting)"
    if args.smoke:
        # shrink groups/minibatch/epochs/net so 4 small updates suffice; logs+ckpt land in logs/ckpt/
        args.env = "smoke"
        args.num_starts, args.rloo_k = 4, 4
        args.total_timesteps = 4 * 4 * 4 * 2  # 4 updates x (4 starts x 4 replays x 2 steps)
        args.minibatch_size, args.update_epochs = 32, 2
        args.hidden_dim = 64
        args.log_file = args.log_file or ROOT / "logs/ckpt/rloo_smoke_log.jsonl"
    elif args.env == "smoke":
        # guard: a real launch missing --env blender used to idle 5 smoke updates before the collapse fuse tripped
        print("WARNING: --env smoke without --smoke: this is a synthetic no-render run with an almost constant reward; "
              "for real training pass --env blender explicitly. To confirm a smoke run, pass --smoke.", file=sys.stderr, flush=True)
        args.out = args.out or ROOT / "logs/ckpt/rloo_actor_smoke.pt"

    # TRY NOT TO MODIFY: seeding (CleanRL verbatim)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup: hold the env list directly (grouped replay needs per-env control, no
    # SyncVectorEnv); starts round-robin across envs. reset_replay protocol implemented by both smoke and blender envs
    if args.env == "blender" and args.num_envs > 1:
        print(f"multi-env true parallel: {args.num_envs} threads each bound to a persistent worker; "
              f"ports {args.port}..{args.port + args.num_envs - 1}")

    def _env_kwargs(i):
        base = {"anchor_lambda": args.anchor_lambda}
        if args.env != "blender":
            return base
        deck_weights = ({s.strip(): args.deck_weight
                         for s in args.deck_weight_skus.split(",") if s.strip()}
                        if args.deck_weight_skus else None)
        return {**base, "rm_ckpt": args.rm_ckpt,
                "seed_ref_embeds": args.seed_ref, "neutral_embeds": args.neutral_ref,
                "reward_embedder": args.reward_embedder, "goal_images": args.goal_images,
                "pixel_obs": args.pixel_obs, "no_jitter": args.pixel_obs,
                "reward_mode": args.reward_mode,
                "port": args.port + i,
                "episodes_per_product": args.episodes_per_product,
                "include_seedless": args.include_seedless,
                "deck_weights": deck_weights}

    envs = [make_env(args.env, args.seed + i, i, _env_kwargs(i))() for i in range(args.num_envs)]

    # atexit backstop: reap render workers on exception/Ctrl+C (a crashed run with live workers leaks GPU memory)
    import atexit
    atexit.register(lambda: [env.close() for env in envs])

    if args.pixel_obs:
        OBS = 72 + 3 * 64 * 64  # pure-pixel obs: config 72 + clay/preview/goal 3-stack 64x64 gray
        print(f"obs_dim = {OBS} (pixel-obs stacked-image CNN trunk)")
    else:
        goal_dim = 0
        if args.goal_cond and args.seed_ref is not None:
            _z = np.load(args.seed_ref)
            goal_dim = int(_z[_z.files[0]].shape[0])
        OBS = OBS_DIM + goal_dim
        print(f"obs_dim = {OBS}" + (f"（goal-cond +{goal_dim}）" if args.goal_cond else ""))
    if args.pixel_obs:
        actor = PixelActor(OBS, CONFIG_DIM, args.hidden_dim).to(device)
    else:
        actor = Actor(OBS, CONFIG_DIM, args.hidden_dim, residual=args.residual).to(device)
    policy_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    # init precedence: resume-actor (full hot start) > bc-ckpt (cold start) > random; init-std overrides last
    if args.resume_actor is not None:
        ck = torch.load(args.resume_actor, map_location="cpu", weights_only=True)
        assert bool(ck.get("residual", False)) == args.residual, \
            "hot-start ckpt and --residual disagree (residual and absolute weights look alike but mean different things; mixing them silently trains wrong)"
        actor.load_state_dict(ck["actor"])
        print(f"actor hot start <- {args.resume_actor} (including fc_logstd exploration radius)")
    elif args.bc_ckpt is not None:
        load_bc_ckpt(actor, args.bc_ckpt)  # init log_std per-dim from BC residuals + warm start when shapes match
    if args.init_std > 0:
        actor.init_log_std_from(np.full(CONFIG_DIM, args.init_std))

    steps_per_update = args.num_starts * args.rloo_k * 2  # episodes are fixed at 2 steps
    num_updates = max(1, args.total_timesteps // steps_per_update)
    n_eps = args.num_starts * args.rloo_k
    batch_size = n_eps * 2  # samples = one per step of each episode

    # ckpt aligned with ppo/sac: ckpt["actor"] loads straight into sac's Actor (eval_policy.py re-eval)
    out = args.out or ROOT / "logs/ckpt" / ("rloo_actor_smoke.pt" if args.smoke else "rloo_actor.pt")
    out.parent.mkdir(parents=True, exist_ok=True)

    def save_ckpt(path, update):
        torch.save({
            "actor": actor.state_dict(),
            "obs_dim": OBS, "action_dim": CONFIG_DIM, "hidden_dim": args.hidden_dim,
            "bc_ckpt": str(args.bc_ckpt) if args.bc_ckpt else None,
            "resume_actor": str(args.resume_actor) if args.resume_actor else None,
            "anchor_lambda": float(args.anchor_lambda),
            "reward_mode": args.reward_mode,
            "anchor_d_star": float(args.anchor_d_star),
            "repel_alpha": float(args.repel_alpha),
            "residual": bool(args.residual),
            "updates": update, "episodes": ep_cnt,
            "smoke": bool(args.smoke),
        }, path)

    start_time = time.time()

    log_f = None
    if args.log_file is not None:
        args.log_file.parent.mkdir(parents=True, exist_ok=True)
        log_f = open(args.log_file, "a")

    def log(rec):
        if log_f:
            log_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            log_f.flush()

    ep_cnt = 0
    global_step = 0
    update_hist = []
    zv_hist, gs_hist = [], []  # zero_var_frac / grp_std history: collapse-warning trend
    warn_streak = 0            # consecutive collapse-warning updates: --stop-after-warns soft stop
    # direct-anchor lambda controller state (dual ascent, persists across updates; only when --anchor-d-star>0)
    log_lam = float(np.log(args.anchor_lambda_init)) if args.anchor_d_star > 0 else None
    if args.anchor_d_star > 0:
        print(f"direct anchor regulariser: d*={args.anchor_d_star} (penalises step-2 mean actions only; never enters reward/baseline); "
              f"λ self-tuning init={args.anchor_lambda_init} max={args.anchor_lambda_max} "
              f"lr={args.anchor_lr}/update")
    # repulsion lambda controller state (same dual ascent; only when --repel-alpha>0)
    log_lam_repel = float(np.log(args.repel_lambda_init)) if args.repel_alpha > 0 else None
    rep_sub_total = 0  # lifetime valid-pair-subset count (smoke "median really computed" assertion)
    if args.repel_alpha > 0:
        print(f"repulsion regulariser: α={args.repel_alpha} (global median hinge; penalises only flattened cross-product spread, never enters reward/baseline); "
              f"λ self-tuning init={args.repel_lambda_init} max={args.repel_lambda_max} "
              f"lr={args.repel_lr}/update")
    for update in range(1, num_updates + 1):
    # ---- grouped on-policy rollout: one reset per start, then K reset_replay replays ----
    # One thread per env; env i handles starts with s % num_envs == i. Slot ei = s*rloo_k + k is
    # preassigned, so threads write disjoint buffers; actor forward is serialized under a lock
    # (milliseconds on GPU); num_envs=1 degrades to the serial path.
        ep_rewards = np.zeros(n_eps, np.float64)
        obs_buf = torch.zeros((n_eps, 2, OBS), device=device)     # per-episode obs for both steps
        act_buf = torch.zeros((n_eps, 2, CONFIG_DIM), device=device)
        logp_buf = torch.zeros((n_eps, 2), device=device)
        base_buf = np.zeros((n_eps, CONFIG_DIM), np.float32) \
            if (args.anchor_d_star > 0 or args.repel_alpha > 0) else None
        # repulsion pair metadata (per start): view + task id (sku map; same sku+view = same task, excluded)
        view_buf = np.zeros(args.num_starts, np.int8) if args.repel_alpha > 0 else None
        task_buf = np.zeros(args.num_starts, np.int32) if args.repel_alpha > 0 else None
        task_ids = {}  # sku -> task id (rebuilt per update)
        VIEW_ID = {"front": 0, "high": 1}

        actor_lock = threading.Lock()
        log_lock = threading.Lock()
        rollout_errors = []

        def rollout_start(s):
            """Handle the K episodes of start s (logic verbatim from the old serial loop)."""
            nonlocal ep_cnt, global_step
            env = envs[s % args.num_envs]  # starts round-robin across envs
            obs1, reset_info = env.reset()  # fresh start (reroll product/view/jitter/inherit noise)
            if base_buf is not None:
                base_buf[s * args.rloo_k:(s + 1) * args.rloo_k] = \
                    np.asarray(reset_info["base_tanh"], np.float32)  # all K episodes of this start share the anchor
            if view_buf is not None:
                assert reset_info.get("view") in VIEW_ID and reset_info.get("sku") is not None, \
                    f"with --repel-alpha the env must provide sku/view (reset_info={reset_info})"
                with log_lock:  # task_ids shared across threads: id assignment must be atomic (len+setdefault races)
                    tid = task_ids.setdefault(reset_info["sku"], len(task_ids))
                view_buf[s] = VIEW_ID[reset_info["view"]]  # slots preassigned per start; array writes are race-free
                task_buf[s] = tid
            for k in range(args.rloo_k):
                if k > 0:
                    obs1, _ = env.reset_replay()  # verbatim replay of the same start
                with actor_lock, torch.no_grad():
                    a1, lp1, _ = actor.get_action(torch.as_tensor(obs1, dtype=torch.float32,
                                                                  device=device).unsqueeze(0))
                obs2, r1, term1, _, _ = env.step(a1.squeeze(0).cpu().numpy())
                assert not term1, "episode step 1 must not terminate"
                with actor_lock, torch.no_grad():
                    a2, lp2, _ = actor.get_action(torch.as_tensor(obs2, dtype=torch.float32,
                                                                  device=device).unsqueeze(0))
                _, r2, term2, _, info = env.step(a2.squeeze(0).cpu().numpy())
                assert term2, "episode step 2 must terminate"

                ei = s * args.rloo_k + k  # preassigned slot: this start owns [s*K, (s+1)*K)
                obs_buf[ei, 0] = torch.as_tensor(obs1, dtype=torch.float32, device=device)
                act_buf[ei, 0] = a1.squeeze(0)
                logp_buf[ei, 0] = lp1.flatten()[0]
                obs_buf[ei, 1] = torch.as_tensor(obs2, dtype=torch.float32, device=device)
                act_buf[ei, 1] = a2.squeeze(0)
                logp_buf[ei, 1] = lp2.flatten()[0]
                ep_rewards[ei] = r1 + r2

                with log_lock:
                    global_step += 2  # old impl +1 per step; only used at episode end, adding 2 once is equivalent
                    ep_cnt += 1
                    rec = {"type": "episode", "global_step": global_step,
                           "episodic_return": float(ep_rewards[ei]), "episodic_length": 2,
                           "rm_score": float(info.get("rm_score", float("nan"))),
                           "anchor_pen": float(info.get("anchor_pen", 0.0)),
                           "sku": reset_info.get("sku"), "view": reset_info.get("view"),
                           "inherit_from": reset_info.get("inherit_from"),
                           "goal": info.get("goal"), "seed_dist": info.get("seed_dist"),
                           "dd_dist": info.get("dd_dist")}  # passthrough of goal-mode reward fields (stratified analysis)
                    print(f"global_step={global_step}, episodic_return={ep_rewards[ei]:.3f}")
                    log(rec)

        def rollout_worker(i):
            try:
                for s in range(i, args.num_starts, args.num_envs):  # s % num_envs == i
                    rollout_start(s)
            except BaseException as e:  # collect in-thread exceptions (worker crash etc.), rethrow on main thread
                rollout_errors.append(e)

        threads = [threading.Thread(target=rollout_worker, args=(i,),
                                    name=f"rollout-env{i}")
                   for i in range(args.num_envs)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        if rollout_errors:
            raise rollout_errors[0]

        # ---- RLOO advantage: in-group leave-one-out baseline, broadcast to both steps of the episode ----
        # low-variance group filter (--min-grp-std): tiny-std groups contribute no gradient (free gradient SNR)
        grp_stds = np.array([ep_rewards[s * args.rloo_k:(s + 1) * args.rloo_k].std()
                             for s in range(args.num_starts)])
        zero_var = grp_stds < args.min_grp_std          # truly no-information groups: filtered
        low_var = (grp_stds >= args.min_grp_std) & (grp_stds < 0.1)  # counted only, not filtered
        adv = np.zeros(n_eps, np.float64)
        for s in range(args.num_starts):
            if zero_var[s]:
                continue  # advantage stays 0: still monitored, no gradient
            g = ep_rewards[s * args.rloo_k:(s + 1) * args.rloo_k]
            loo_mean = (g.sum() - g) / (args.rloo_k - 1)  # mean of the other K-1
            adv[s * args.rloo_k:(s + 1) * args.rloo_k] = g - loo_mean
        b_advantages = torch.as_tensor(adv, dtype=torch.float32, device=device) \
            .unsqueeze(1).expand(n_eps, 2).reshape(-1)

        # flatten the batch (2 steps -> 2*n_eps samples)
        b_obs = obs_buf.reshape(-1, OBS)
        b_actions = act_buf.reshape(-1, CONFIG_DIM)
        b_logprobs = logp_buf.reshape(-1)
        # direct anchor: flattened row i = step i%2 of episode i//2; anchor config expanded per episode, step-2 only
        base_flat = step2_mask = None
        if args.anchor_d_star > 0:
            base_flat = torch.as_tensor(base_buf, device=device).repeat_interleave(2, dim=0)
            step2_mask = torch.arange(batch_size, device=device) % 2 == 1
        d_sum, d_cnt = 0.0, 0  # measured drift this update (controller input)
        # repulsion: per-update precompute of start-level pairwise anchor distances (constant, no grad) + pair metadata to device
        d_base_pair = view_t = task_t = None
        if args.repel_alpha > 0:
            starts_base = torch.as_tensor(base_buf[::args.rloo_k], device=device)  # (S,72)
            d_base_pair = torch.sqrt(((starts_base.unsqueeze(0) - starts_base.unsqueeze(1)) ** 2)
                                     .mean(dim=-1) + 1e-12)  # (S,S)
            view_t = torch.as_tensor(view_buf, device=device)
            task_t = torch.as_tensor(task_buf, device=device)
        rep_h_sum, rep_h_cnt = 0.0, 0    # subset hinge accumulator (controller error)
        rep_r_sum, rep_r_cnt = 0.0, 0    # retention M_det/M_base accumulator (readout; M_base~=0 subsets excluded)

        # ---- RLOO update: clipped surrogate (PPO brakes), no critic ----
        b_inds = np.arange(batch_size)
        kl_stop = False
        grad_steps = 0
        agg = {"pg_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0, "clipfrac": 0.0}
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, args.minibatch_size):
                mb_inds = b_inds[start:start + args.minibatch_size]

                need_det = base_flat is not None or d_base_pair is not None
                if need_det:
                    newlogprob, ent_est, det = get_logprob_entropy(
                        actor, b_obs[mb_inds], b_actions[mb_inds], return_det=True)
                else:
                    newlogprob, ent_est = get_logprob_entropy(actor, b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                with torch.no_grad():
                    # k3 unbiased KL estimate (exp(logr)-logr-1); do NOT use mean(logr) — can go negative, under-reports
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfrac = ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()

                mb_adv = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                # Policy loss (clipped surrogate, CleanRL verbatim)
                pg_loss = torch.max(-mb_adv * ratio,
                                    -mb_adv * torch.clamp(ratio, 1 - args.clip_coef,
                                                          1 + args.clip_coef)).mean()
                entropy_loss = ent_est.mean()
                loss = pg_loss - args.ent_coef * entropy_loss
                if base_flat is not None:
                    # direct-anchor penalty: lambda * d(tanh mu, anchor) on step-2 mean action
                    # (differentiable, zero sampling variance, bypasses LOO baseline — the reward route dies by baseline absorption)
                    m2 = step2_mask[mb_inds]
                    if m2.any():
                        d_mb = torch.sqrt(((det[m2] - base_flat[mb_inds][m2]) ** 2)
                                          .mean(dim=1) + 1e-12)
                        loss = loss + float(np.exp(log_lam)) * d_mb.mean()
                        d_sum += float(d_mb.detach().sum())
                        d_cnt += int(m2.sum())
                if d_base_pair is not None:
    # Repulsion regulariser (--repel-alpha, archived): median-hinge penalty on pairwise distances
    # across distinct-task starts; only flattened cross-product spread is penalised. Off by default.
                    s_ids = torch.as_tensor((mb_inds // 2) // args.rloo_k, device=device)
                    step_ids = torch.as_tensor(mb_inds % 2, device=device)
                    v_row, t_row = view_t[s_ids], task_t[s_ids]
                    pen, n_sub = None, 0
                    for v in (0, 1):
                        for st in (0, 1):
                            sub = (v_row == v) & (step_ids == st)
                            if int(sub.sum()) < 6:
                                continue
                            sid_sub, tid_sub = s_ids[sub], t_row[sub]
                            pm = (tid_sub.unsqueeze(0) != tid_sub.unsqueeze(1)) & torch.triu(
                                torch.ones(det[sub].shape[0], det[sub].shape[0],
                                           dtype=torch.bool, device=device), diagonal=1)
                            if int(pm.sum()) < 8:
                                continue
                            det_sub = det[sub]
                            d_det = torch.sqrt(((det_sub.unsqueeze(0) - det_sub.unsqueeze(1)) ** 2)
                                               .mean(dim=-1) + 1e-12)
                            m_b = d_base_pair[sid_sub][:, sid_sub][pm].median()
                            m_d = d_det[pm].median()
                            hinge = torch.relu(args.repel_alpha * m_b - m_d)
                            pen = hinge if pen is None else pen + hinge
                            n_sub += 1
                            rep_h_sum += float(hinge.detach())
                            rep_h_cnt += 1
                            if float(m_b) > 0.05:  # subsets with M_base~0 carry no spread info, excluded from retention readout
                                rep_r_sum += float((m_d / (m_b + 1e-8)).detach())
                                rep_r_cnt += 1
                    if n_sub > 0:
                        loss = loss + float(np.exp(log_lam_repel)) * (pen / n_sub)
                policy_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), args.max_grad_norm)
                policy_optimizer.step()

                grad_steps += 1
                agg["pg_loss"] += pg_loss.item()
                agg["entropy"] += entropy_loss.item()
                agg["approx_kl"] += approx_kl.item()
                agg["clipfrac"] += clipfrac
                if args.target_kl is not None and approx_kl.item() > args.target_kl:
                    kl_stop = True  # KL over threshold: early-stop remaining epochs this update (minibatches too)
                    break
            if kl_stop:
                break

        # direct-anchor controller (dual ascent): raise lambda when measured d exceeds d*, relax
        # otherwise; the new lambda applies from the next update
        d_meas = lam_used = None
        if args.anchor_d_star > 0:
            d_meas = d_sum / max(d_cnt, 1)
            lam_used = float(np.exp(log_lam))
            log_lam = float(np.clip(log_lam + args.anchor_lr * (d_meas - args.anchor_d_star),
                                    np.log(1e-3), np.log(args.anchor_lambda_max)))
        # repulsion controller (same dual ascent): bigger measured mean hinge -> higher lambda;
        # when the constraint is met (hinge->0) lambda falls back to the floor — the error IS the penalty
        rep_hinge = rep_ratio = rep_lam_used = None
        if args.repel_alpha > 0:
            rep_hinge = rep_h_sum / max(rep_h_cnt, 1)
            rep_ratio = rep_r_sum / max(rep_r_cnt, 1)
            rep_lam_used = float(np.exp(log_lam_repel))
            rep_sub_total += rep_h_cnt
            log_lam_repel = float(np.clip(log_lam_repel + args.repel_lr * rep_hinge,
                                          np.log(1e-3), np.log(args.repel_lambda_max)))

        if update % args.log_every == 0:
            n = max(grad_steps, 1)
            grp_std = float(grp_stds.mean())
            zero_var_frac = float(zero_var.mean())
            low_var_frac = float(low_var.mean())
    # Collapse warning (trend-based; low grp_std late in convergence is a reward, not an accident):
    #   1. zero_var_frac >= 0.5 for 3 consecutive updates (sampling largely wasted)
    #   2. grp_std median halved (last 10 rounds vs prior 10) — armed only past half the run so
    #      natural early decay does not false-alarm; deployment best-of-16 needs a live sigma.
            zv_hist.append(zero_var_frac)
            gs_hist.append(grp_std)
            collapse_warn = len(zv_hist) >= 3 and all(v >= 0.5 for v in zv_hist[-3:])
            if update > num_updates / 2 and len(gs_hist) >= 20:
                recent, prior = np.median(gs_hist[-10:]), np.median(gs_hist[-20:-10])
                collapse_warn = collapse_warn or bool(recent < prior / 2)
            rec = {"type": "update", "update": update, "global_step": global_step,
                   "pg_loss": agg["pg_loss"] / n, "entropy": agg["entropy"] / n,
                   "approx_kl": agg["approx_kl"] / n, "clipfrac": agg["clipfrac"] / n,
                   "kl_stop": kl_stop,
                   "rew_mean": float(ep_rewards.mean()), "rew_std": float(ep_rewards.std()),
                   "grp_std": grp_std, "adv_std": float(adv.std()),
                   "zero_var_frac": zero_var_frac, "low_var_frac": low_var_frac,
                   "collapse_warn": bool(collapse_warn),
                   "sps": int(global_step / (time.time() - start_time))}
            if args.residual:
    # Delta monitoring (residual mode): sat = fraction of |delta| >= 1.5, the precursor of the
    # +/-2 mean hard bound (early signal of the delta-saturation drift channel).
                dlt = (act_buf - obs_buf[..., STATIC_DIM:STATIC_DIM + CONFIG_DIM]).abs().flatten()
                rec["delta_p50"] = float(dlt.quantile(0.5))
                rec["delta_p95"] = float(dlt.quantile(0.95))
                rec["delta_sat"] = float((dlt >= 1.5).double().mean())
            if d_meas is not None:
                rec["anchor_d"] = float(d_meas)       # measured this update: drift of step-2 mean action to anchor
                rec["anchor_lam"] = float(lam_used)   # lambda actually used this update (controller's new value applies next update)
            if rep_hinge is not None:
                rec["rep_hinge"] = float(rep_hinge)   # measured: subset mean hinge (controller error, RMS units)
                rec["rep_lam"] = float(rep_lam_used)  # repulsion lambda actually used this update (new value next update)
                rec["rep_ratio"] = float(rep_ratio)   # retention = subset mean of M_det/M_base (target = repel-alpha)
            update_hist.append(rec)
            print(f"update={update}/{num_updates} pg_loss={rec['pg_loss']:.4f} "
                  f"entropy={rec['entropy']:.3f} kl={rec['approx_kl']:.4f} "
                  f"rew={rec['rew_mean']:+.3f}±{rec['rew_std']:.3f} grp_std={grp_std:.3f} "
                  f"zero/low={zero_var_frac:.2f}/{low_var_frac:.2f}"
                  f"{' [collapse warning]' if collapse_warn else ''}"
                  + (f" δp95={rec['delta_p95']:.3f} sat={rec['delta_sat']:.3f}" if args.residual else "")
                  + (f" anchor_d={rec['anchor_d']:.3f} λ={rec['anchor_lam']:.2f}" if d_meas is not None else "")
                  + (f" repel_h={rec['rep_hinge']:.3f} λ={rec['rep_lam']:.2f} keep={rec['rep_ratio']:.2f}"
                     if rep_hinge is not None else "")
                  + f" SPS={rec['sps']}")
            log(rec)
            # soft stop: --stop-after-warns consecutive warned updates (segment ckpts already on disk, no waste)
            warn_streak = warn_streak + 1 if collapse_warn else 0
            if 0 < args.stop_after_warns <= warn_streak:
                print(f"collapse warning triggered {warn_streak} rounds in a row -> soft stop (ckpts for finished rounds are all on disk)")
                log({"type": "soft_stop", "update": update, "warn_streak": warn_streak})
                break
        if args.ckpt_every > 0 and update % args.ckpt_every == 0:
            # unique names, no overwrite (rolling latest loses mid-run re-eval points)
            save_ckpt(out.with_name(f"{out.stem}_u{update}.pt"), update)

    for env in envs:
        env.close()
    if log_f:
        log_f.close()

    save_ckpt(out, update)
    print(f"checkpoint saved {out} (updates={update}, episodes={ep_cnt})")

    if args.smoke:
        assert update == num_updates, f"update count {update} != {num_updates}"
        assert update_hist, "no log records for the training segment"
        for rec in update_hist:
            for k in ("pg_loss", "entropy", "approx_kl", "clipfrac", "rew_mean", "grp_std",
                      "zero_var_frac", "low_var_frac"):
                assert np.isfinite(rec[k]), f"{k} is NaN/Inf: {rec}"
            if args.residual:
                for k in ("delta_p50", "delta_p95", "delta_sat"):
                    assert np.isfinite(rec[k]), f"{k} is NaN/Inf: {rec}"
        # buffer shape assertions (grouped on-policy storage = (n_eps, 2, ...))
        assert obs_buf.shape == (n_eps, 2, OBS)
        assert act_buf.shape == (n_eps, 2, CONFIG_DIM)
        assert rew_sum_check(ep_rewards, n_eps), "episode return records missing"
        assert global_step == num_updates * steps_per_update
        # LOO advantage sums to ~0 within a group (identity: sum(g_i - mean_!=i) = 0)
        for s in range(args.num_starts):
            g = adv[s * args.rloo_k:(s + 1) * args.rloo_k]
            assert abs(g.sum()) < 1e-6, f"LOO advantages do not sum to 0 within a group: {g.sum()}"
        if args.anchor_d_star > 0:
            # direct anchor: finite readouts + correct controller direction (lambda must rise monotonically when d exceeds target, unless d already compliant)
            ds = [r["anchor_d"] for r in update_hist]
            ls = [r["anchor_lam"] for r in update_hist]
            assert np.isfinite(ds).all() and np.isfinite(ls).all(), f"direct-anchor readings invalid: {list(zip(ds, ls))}"
            assert ls[-1] > ls[0] or ds[-1] <= args.anchor_d_star, \
                f"λ controller direction wrong: d {ds[0]:.3f}->{ds[-1]:.3f} (d*={args.anchor_d_star}), " \
                f"λ {ls[0]:.3f}->{ls[-1]:.3f} unchanged"
        if args.repel_alpha > 0:
    # Repulsion: finite readouts, valid pairs actually computed, controller direction correct.
            hs = [r["rep_hinge"] for r in update_hist]
            ls = [r["rep_lam"] for r in update_hist]
            rs = [r["rep_ratio"] for r in update_hist]
            assert np.isfinite(hs).all() and np.isfinite(ls).all() and np.isfinite(rs).all(), \
                f"repulsion regulariser readings invalid: {list(zip(hs, ls, rs))}"
            assert rep_sub_total > 0, "repulsion regulariser never produced a valid pair (empty pair subset, no median computed)"
            assert ls[-1] > ls[0] or hs[-1] <= 1e-12, \
                f"repulsion λ controller direction wrong: hinge {hs[0]:.3f}->{hs[-1]:.3f}, " \
                f"λ {ls[0]:.3f}->{ls[-1]:.3f} did not rise"
        # ckpt structurally compatible with eval_policy.py: loads into sac Actor, get_action third return tanh(mean)
        ck = torch.load(out, map_location="cpu", weights_only=True)
        a2 = Actor(ck["obs_dim"], ck["action_dim"], ck["hidden_dim"], residual=ck.get("residual", False))
        a2.load_state_dict(ck["actor"])
        act, lp, mean = a2.get_action(torch.zeros(1, OBS))
        assert act.shape == (1, CONFIG_DIM) and torch.isfinite(lp).all() and torch.isfinite(mean).all()
        if args.residual:
            # identity start: zeroing fc_mean -> deterministic action = obs config segment verbatim (zero correction = noisy-inherit passthrough = baseline arm)
            a3 = Actor(OBS, CONFIG_DIM, args.hidden_dim, residual=True)
            with torch.no_grad():
                a3.fc_mean.weight.zero_()
                a3.fc_mean.bias.zero_()
            probe = torch.rand(8, OBS) * 2 - 1
            _, _, m = a3.get_action(probe)
            assert torch.allclose(m, probe[:, STATIC_DIM:STATIC_DIM + CONFIG_DIM].clamp(-1, 1),
                                  atol=1e-6), "residual identity start does not hold (zero correction should reproduce the obs config segment)"
        print(f"RLOO SMOKE: PASS ({update} updates / {ep_cnt} episodes, "
              f"last pg_loss={update_hist[-1]['pg_loss']:.4f} rew={update_hist[-1]['rew_mean']:+.3f})")


def rew_sum_check(ep_rewards, n_eps):
    return ep_rewards.shape == (n_eps,) and np.isfinite(ep_rewards).all()


if __name__ == "__main__":
    main()
