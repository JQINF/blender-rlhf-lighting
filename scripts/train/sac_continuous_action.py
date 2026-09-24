# Vendored from CleanRL: cleanrl/sac_continuous_action.py (commit fe8d8a03, master @ 2026-04-20),
# adapted for this project; algorithm body and tanh Jacobian correction kept verbatim.
#
# This file is the shared model/env library: Actor / PixelActor / load_bc_ckpt / make_env /
# RefineEnv protocol / dimension constants. The active RL trainer is rloo_continuous_action.py;
# main() below is archived and off the shared path.
#
# Key adaptations: gamma=1.0 (2-step loop); action space is the [-1,1]^72 tanh space;
# obs = [static 2056 | config 72 | render embedding 512] with the embedding segment zero-filled
# at step 1; no wandb/tensorboard/tyro; --bc-ckpt initialises log_std from BC residuals;
# gymnasium autoreset disabled (manual reset after termination).
import argparse
import json
import random
import time
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gymnasium.spaces import Box
from gymnasium.vector import AutoresetMode, SyncVectorEnv

ROOT = Path(__file__).resolve().parents[2]

STATIC_DIM = 2056   # cached static state (clay 3 views 3×512 + neutral-light 512 + bbox 3 + is_grounded 1 + view one-hot 2 + camera-jitter sin/cos 2)
CONFIG_DIM = 72     # action = full 8-slot × 9-dim config in tanh space (full output each step, not incremental)
EMB_DIM = 512       # current render embedding
OBS_DIM = STATIC_DIM + CONFIG_DIM + EMB_DIM  # 2640; step-1 zero-fills the emb segment (actual input 2128)


def parse_args():
    p = argparse.ArgumentParser(description="SAC 2-step refine loop (vendored from CleanRL, see file header)",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--torch-deterministic", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--cuda", action=argparse.BooleanOptionalAction, default=True)
    # Algorithm hyperparameters
    p.add_argument("--env", type=str, default="smoke",
                   help="env name: smoke = synthetic self-test; blender = real BlenderRefineEnv loop (num-envs 1 only)")
    p.add_argument("--rm-ckpt", type=Path, default=ROOT / "logs/ckpt/rm_v19c.pt",
                   help="RM checkpoint (--env blender)")
    p.add_argument("--port", type=int, default=17871, help="render worker TCP port (--env blender)")
    p.add_argument("--episodes-per-product", type=int, default=8,
                   help="episodes per product in a row, amortizing per-product BVH warm-up cost (--env blender)")
    p.add_argument("--total-timesteps", type=int, default=3400,
                   help="= learning_starts 400 + updates 3000")
    p.add_argument("--num-envs", type=int, default=1)
    p.add_argument("--buffer-size", type=int, default=100000,
                   help="obs is 2640-dim float32, 100k entries ≈ 2.2G RAM (do not raise on a 15G-RAM machine)")
    p.add_argument("--gamma", type=float, default=1.0, help="2-step short-horizon loop, gamma = 1.0 (plan §1)")
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--learning-starts", type=int, default=400, help="buffer fill phase (random actions)")
    p.add_argument("--critic-warmup-updates", type=int, default=400,
                   help="critic warmup: first N updates train critic only, no actor/alpha (300-500 recommended)")
    p.add_argument("--policy-lr", type=float, default=1e-4)
    p.add_argument("--q-lr", type=float, default=3e-4)
    p.add_argument("--policy-frequency", type=int, default=2)
    p.add_argument("--target-network-frequency", type=int, default=1)
    p.add_argument("--alpha", type=float, default=0.2, help="fixed entropy coefficient (used with --no-autotune)")
    p.add_argument("--critic-ln", action=argparse.BooleanOptionalAction, default=True,
                   help="LayerNorm on the critic (default on; guards against Q-overestimation exploits)")
    p.add_argument("--autotune", action=argparse.BooleanOptionalAction, default=True,
                   help="auto-tune the entropy coefficient; target entropy = −72 = −prod(action_dim)")
    p.add_argument("--hidden-dim", type=int, default=256)
    # project-specific additions
    p.add_argument("--bc-ckpt", type=Path, default=None,
                   help="BC checkpoint: log_std init per-dim from BC residuals (warm-starts the actor when shapes match)")
    p.add_argument("--bc-reg-lambda", type=float, default=1.0,
                   help="BC regularization strength λ₀: linear decay over the first 10%% updates; 0 = off (requires --bc-ckpt with matching dims)")
    p.add_argument("--bc-reg-floor", type=float, default=0.0,
                   help="λ decay floor (fraction of λ₀): >0 keeps BC anchoring for the whole run — with λ at 0,"
                        " pure SAC exploits critic overestimation and drifts the policy (mean-policy z 0.83 below baseline)")
    p.add_argument("--log-file", type=Path, default=None, help="jsonl log path (print only by default)")
    p.add_argument("--out", type=Path, default=None, help="save checkpoint at exit (default logs/ckpt/sac_actor[_smoke].pt)")
    p.add_argument("--log-every", type=int, default=100, help="log every N updates")
    p.add_argument("--smoke", action="store_true",
                   help="self-test: SmokeRefineEnv runs a few dozen updates, asserts losses have no NaN")
    return p.parse_args()


# ---------- Inline ReplayBuffer (replaces cleanrl_utils.buffers.ReplayBuffer) ----------

class ReplayBuffer:
    """Minimal ring buffer: numpy storage, moved to device when sampling; samples with (capacity, n_envs, ...) flattened."""

    def __init__(self, capacity, obs_shape, action_shape, n_envs, seed, device):
        self.capacity, self.n_envs, self.device = capacity, n_envs, device
        self.obs = np.zeros((capacity, n_envs, *obs_shape), np.float32)
        self.next_obs = np.zeros_like(self.obs)
        self.actions = np.zeros((capacity, n_envs, *action_shape), np.float32)
        self.rewards = np.zeros((capacity, n_envs), np.float32)
        self.dones = np.zeros((capacity, n_envs), np.float32)
        self.pos, self.full = 0, False
        self.rng = np.random.default_rng(seed)

    def add(self, obs, next_obs, actions, rewards, dones):
        p = self.pos
        self.obs[p], self.next_obs[p], self.actions[p] = obs, next_obs, actions
        self.rewards[p], self.dones[p] = rewards, dones
        self.pos = (p + 1) % self.capacity
        self.full = self.full or self.pos == 0

    def __len__(self):
        return (self.capacity if self.full else self.pos) * self.n_envs

    def sample(self, batch_size):
        n = self.capacity if self.full else self.pos
        i, j = np.divmod(self.rng.integers(0, n * self.n_envs, size=batch_size), self.n_envs)
        g = lambda a: torch.as_tensor(a[i, j], device=self.device)
        return SimpleNamespace(observations=g(self.obs), next_observations=g(self.next_obs),
                               actions=g(self.actions), rewards=g(self.rewards).view(-1),
                               dones=g(self.dones).view(-1))


# ---------- 2-step refine-loop env protocol ----------

class RefineEnv(gym.Env):
    """2-step refine-loop env protocol. episode = single (product, view); no cross-view averaging (plan §1).

    obs is uniformly (2640,) float32 = [static 2056 | current config 72 | render emb 512];
    action = (72,) ∈ [-1,1] tanh-space full config (full output each step, not incremental).
    Timeline:
      reset()  → obs1 = [static | inherited config | 0×512]   (inherited config = NN-inherit / template fallback + noise)
      step(a1) → env renders an 8spp preview + frozen-backbone embedding
               → obs2 = [static | step-1 config a1 | preview emb 512], reward 0, terminated False
      step(a2) → final render (256²@25spp+OIDN) → RM score = terminal reward
               → obs arbitrary (terminal state), terminated True, info["rm_score"] = RM score
    The real implementation (Blender rendering + RM scoring) is refine_env.BlenderRefineEnv; this class pins the interface only.
    """

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = Box(low=-1.0, high=1.0, shape=(CONFIG_DIM,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        raise NotImplementedError("RefineEnv is a protocol base class; the real implementation is the Blender closed loop; use SmokeRefineEnv for self-test")

    def step(self, action):
        raise NotImplementedError("RefineEnv is a protocol base class; the real implementation is the Blender closed loop; use SmokeRefineEnv for self-test")


class SmokeRefineEnv(RefineEnv):
    """Synthetic self-test env: no Blender/RM/rendering; correct shapes with a bounded learnable reward
    (terminal reward = 1 − mean(a₂²)). static/inherited config randomized per episode;
    preview emb = deterministic function of (static, step-1 action)."""

    def __init__(self, anchor_lambda=0.0):
        super().__init__()
        self._t = 0
        self.anchor_lambda = float(anchor_lambda)  # same as BlenderRefineEnv: reward minus λ·d(a₂, starting inherit)
        self._static = np.zeros(STATIC_DIM, np.float32)
        self._ep_idx = 0  # reset counter: derives sku/view (for repel pairing), deterministic, uses no np_random

    @staticmethod
    def _pack(static, config, emb):
        return np.concatenate([static, config, emb]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        gym.Env.reset(self, seed=seed)  # NOTE: cannot super().reset() — RefineEnv.reset raises directly
        self._t = 0
        n = self._ep_idx
        self._ep_idx += 1
        self._sku = f"smoke_{n % 3}"
        self._view = "front" if n % 2 == 0 else "high"
        self._static = (self.np_random.standard_normal(STATIC_DIM) * 0.5).astype(np.float32)
        self._inherit = self.np_random.uniform(-1.0, 1.0, CONFIG_DIM).astype(np.float32)
        return self._pack(self._static, self._inherit, np.zeros(EMB_DIM, np.float32)), \
            {"base_tanh": self._inherit.copy(),  # direct-anchor regularization reads the anchor config (smoke anchor = inherit itself)
             "sku": self._sku, "view": self._view}  # repel-pairing fields (same field names as BlenderRefineEnv)

    def reset_replay(self):
        """RLOO same-startpoint replay (same protocol as BlenderRefineEnv): returns to the last reset's starting point."""
        self._t = 0
        return self._pack(self._static, self._inherit, np.zeros(EMB_DIM, np.float32)), \
            {"replay": True, "base_tanh": self._inherit.copy(),
             "sku": self._sku, "view": self._view}

    def step(self, action):
        a_raw = np.asarray(action, np.float32).reshape(-1)  # anchor reads raw (same convention as BlenderRefineEnv)
        a = np.clip(a_raw, -1.0, 1.0)
        assert a.shape == (CONFIG_DIM,), f"action dims {a.shape}, expected ({CONFIG_DIM},)"
        if self._t == 0:
            # step-1: pseudo preview embedding (deterministic, bounded, gives the critic a learnable signal)
            self._t = 1
            tiled = np.tile(a, int(np.ceil(EMB_DIM / CONFIG_DIM)))[:EMB_DIM]
            emb = np.sin(2.0 * self._static[:EMB_DIM] + np.pi * tiled).astype(np.float32)
            return self._pack(self._static, a, emb), 0.0, False, False, {}
        # step-2: pseudo-RM terminal reward (+ anchor term matching BlenderRefineEnv, for --anchor-lambda self-tests)
        self._t = 0
        reward = float(1.0 - np.mean(a ** 2))
        pen = self.anchor_lambda * float(np.linalg.norm(a_raw - self._inherit) / np.sqrt(CONFIG_DIM))
        return (np.zeros(OBS_DIM, np.float32), reward - pen, True, False,
                {"rm_score": reward, "anchor_pen": pen})


def make_env(name, seed, idx, env_kwargs=None):
    def thunk():
        if name == "smoke":
            env = SmokeRefineEnv(**(env_kwargs or {}))
        elif name == "blender":
            # lazy import (same directory): keeps the smoke path free of torchvision/transformers/Blender-subprocess deps
            from refine_env import BlenderRefineEnv
            env = BlenderRefineEnv(seed=seed, **(env_kwargs or {}))
        else:
            raise NotImplementedError(f"env '{name}' not registered (options: smoke / blender)")
        env.action_space.seed(seed + idx)
        return env
    return thunk


# ---------- ALGO LOGIC: networks (same structure as CleanRL, hidden dim parameterized) ----------

class SoftQNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden=256, layer_norm=True):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim + action_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 1)
        # LayerNorm critic: suppresses bootstrap-extrapolation overestimation —
        # pilots #1-3 had Q estimates of 9~19 vs actual return -2, and the actor exploited the critic error (ReBRAC fix)
        self.ln1 = nn.LayerNorm(hidden) if layer_norm else nn.Identity()
        self.ln2 = nn.LayerNorm(hidden) if layer_norm else nn.Identity()

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        return self.fc3(x)


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden=256, residual=False):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc_mean = nn.Linear(hidden, action_dim)
        self.fc_logstd = nn.Linear(hidden, action_dim)
    # Residual mode (archived): the network outputs only a correction delta; action = clamp(obs
    # config + delta). Residual and absolute ckpts are same-shape but not interchangeable.
        self.residual = bool(residual)
        assert not self.residual or obs_dim >= STATIC_DIM + action_dim, \
            "residual mode requires obs in the standard [static|config|emb] layout (the config segment must be readable)"
    # Action space is the [-1,1]^72 tanh space: rescale structure kept (Jacobian formula unchanged).
    # Residual mode bypasses squash: delta is an unbounded Gaussian (see get_action).
        self.register_buffer("action_scale", torch.ones(action_dim))
        self.register_buffer("action_bias", torch.zeros(action_dim))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats
        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        if self.residual:
    # Residual mode: delta = unsquashed Gaussian, unbounded (the render side clips). No tanh:
    # near-origin double amplification would distort BC regression semantics; log_prob stays exact.
    # Sampled actions are left unclamped; the deterministic mean is clamped to a +/-2 hard bound
    # (lossless: cfg in [-1,1] means delta=+/-2 spans the range; clipped dims get zero gradient).
            mean = mean.clamp(-2.0, 2.0)
            normal = torch.distributions.Normal(mean, std)
            cfg = x[:, STATIC_DIM:STATIC_DIM + mean.shape[1]]
            delta = normal.rsample()
            return cfg + delta, normal.log_prob(delta).sum(1, keepdim=True), \
                (cfg + mean).clamp(-1.0, 1.0)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound (tanh Jacobian correction, kept verbatim from CleanRL)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    @torch.no_grad()
    def init_log_std_from(self, residual_std):
        """Init the log_std bias to the per-dim BC residual level.
        Inverse squash: log_std = MIN + 0.5(MAX−MIN)(tanh(raw)+1) → raw = atanh((2·log_std − MAX − MIN)/(MAX − MIN))."""
        std = np.clip(np.asarray(residual_std, np.float64), 1e-3, None)  # floor for ≈0-residual dims (e.g. soft-off canonical constants)
        target = np.clip(np.log(std), LOG_STD_MIN + 0.1, LOG_STD_MAX - 0.1)
        raw = np.arctanh((2.0 * target - LOG_STD_MAX - LOG_STD_MIN) / (LOG_STD_MAX - LOG_STD_MIN))
        dev = self.fc_logstd.bias.device
        self.fc_logstd.bias.copy_(torch.tensor(raw, dtype=torch.float32, device=dev))
        self.fc_logstd.weight.mul_(0.1)  # start bias-dominated, keep state-dependent modulation
        print(f"log_std initialised from BC residuals: median std {np.median(std):.4f}"
              f"(median log_std {np.median(target):.3f})")


def load_bc_ckpt(actor, bc_ckpt):
    """--bc-ckpt: log_std init from residual_std; warm-starts trunk/fc_mean when shapes match."""
    ckpt = torch.load(bc_ckpt, map_location="cpu", weights_only=True)  # stay on CPU: copy_ is cross-device safe, .numpy() requires CPU
    assert bool(ckpt.get("residual", False)) == getattr(actor, "residual", False), \
        "BC ckpt and Actor disagree on the residual convention (a residual BC is required for residual training; " \
        "an old absolute BC only fits an absolute Actor)"
    resid = ckpt["residual_std"]
    actor.init_log_std_from(resid.numpy() if torch.is_tensor(resid) else np.asarray(resid))
    bsd = ckpt.get("state_dict", {})
    loaded, skipped = [], []
    for name in ("fc1", "fc2", "fc_mean"):
        for pname in ("weight", "bias"):
            key = f"{name}.{pname}"
            dst = getattr(getattr(actor, name), pname)
            if key in bsd and bsd[key].shape == dst.shape:
                dst.data.copy_(bsd[key])
                loaded.append(key)
            else:
                skipped.append(key)
    if loaded:
        print(f"BC warm start: loaded {len(loaded)} tensors ({', '.join(sorted(set(k.split('.')[0] for k in loaded)))})")
    if skipped:
        print(f"BC warm start: skipped {len(skipped)} tensors with shape mismatch ({skipped[0]} etc.; "
              "expected when the BC was not trained with --use-render-emb: fc1 input 2128 != 2640)")


class PixelActor(nn.Module):
    """Pixel-observation policy: obs = [config 72 | clay 64² gray | preview 64² gray | goal 64² gray]
    (three images as three channels; a small CNN trunk learns spatial features — the canonical fix for
    pooled, structure-blind observations; all pooled vectors removed).
    get_action matches the Actor interface (absolute mode: rsample+tanh, third return tanh(mean))."""

    IMG = 64

    def __init__(self, obs_dim=72 + 3 * 64 * 64, action_dim=CONFIG_DIM, hidden=256,
                 residual=False):
        super().__init__()
        assert not residual, "PixelActor supports absolute mode only"
        self.obs_dim, self.action_dim = obs_dim, action_dim
        self.trunk = nn.Sequential(
            nn.Conv2d(3, 32, 5, 2, 2), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 2, 1), nn.ReLU(),
        )
        d = 64 * (self.IMG // 8) ** 2
        self.fc1 = nn.Linear(d + action_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc_mean = nn.Linear(hidden, action_dim)
        self.fc_logstd = nn.Linear(hidden, action_dim)
        self.register_buffer("action_scale", torch.ones(action_dim))   # ppo helper compatibility (identity)
        self.register_buffer("action_bias", torch.zeros(action_dim))

    def _split(self, x):
        n = x.shape[0]
        cfg = x[:, :72]
        imgs = x[:, 72:].reshape(n, 3, self.IMG, self.IMG)
        return cfg, imgs

    def forward(self, x):
        cfg, imgs = self._split(x)
        f = self.trunk(imgs).flatten(1)
        h = torch.relu(self.fc1(torch.cat([f, cfg], 1)))
        h = torch.relu(self.fc2(h))
        return self.fc_mean(h), self.fc_logstd(h).clamp(-5, 2)

    def init_log_std_from(self, arr):
        with torch.no_grad():
            self.fc_logstd.bias.copy_(torch.as_tensor(np.log(arr), dtype=torch.float32))

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        a = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t).sum(-1) - torch.log(1 - a.pow(2) + 1e-6).sum(-1)
        return a, log_prob, torch.tanh(mean)


class BCActorRef(nn.Module):
    """Read-only isomorphic copy of the BC actor (fc1/fc2/fc_mean, no output activation) — reference policy for BC regularization.
    residual=True (residual-convention BC): output = clamp(obs config segment + correction), same semantics as Actor residual mode."""

    def __init__(self, obs_dim, hidden, out_dim, residual=False):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc_mean = nn.Linear(hidden, out_dim)
        self.residual = bool(residual)
        assert not self.residual or obs_dim >= STATIC_DIM + out_dim

    def forward(self, x):
        d = self.fc_mean(F.relu(self.fc2(F.relu(self.fc1(x)))))
        if self.residual:
            return (x[:, STATIC_DIM:STATIC_DIM + d.shape[1]] + d).clamp(-1.0, 1.0)
        return d


def build_bc_ref(bc_ckpt, device):
    """Build the frozen BC reference policy from a BC checkpoint; returns None when input dim ≠ OBS_DIM (log_std init only)."""
    ckpt = torch.load(bc_ckpt, map_location="cpu", weights_only=True)
    in_dim = ckpt.get("obs_dim", 0)
    if in_dim != OBS_DIM:
        print(f"BC regulariser skipped: BC input {in_dim} dims != SAC obs {OBS_DIM} (the BC needs --use-render-emb to be 2640)")
        return None
    ref = BCActorRef(in_dim, ckpt.get("hidden", 256), CONFIG_DIM,
                     residual=ckpt.get("residual", False))
    ref.load_state_dict(ckpt["state_dict"])
    ref.requires_grad_(False)
    return ref.eval().to(device)


def main():
    args = parse_args()
    if args.smoke:
        # a few dozen updates is enough: shrink window/batch/network; logs and ckpt go to logs/ckpt/
        args.env = "smoke"
        args.total_timesteps, args.learning_starts = 96, 16
        args.batch_size, args.buffer_size, args.hidden_dim = 32, 2048, 64
        args.critic_warmup_updates, args.log_every = 4, 20
        args.log_file = args.log_file or ROOT / "logs/ckpt/sac_smoke_log.jsonl"
        args.out = args.out or ROOT / "logs/ckpt/sac_actor_smoke.pt"

    # TRY NOT TO MODIFY: seeding (verbatim from CleanRL)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup (change list 5/6: DISABLED autoreset + manual reset; no RecordEpisodeStatistics wrapper)
    if args.env == "blender":
        assert args.num_envs == 1, "--env blender supports --num-envs 1 only (one persistent worker per GPU)"
    env_kwargs = ({"rm_ckpt": args.rm_ckpt, "port": args.port,
                   "episodes_per_product": args.episodes_per_product}
                  if args.env == "blender" else None)
    envs = SyncVectorEnv(
        [make_env(args.env, args.seed + i, i, env_kwargs) for i in range(args.num_envs)],
        autoreset_mode=AutoresetMode.DISABLED,
    )
    assert isinstance(envs.single_action_space, Box), "only continuous action space is supported"
    assert envs.single_observation_space.shape == (OBS_DIM,)

    actor = Actor(OBS_DIM, CONFIG_DIM, args.hidden_dim).to(device)
    qf1 = SoftQNetwork(OBS_DIM, CONFIG_DIM, args.hidden_dim, layer_norm=args.critic_ln).to(device)
    qf2 = SoftQNetwork(OBS_DIM, CONFIG_DIM, args.hidden_dim, layer_norm=args.critic_ln).to(device)
    qf1_target = SoftQNetwork(OBS_DIM, CONFIG_DIM, args.hidden_dim, layer_norm=args.critic_ln).to(device)
    qf2_target = SoftQNetwork(OBS_DIM, CONFIG_DIM, args.hidden_dim, layer_norm=args.critic_ln).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    bc_ref = None
    if args.bc_ckpt is not None:
        load_bc_ckpt(actor, args.bc_ckpt)
        if args.bc_reg_lambda > 0:
            bc_ref = build_bc_ref(args.bc_ckpt, device)
    total_updates = max(1, args.total_timesteps - args.learning_starts)
    bc_decay_updates = max(1, int(0.1 * total_updates))  # λ_t decays linearly to 0 over the first 10% of updates
    bc_lam = 0.0  # current λ_t (for logging)

    # Automatic entropy tuning (target entropy −72 = −prod(action_dim))
    if args.autotune:
        target_entropy = -float(np.prod(envs.single_action_space.shape))
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    rb = ReplayBuffer(args.buffer_size, envs.single_observation_space.shape,
                      envs.single_action_space.shape, args.num_envs, args.seed, device)
    start_time = time.time()

    log_f = None
    if args.log_file is not None:
        args.log_file.parent.mkdir(parents=True, exist_ok=True)
        log_f = open(args.log_file, "a")

    def log(rec):
        if log_f:
            log_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            log_f.flush()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    ep_ret = np.zeros(args.num_envs, np.float64)   # episodes are fixed-length 2 steps, return computed by hand (change list 5)
    ep_cnt = 0
    update = 0
    loss_hist = []
    actor_loss = torch.tensor(float("nan"))
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        ep_ret += rewards
        for idx, done in enumerate(terminations):
            if done:
                ep_cnt += 1
                rec = {"type": "episode", "global_step": global_step,
                       "episodic_return": float(ep_ret[idx]), "episodic_length": 2}
                print(f"global_step={global_step}, episodic_return={ep_ret[idx]:.3f}")
                log(rec)
                ep_ret[idx] = 0.0

    # TRY NOT TO MODIFY: save data to the replay buffer. Episodes are fixed-length with no
    # TimeLimit, so real_next_obs == next_obs; restore CleanRL's final_observation handling if
    # truncation is ever added.
        rb.add(obs, next_obs, actions, rewards, terminations)

        # DISABLED autoreset: manual reset whenever any env terminates (episodes are fixed-length 2 steps, multi-env stays in sync)
        if terminations.any() or truncations.any():
            obs, _ = envs.reset()
        else:
            obs = next_obs

        # ALGO LOGIC: training.
        if global_step >= args.learning_starts:
            update += 1
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_observations)
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # optimize the model
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            # critic warmup: first critic_warmup_updates updates train critic only, no actor/alpha
            if update > args.critic_warmup_updates and update % args.policy_frequency == 0:
                for _ in range(args.policy_frequency):  # compensate for the delay
                    pi, log_pi, pi_mean = actor.get_action(data.observations)
                    qf1_pi = qf1(data.observations, pi)
                    qf2_pi = qf2(data.observations, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()
                    # BC regularization (change list 9): λ_t decays linearly over the first 10% of updates; --bc-reg-floor>0 keeps the floor anchor
                    if bc_ref is not None:
                        bc_lam = args.bc_reg_lambda * max(args.bc_reg_floor,
                                                          1.0 - update / bc_decay_updates)
                        if bc_lam > 0:
                            with torch.no_grad():
                                bc_pred = bc_ref(data.observations)
                            actor_loss = actor_loss + bc_lam * F.mse_loss(pi_mean, bc_pred)

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(data.observations)
                        alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            # update the target networks
            if update % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if update % args.log_every == 0:
                rec = {"type": "update", "update": update, "global_step": global_step,
                       "qf1_values": qf1_a_values.mean().item(), "qf2_values": qf2_a_values.mean().item(),
                       "qf1_loss": qf1_loss.item(), "qf2_loss": qf2_loss.item(),
                       "qf_loss": qf_loss.item() / 2.0, "actor_loss": actor_loss.item(),
                       "alpha": alpha, "bc_lambda": bc_lam,
                       "sps": int(global_step / (time.time() - start_time))}
                loss_hist.append(rec)
                print(f"update={update} qf_loss={rec['qf_loss']:.4f} actor_loss={rec['actor_loss']:.4f} "
                      f"alpha={alpha:.3f} SPS={rec['sps']}")
                log(rec)

    envs.close()
    if log_f:
        log_f.close()

    out = args.out or ROOT / "logs/ckpt" / ("sac_actor_smoke.pt" if args.smoke else "sac_actor.pt")
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "actor": actor.state_dict(),
        "qf1": qf1.state_dict(), "qf2": qf2.state_dict(),
        "alpha": float(alpha),
        "log_alpha": log_alpha.detach().cpu() if args.autotune else None,
        "obs_dim": OBS_DIM, "action_dim": CONFIG_DIM, "hidden_dim": args.hidden_dim,
        "gamma": args.gamma, "updates": update, "episodes": ep_cnt,
        "bc_ckpt": str(args.bc_ckpt) if args.bc_ckpt else None,
        "bc_reg_lambda": args.bc_reg_lambda if bc_ref is not None else None,
        "bc_reg_floor": args.bc_reg_floor if bc_ref is not None else None,
        "smoke": bool(args.smoke),
    }, out)
    print(f"checkpoint saved {out} (updates={update}, episodes={ep_cnt})")

    if args.smoke:
        assert update >= 50, f"only {update} updates, too few"
        assert len(rb) == args.total_timesteps * args.num_envs
        assert loss_hist, "no log records for the training segment"
        for rec in loss_hist:
            for k in ("qf1_loss", "qf2_loss", "actor_loss", "alpha"):
                assert np.isfinite(rec[k]), f"{k} is NaN/Inf: {rec}"
        assert np.isfinite(actor_loss.item()), "actor_loss is NaN/Inf"
        print(f"SAC SMOKE: PASS ({update} updates / {ep_cnt} episodes, "
              f"last qf_loss={loss_hist[-1]['qf_loss']:.4f} actor_loss={loss_hist[-1]['actor_loss']:.4f})")


if __name__ == "__main__":
    main()
