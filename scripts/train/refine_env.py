"""refine_env.py — Blender closed-loop env BlenderRefineEnv.

Concrete implementation of the RefineEnv protocol (interface pinned at sac_continuous_action.py:127):
  reset()  → draw (product, view) (no-replacement shuffled deck + per-block product switching to
             amortize BVH warmup) → NN-inherit (leave-self-out, template_generic fallback)
             → perturb_bc noise → camera/product ±10° jitter
             → worker set_episode → obs1 = [static 2056 | noisy inherit 72 | 0×512]
  step(a1) → worker renders preview tier (256²@8spp+OIDN) → ResNet18 embedding
             → obs2 = [static | a1 | preview emb 512], reward 0
  step(a2) → worker renders train tier (256²@25spp+OIDN) → RM score (z-score) = terminal reward
             → terminated True, info{"rm_score","rm_score_raw","sku","view"}

worker = persistent Blender process (scripts/env/render_worker.py, TCP JSON-lines) — benchmark
iron rule: no per-frame new processes. Worker crash → connection drop → raise and stop (no auto-restart).
With multiple envs, each env spawns one worker (incrementing port); ResNet18/RM share one in-process cache.

RM is specified by rm_ckpt (PickScore backbone, weights not embedded → needs HF cache;
set HF_HUB_OFFLINE=1 for offline training; default logs/ckpt/rm_v19c.pt).
Render-embedding backbone matches the static cache (frozen ResNet18,
reusing extract_embeddings.build_encoder).

Standalone smoke test:
  HF_HUB_OFFLINE=1 .venv/bin/python scripts/train/refine_env.py [--episodes 3]
"""

import importlib.util
import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
_REWARD_VGG = None  # run32 reward-path VGG embedding, lazy-load cache

sys.path.insert(0, str(ROOT / "addons"))
from lighting_rl import core  # noqa: E402

from sac_continuous_action import EMB_DIM, OBS_DIM, RefineEnv  # noqa: E402  same dir (script run adds it to sys.path)

VIEW_ONEHOT = {"front": [1.0, 0.0], "high": [0.0, 1.0]}
CAM_JITTER_DEG = 10.0   # camera orbits origin around Z ±10° (plan §1)
PROD_JITTER_DEG = 10.0  # product rotates around origin Z ±10° (plan §1, stacks with camera jitter)


def _load(alias, path):
    """Load project-flat scripts by absolute path via importlib (self-executes only under __main__ guard)."""
    spec = importlib.util.spec_from_file_location(alias, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


retrieve_mod = _load("rl_nn_retrieve", ROOT / "scripts" / "nn_retrieve.py")
extract_mod = _load("rl_extract", ROOT / "scripts" / "extract_embeddings.py")
perturb_mod = _load("rl_perturb", ROOT / "scripts" / "train" / "perturb_bc.py")
rank_mod = _load("rl_rank", ROOT / "scripts" / "train" / "rank_candidates.py")

# in-process shared cache: with multiple envs (--num-envs 2+) the ~3.9G PickScore weights load once, avoiding VRAM doubling
_ENC_CACHE = None
_RM_CACHE = {}
# double-checked lock for lazy init: prevents duplicate construction in threaded rollouts
# (RLOO num-envs>1); serial single-thread construction makes this a pure fallback
_MODEL_INIT_LOCK = threading.Lock()


def _get_encoder():
    global _ENC_CACHE
    if _ENC_CACHE is None:
        with _MODEL_INIT_LOCK:
            if _ENC_CACHE is None:
                _ENC_CACHE = extract_mod.build_encoder()
    return _ENC_CACHE


def _get_rm(ckpt, dev):
    key = str(ckpt)
    if key not in _RM_CACHE:
        with _MODEL_INIT_LOCK:
            if key not in _RM_CACHE:
                _RM_CACHE[key] = rank_mod.build_rm(ckpt, dev)
    return _RM_CACHE[key]


class BlenderRefineEnv(RefineEnv):
    """Real closed loop: persistent Blender worker rendering + ResNet18 preview embedding + RM (per rm_ckpt) terminal reward."""

    def __init__(self, rm_ckpt=None, port=17871, episodes_per_product=8, seed=1,
                 blender=None, device="OPTIX", holdout=False, include_seedless=False,
                 skus=None, deck_weights=None, anchor_lambda=0.0, reward_mode="rm",
                 seed_ref_embeds=None, neutral_embeds=None, reward_embedder="r18",
                 goal_images=None, pixel_obs=False, no_jitter=False, max_steps=2):
        self.pixel_obs = pixel_obs
        self.no_jitter = no_jitter
        self.max_steps = int(max_steps)  # run36: 6-step refine loop (5 preview + 1 train final)
        self.reward_embedder = reward_embedder
        super().__init__()
        self.rng = np.random.default_rng(seed)
        self.episodes_per_product = int(episodes_per_product)
    # reward_mode="seed" (archived): terminal reward = -tanh-RMS(a2, own seed), zero RM.
        assert reward_mode in ("rm", "seed", "render_emb", "render_emb_dd", "render_emb_goal", "pixel_goal")
        self.reward_mode = reward_mode
    # Archived reward modes (kept for reproducibility, off the active path):
    # render_emb = -||E(final) - E(own seed render)||; render_emb_dd = identity-cancelling
    # diff-of-diffs against a goal image embedding.
        self.clay_cache = {}
        self.mask_cache = {}
        self.goal_images = None
        if reward_mode == "pixel_goal":
            assert goal_images is not None, "pixel_goal requires goal_images (reference render directory)"
            from PIL import Image as _Im
            self.goal_images = {}
            mask_dir = ROOT / "data/raft/masks"
            for pth in sorted(Path(goal_images).glob("ref_*.png")):
                key = pth.stem.replace("ref_", "", 1)
                g = np.asarray(_Im.open(pth).convert("L").resize((64, 64), _Im.BILINEAR),
                               np.float32) / 255.0
                mp = mask_dir / f"{key}.png"
                if mp.is_file():  # ground-truth mask (rendered from Blender "product" collection, immune to white products); threshold fallback if missing
                    m = np.asarray(_Im.open(mp).convert("L").resize((64, 64), _Im.NEAREST),
                                   np.float32) / 255.0
                    m = (m > 0.5).astype(np.float32)
                else:
                    m = (g < 0.985).astype(np.float32)
                self.goal_images[key] = (g, m)
        self.seed_ref_embeds = None
        self.goal_dim = 0
        if reward_mode in ("render_emb", "render_emb_dd", "render_emb_goal", "pixel_goal"):
            assert seed_ref_embeds is not None, "render_emb* modes require seed_ref_embeds (npz path)"
            z = np.load(seed_ref_embeds)
            self.seed_ref_embeds = {k: z[k] for k in z.files}
            if reward_mode in ("render_emb_dd", "render_emb_goal", "pixel_goal"):
                self.goal_dim = int(self.seed_ref_embeds[list(self.seed_ref_embeds)[0]].shape[0])
        self.neutral_embeds = None
        if reward_mode == "render_emb_dd":
            assert neutral_embeds is not None, "render_emb_dd requires neutral_embeds (npz path)"
            zn = np.load(neutral_embeds)
            self.neutral_embeds = {k: zn[k] for k in zn.files}
        self.obs_dim = OBS_DIM + self.goal_dim  # goal channel sized by goal embedding dim (512/768…)
        if pixel_obs:
            self.obs_dim = 72 + 3 * 64 * 64  # run35 pure-pixel obs override
        from gymnasium.spaces import Box as _Box
        self.observation_space = _Box(low=-np.inf, high=np.inf,
                                      shape=(self.obs_dim,), dtype=np.float32)
    # Start anchor (--anchor-lambda, archived): reward = RM z - lambda*d(a2, this episode's clean
    # inherited config), structurally countering cross-product common-mode drift. lambda=0 = off.
        self.anchor_lambda = float(anchor_lambda)

    # Training pool = seeds ∩ embeddings, minus template and the 6 holdout (test_*) products.
    # holdout=True: only the 6 test_* (generalization eval, never trained on).
    # include_seedless=True: products with an embedding but no seed join the pool; their reset
    # falls back to an NN-inherit start. skus=<list> fully specifies the pool (overrides both).
        seed_skus = {p.stem for p in (ROOT / "seeds").glob("*.json")} - {"template_generic"}
        emb_skus = {p.stem for p in (ROOT / "data" / "embeddings").glob("*.npy")}
        if skus is not None:
            pool = sorted({str(s) for s in skus})
            missing = [s for s in pool if s not in emb_skus]
            assert not missing, f"skus pool has products without embeddings (run extract_embeddings first): {missing}"
            self.skus = pool
        elif holdout:
            self.skus = sorted(s for s in emb_skus if s.startswith("test_"))
        else:
            pool = {s for s in seed_skus & emb_skus if not s.startswith("test_")}
            if include_seedless:
                pool |= {s for s in emb_skus - seed_skus if not s.startswith("test_")}
            self.skus = sorted(pool)
        assert self.skus, "no usable products (training pool = seeds/ ∩ data/embeddings/; holdout = test_* embeddings)"

        # deck weighting (E8, run6 new-product weighting): listed products appear N times per deck,
        # familiar ones keep 1 as an anti-forgetting anchor
        self.deck_weights = {str(k): int(v) for k, v in (deck_weights or {}).items()}
        assert all(v >= 1 for v in self.deck_weights.values()), "deck_weights must be >=1"
        unknown = sorted(set(self.deck_weights) - set(self.skus))
        assert not unknown, f"deck_weights contains products not in the pool: {unknown}"

        # ---- spawn persistent render worker (stdout/stderr to log file, not PIPE, to avoid deadlock from bloated render logs) ----
        blender = blender or os.environ.get("BLENDER", "blender")
        self.port = port  # keep port: step() tmp png names include it so parallel renders don't collide
        self._tmp = ROOT / "logs" / "rollout" / "tmp"
        self._tmp.mkdir(parents=True, exist_ok=True)
        log_path = self._tmp.parent / "worker.log"
        self._log_f = open(log_path, "a")
        self.proc = subprocess.Popen(
            [blender, "-b", str(ROOT / "scene" / "stage.blend"),
             "--python", str(ROOT / "scripts" / "env" / "render_worker.py"),
             "--", "--port", str(port), "--device", device],
            stdout=self._log_f, stderr=subprocess.STDOUT, text=True)
        # poll the port until the worker is ready (Blender startup + scene load takes ~seconds)
        t0 = time.time()
        while True:
            if self.proc.poll() is not None:
                raise RuntimeError(f"render worker exited immediately with rc={self.proc.returncode}, see {log_path}")
            try:
                self.sock = socket.create_connection(("127.0.0.1", port), timeout=1.0)
                break
            except OSError:
                if time.time() - t0 > 120:
                    raise RuntimeError(f"render worker not ready after 120s, see {log_path}")
                time.sleep(0.5)
        self.sock.settimeout(600.0)  # per-command timeout backstop (slowest render ~3s, generous 10 min)
        self.sock_f = self.sock.makefile("rw", encoding="utf-8", newline="\n")
        self._cmd({"cmd": "ping"})

        # ---- frozen backbones: ResNet18 (preview embedding) + RM (terminal reward), both via in-process shared cache ----
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.enc_model, self.enc_prep = _get_encoder()
        if reward_mode == "rm":
            rm_ckpt = rm_ckpt or ROOT / "logs" / "ckpt" / "rm_v19c.pt"
            self.rm_encode, self.rm_head, self.rm_mean, self.rm_std, rm_src = _get_rm(rm_ckpt, dev)
        else:
            self.rm_encode = self.rm_head = None
            self.rm_mean = self.rm_std = 0.0
            rm_src = "seed-dist (no RM)" if reward_mode == "seed" else "render-emb seed-ref (no RM)"
        w_note = f", deck weights {self.deck_weights}" if self.deck_weights else ""
        print(f"BlenderRefineEnv: worker ready (port {port}), {len(self.skus)} products{w_note}, RM={rm_src}")

        self._t = 0
        self._cur_sku = None
        self._ep_in_block = 0
        self._deck = []  # product deck (no-replacement shuffled multiset per deck_weights), reshuffled when empty — see _next_sku()

    # ---------- internal ----------

    def _pack(self, static, config, emb):
        if getattr(self, "pixel_obs", False):
            return self._pack_pixel(config)
        base = [static, config, emb]
        if self.reward_mode in ("render_emb_dd", "render_emb_goal", "pixel_goal"):
            base.append(self._goal_emb)
        return np.concatenate(base).astype(np.float32)

    def _clay64(self, sku):
        if sku not in self.clay_cache:
            from PIL import Image as _Im
            p = ROOT / "data/embeddings/img" / f"{sku}_clay_front.png"
            g = np.asarray(_Im.open(p).convert("L").resize((64, 64), _Im.BILINEAR),
                           np.float32) / 255.0
            self.clay_cache[sku] = g
            self.mask_cache[sku] = (g < 0.985).astype(np.float32)
        return self.clay_cache[sku]

    def _mask64(self, sku, view=None):
        mp = ROOT / "data/raft/masks" / f"{sku}_{view or self._view}.png"
        if mp.is_file():
            from PIL import Image as _Im
            m = np.asarray(_Im.open(mp).convert("L").resize((64, 64), _Im.NEAREST),
                           np.float32) / 255.0
            return (m > 0.5).astype(np.float32)
        self._clay64(sku)
        return self.mask_cache[sku]

    def _pack_pixel(self, config):
        """run35: obs = [config 72 | clay64 | preview64 | goal64]; preview defaults to zeros before the first render."""
        prev = getattr(self, "_prev64", None)
        if prev is None:
            prev = np.zeros((64, 64), np.float32)
        return np.concatenate([
            config.astype(np.float32).ravel(),
            self._clay64(self._sku).ravel(),
            prev.ravel(),
            (self.goal_images[self._goal_key][0] if self.reward_mode in
             ("render_emb_dd", "render_emb_goal", "pixel_goal")
             else np.zeros((64, 64), np.float32)).ravel(),  # no-goal modes: channel left zero
        ]).astype(np.float32)

    def _next_sku(self):
        """Draw next product: deck = pool repeated per deck_weights as a multiset, shuffled
        no-replacement; each product appears exactly weight times (default 1); reshuffle when empty."""
        if not self._deck:
            weighted = [s for s in self.skus for _ in range(self.deck_weights.get(s, 1))]
            self._deck = list(self.rng.permutation(weighted))
        return str(self._deck.pop())

    def _cmd(self, d):
        try:
            self.sock_f.write(json.dumps(d) + "\n")
            self.sock_f.flush()
            line = self.sock_f.readline()
        except (BrokenPipeError, ConnectionResetError, socket.timeout) as e:
            raise RuntimeError(f"render worker connection error ({type(e).__name__}, process crashed? "
                               f"see logs/rollout/worker.log)") from e
        if not line:
            raise RuntimeError("render worker connection dropped (process crashed? see logs/rollout/worker.log)")
        resp = json.loads(line)
        if not resp.get("ok"):
            raise RuntimeError(f"worker command failed {d.get('cmd')}: {resp.get('error')}")
        return resp

    def _reward_embed(self, path):
        """Reward-path embedding (run32 single variable): r18 = ResNet18 512-d; vgg = VGG16
        conv3_3/4_3 normalized-pooled 768-d (LPIPS backend family). Reward only, never into obs."""
        return self._embed(path, rwd=True)

    def _embed(self, path, rwd=False):
        from PIL import Image
        if rwd and getattr(self, "reward_embedder", "r18") == "vgg":
            global _REWARD_VGG
            if _REWARD_VGG is None:
                from torchvision.models import vgg16, VGG16_Weights
                _REWARD_VGG = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)\
                    .features.eval().to(torch.device("cuda"))
            x = torch.as_tensor(np.asarray(Image.open(path).convert("RGB")
                                            .resize((224, 224)), np.float32) / 255.0)\
                .permute(2, 0, 1)[None].to(next(_REWARD_VGG.parameters()).device)
            x = (x - torch.tensor([0.485, 0.456, 0.406], device=x.device)[:, None, None]) / \
                torch.tensor([0.229, 0.224, 0.225], device=x.device)[:, None, None]
            fs = []
            with torch.no_grad():
                for i, layer in enumerate(_REWARD_VGG):
                    x = layer(x)
                    if i in (16, 23):
                        f = x.mean(dim=(2, 3))
                        fs.append(f / f.norm(dim=1, keepdim=True))
            return torch.cat(fs, 1).cpu().numpy()[0]
        with torch.no_grad():
            return self.enc_model(self.enc_prep([Image.open(path).convert("RGB")])).float().cpu().numpy()[0]

    def _score(self, path):
        """RM scoring → (raw, z). Features L2-normalized (same recipe as train_rm)."""
        with torch.no_grad():
            f = self.rm_encode([str(path)])
            f = f / f.norm(dim=-1, keepdim=True)
            raw = float(self.rm_head(f.float()).squeeze(-1).detach().cpu())
        return raw, (raw - self.rm_mean) / max(self.rm_std, 1e-6)

    # ---------- protocol ----------

    def reset(self, *, seed=None, options=None):
        import gymnasium as gym
        gym.Env.reset(self, seed=seed)  # no super().reset(): RefineEnv.reset raises directly
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._t = 0

    # Per-block product switching: product changes only across blocks (amortizes BVH warmup),
    # view/jitter are drawn per episode. Products come from a no-replacement shuffled deck, so
    # each update sees a fixed, full-coverage set (no inter-update composition noise);
    # deck_weights repeats listed products N times per deck.
        if self._cur_sku is None or self._ep_in_block >= self.episodes_per_product:
            self._cur_sku = self._next_sku()
            self._ep_in_block = 0
        self._ep_in_block += 1
        sku, view = self._cur_sku, ("front" if self.rng.integers(2) == 0 else "high")

        # start = NN-inherit (leave-self-out) / template fallback + perturbation noise
        hit = retrieve_mod.retrieve(sku, exclude_self=True)
        seed_path = hit["seed"] if hit else ROOT / "seeds" / "template_generic.json"
        seed_cfgs = core.load_seed(seed_path)
        base = seed_cfgs.get(view) or next(iter(seed_cfgs.values()))
        self._base_tanh = core.to_tanh(base).astype(np.float32)  # clean inherited config = anchor (for anchor_lambda)
        inherit = perturb_mod.perturb_config(base, self.rng)

        if getattr(self, "no_jitter", False):
            cam_jitter, prod_z = 0.0, 0.0
        else:
            cam_jitter = float(self.rng.uniform(-CAM_JITTER_DEG, CAM_JITTER_DEG))
            prod_z = float(self.rng.uniform(-PROD_JITTER_DEG, PROD_JITTER_DEG))
        self._cam_jitter, self._prod_z = cam_jitter, prod_z  # stash start: for RLOO reset_replay
        self._static = np.concatenate([
            np.load(ROOT / "data" / "embeddings" / f"{sku}.npy"),  # 2052
            np.asarray(VIEW_ONEHOT[view], np.float32),
            np.asarray([np.sin(np.deg2rad(cam_jitter)), np.cos(np.deg2rad(cam_jitter))], np.float32),
        ])
        assert self._static.shape == (2056,)
        self._sku, self._view = sku, view
        if self.reward_mode == "seed":
            own = core.load_seed(ROOT / "seeds" / f"{sku}.json")
            self._seed_tanh = core.to_tanh(
                own.get(view) or next(iter(own.values()))).astype(np.float32)
        if self.reward_mode in ("render_emb_dd", "render_emb_goal", "pixel_goal"):
    # The goal must be cross-product: otherwise it is a near-deterministic function of the product
    # identity already in obs, and the channel is dropped as redundant (untrained input at deploy).
    # Training goal = a random other product's same-view seed render; only reading it scores.
            cands = [k for k in self.seed_ref_embeds
                     if k.split("_")[-1] == view and not k.startswith(f"{sku}_")]
            self._goal_key = str(self.rng.choice(cands))
            self._goal_emb = self.seed_ref_embeds[self._goal_key].astype(np.float32)
        if getattr(self, "pixel_obs", False):
            self._prev64 = None  # new episode: preview channel zeroed
        self._cmd({"cmd": "set_episode", "sku": sku, "view": view,
                   "cam_jitter": cam_jitter, "prod_z": prod_z})
        self._inherit_tanh = core.to_tanh(inherit).astype(np.float32)
        return self._pack(self._static, self._inherit_tanh, np.zeros(EMB_DIM, np.float32)), \
            {"sku": sku, "view": view, "inherit_from": Path(seed_path).stem,
             "base_tanh": self._base_tanh.copy()}  # RLOO direct-anchor regularizer reads the anchor config

    def reset_replay(self):
        """RLOO same-start replay: return verbatim to the last reset's drawn start (same
        product/view/camera jitter/product rotation/inherit noise) without touching rng or block
        counters. Only exists after reset()."""
        assert self._cur_sku is not None, "reset() must be called at least once before reset_replay()"
        self._t = 0
        if getattr(self, "pixel_obs", False):
            self._prev64 = None  # replay zeroes preview channel (avoid cross-episode contamination)
        self._cmd({"cmd": "set_episode", "sku": self._sku, "view": self._view,
                   "cam_jitter": self._cam_jitter, "prod_z": self._prod_z})
        return self._pack(self._static, self._inherit_tanh, np.zeros(EMB_DIM, np.float32)), \
            {"sku": self._sku, "view": self._view, "replay": True,
             "base_tanh": self._base_tanh.copy()}

    def step(self, action):
        a_raw = np.asarray(action, np.float32).reshape(-1)  # residual mode can exceed bounds: anchor reads raw to keep pull-back
        a = np.clip(a_raw, -1.0, 1.0)
        assert a.shape == (72,), f"action dims {a.shape}, expected (72,)"
        cfg = core.from_tanh(a)
        if self._t < self.max_steps - 1:  # intermediate step (run36: 6 steps = 5 preview + per-step pixel shaping)
            self._t += 1
            out = self._tmp / f"{os.getpid()}_{self.port}_s{self._t}.png"  # port in filename: parallel multi-thread/multi-env renders don't collide
            self._cmd({"cmd": "render", "config": cfg, "tier": "preview", "out": str(out)})
            emb = self._embed(out)
            r1 = 0.0
            if self.reward_mode == "pixel_goal":  # run34/35/36 process reward: every step gives immediate direction
                from PIL import Image as _Im
                g64 = np.asarray(_Im.open(out).convert("L").resize((64, 64), _Im.BILINEAR),
                                 np.float32) / 255.0
                if getattr(self, "pixel_obs", False):
                    self._prev64 = g64  # preview channel for next step's obs
                _gm, _mm = self.goal_images[self._goal_key]
                _m = np.maximum(_mm, self._mask64(self._sku))  # union of both products' masks (no pixel escapes penalty)
                r1 = -float((((g64 - _gm) ** 2) * _m).sum() / max(_m.sum(), 1.0))
            return self._pack(self._static, a, emb), r1, False, False, {}
        self._t = 0
        pen = self.anchor_lambda * float(np.linalg.norm(a_raw - self._base_tanh) / np.sqrt(72))
        if self.reward_mode == "pixel_goal":
            out = self._tmp / f"{os.getpid()}_{self.port}_s2.png"
            self._cmd({"cmd": "render", "config": cfg, "tier": "train", "out": str(out)})
            from PIL import Image as _Im
            g64 = np.asarray(_Im.open(out).convert("L").resize((64, 64), _Im.BILINEAR),
                             np.float32) / 255.0
            _gm, _mm = self.goal_images[self._goal_key]
            _m = np.maximum(_mm, self._mask64(self._sku))
            r = -float((((g64 - _gm) ** 2) * _m).sum() / max(_m.sum(), 1.0))
            return np.zeros(self.obs_dim, np.float32), r - pen, True, False, \
                {"rm_score": r, "rm_score_raw": r, "goal": self._goal_key,
                 "anchor_pen": pen, "sku": self._sku, "view": self._view}
        if self.reward_mode == "render_emb_goal":
            # run33: goal-conditioned goal-reaching — reward = −LPIPS-feature distance to a cross-product goal (no diff-of-diffs, no neutral image)
            out = self._tmp / f"{os.getpid()}_{self.port}_s2.png"
            self._cmd({"cmd": "render", "config": cfg, "tier": "train", "out": str(out)})
            emb = self._reward_embed(out)
            r = -float(np.linalg.norm(emb - self._goal_emb))
            return np.zeros(self.obs_dim, np.float32), r - pen, True, False, \
                {"rm_score": r, "rm_score_raw": r, "goal": self._goal_key,
                 "anchor_pen": pen, "sku": self._sku, "view": self._view}
        if self.reward_mode == "render_emb_dd":
            out = self._tmp / f"{os.getpid()}_{self.port}_s2.png"
            self._cmd({"cmd": "render", "config": cfg, "tier": "train", "out": str(out)})
            emb = self._reward_embed(out)
            key = f"{self._sku}_{self._view}"
            delta_ours = emb - self.neutral_embeds[key]
            delta_ref = self._goal_emb - self.neutral_embeds[self._goal_key]  # cross-product: use goal product's neutral image
            r = -float(np.linalg.norm(delta_ours - delta_ref))
            return np.zeros(self.obs_dim, np.float32), r - pen, True, False, \
                {"rm_score": r, "rm_score_raw": r, "dd_dist": -r, "anchor_pen": pen,
                 "sku": self._sku, "view": self._view}
        if self.reward_mode == "render_emb":
            out = self._tmp / f"{os.getpid()}_{self.port}_s2.png"
            self._cmd({"cmd": "render", "config": cfg, "tier": "train", "out": str(out)})
            emb = self._reward_embed(out)
            r = -float(np.linalg.norm(emb - self.seed_ref_embeds[f"{self._sku}_{self._view}"]))
            return np.zeros(self.obs_dim, np.float32), r - pen, True, False, \
                {"rm_score": r, "rm_score_raw": r, "seed_ref_dist": -r, "anchor_pen": pen,
                 "sku": self._sku, "view": self._view}
        if self.reward_mode == "seed":
            # seed-distance compass: reward in config space, skips train-tier render and RM forward
            # (big SPS win); info rm_score filled with −d placeholder so episode logs/read_train_log stay unchanged
            d = float(np.linalg.norm(a - self._seed_tanh) / np.sqrt(72))
            return np.zeros(self.obs_dim, np.float32), -d - pen, True, False, \
                {"rm_score": -d, "rm_score_raw": -d, "seed_dist": d, "anchor_pen": pen,
                 "sku": self._sku, "view": self._view}
        out = self._tmp / f"{os.getpid()}_{self.port}_s2.png"
        self._cmd({"cmd": "render", "config": cfg, "tier": "train", "out": str(out)})
        raw, z = self._score(out)
    # The anchor reads the pre-clip raw action so its contrast survives inside the clipped
    # saturation zone; absolute mode has raw == clip, so behaviour is unchanged there.
        return np.zeros(self.obs_dim, np.float32), z - pen, True, False, \
            {"rm_score": z, "rm_score_raw": raw, "anchor_pen": pen,
             "sku": self._sku, "view": self._view}

    def close(self):
        try:
            if getattr(self, "sock_f", None):
                self._cmd({"cmd": "shutdown"})
                self.sock_f.close()
                self.sock.close()
        except Exception:
            pass
        if getattr(self, "proc", None):
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.terminate()
        if getattr(self, "_log_f", None):
            self._log_f.close()
        super().close()


if __name__ == "__main__":
    argv = sys.argv[1:]
    if "-h" in argv or "--help" in argv:
        print("Standalone env smoke test (module library; the trainer is rloo_continuous_action.py).\n"
              "usage: python scripts/train/refine_env.py [--episodes N]\n"
              "requires: scene/stage.blend, model/<sku>.blend + data/embeddings/<sku>.npy (seeds optional)")
        raise SystemExit(0)
    n_ep = int(argv[argv.index("--episodes") + 1]) if "--episodes" in argv else 3
    env = BlenderRefineEnv()
    try:
        for ep in range(n_ep):
            t0 = time.time()
            obs, info = env.reset()
            assert obs.shape == (OBS_DIM,) and np.isfinite(obs).all()
            t1 = time.time()
            a1 = env.action_space.sample() * 0.2  # small random action, don't blow up the whole setup
            obs2, r1, term1, _, _ = env.step(a1)
            assert obs2.shape == (OBS_DIM,) and np.isfinite(obs2).all() and not term1 and r1 == 0.0
            assert not np.allclose(obs2[-EMB_DIM:], 0), "preview embedding all zeros?"
            t2 = time.time()
            a2 = env.action_space.sample() * 0.2
            _, r2, term2, _, info2 = env.step(a2)
            assert term2 and np.isfinite(r2)
            t3 = time.time()
            print(f"[ep {ep + 1}] {info['sku']}_{info['view']} inherited from {info['inherit_from']} | "
                  f"reset {t1 - t0:.2f}s / step1 {t2 - t1:.2f}s / step2 {t3 - t2:.2f}s | "
                  f"reward z={r2:+.3f} raw={info2['rm_score_raw']:+.3f}", flush=True)
        print("REFINE ENV SMOKE: PASS")
    finally:
        env.close()
