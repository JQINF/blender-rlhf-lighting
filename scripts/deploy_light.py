"""deploy_light.py — deployment auto-lighting: NN inherit → policy 2-step correction → sample N step-2 candidates → RM adjudication → do-no-harm gate.

Deployment contract:
  start = raw NN inheritance (exclude_self=False, falls back to template_generic), no perturbation, no camera/product jitter
  (the obs jitter sincos segment gets (0,1), inside the training distribution);
  step-1 deterministic tanh(mean) → preview-tier render (256²@8spp+OIDN) → ResNet18 embedding appended to obs2;
  step-2 sample N candidates from π(·|obs2) → one train-tier render each (256²@25spp+OIDN) → RM z-score descending;
  **do-no-harm gate**: the raw inherited config goes through the same RM, Δz = z(winner) − z(inherit)
  ≤ --gate-tau (default 0.15) keeps the inherited lighting and skips the correction — preregistered E6
  validation raised agreement with human picks from 0.548 to 0.667 (+11.9pt over 42 new products × both views).
  The winner config is written to <renders-dir>/winner_config.json (the inherited config when the gate fires);
  core.save_config schema validation comes for free.

obs layout verbatim-aligned with refine_env: [static 2056 = embedding 2052 + view onehot 2 + jitter sincos 2 | config_tanh 72 | render embedding 512].

The addon shell (addons/lighting_deploy) subprocess-calls this script; the --out result.json contract:
  {"sku","view","inherit_from","n","preview_png",
   "winner":{"index","score_z","score_raw","png","config_path"}   # gate fired: index=-1, config=inherit
   "gate":{"tau","applied","winner_z","inherit_z","delta_z","inherit_png","inherit_config_path"},
   "candidates":[{"index","score_z","score_raw","png","config_path"}...]}  (candidates sorted by z descending)

Usage (venv):
  HF_HUB_OFFLINE=1 .venv/bin/python scripts/deploy_light.py --sku 54 --view front \
      --actor logs/ckpt/run38e_final.pt --rm-ckpt logs/ckpt/rm_v19c.pt \
      --n 16 --out logs/rollout/tmp/deploy_54_front.json
"""

import argparse
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "train"))  # same-dir imports for refine_env / sac_continuous_action
sys.path.insert(0, str(ROOT / "addons"))

from lighting_rl import core  # noqa: E402
from refine_env import VIEW_ONEHOT, _get_encoder, _get_rm  # noqa: E402  reuses the in-process shared caches
from sac_continuous_action import CONFIG_DIM, OBS_DIM, Actor  # noqa: E402

STATIC_DIM = OBS_DIM - CONFIG_DIM - 512  # 2056
EMB_DIM = 512

# Shipped defaults: the policy and RM published with this repo (put both in logs/ckpt/).
DEFAULT_ACTOR = ROOT / "logs" / "ckpt" / "run38e_final.pt"
# The shipped RM is rm_v19c; its E6-calibrated gate threshold is τ=0.15 (see --gate-tau).
DEFAULT_RM = ROOT / "logs" / "ckpt" / "rm_v19c.pt"


def _load(alias, path):
    spec = importlib.util.spec_from_file_location(alias, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


retrieve_mod = _load("rl_nn_retrieve", ROOT / "scripts" / "nn_retrieve.py")
rank_mod = _load("rl_rank", ROOT / "scripts" / "train" / "rank_candidates.py")


class Worker:
    """render_worker persistent-process client (protocol in the header of scripts/env/render_worker.py).

    stdout goes through PIPE + a background drain thread writing to the log: a snap-packaged Blender's
    child process exits silently with rc=0 if stdout is handed a file object directly (measured 2026-09-05;
    terminal-launched Blender has no such issue), while a bare PIPE risks the classic pipe-buffer deadlock
    from verbose render logs — the drain thread avoids both."""

    def __init__(self, port, device):
        blender = os.environ.get("BLENDER", "blender")
        log_path = ROOT / "logs" / "rollout" / "deploy_worker.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_f = open(log_path, "a")
        self.proc = subprocess.Popen(
            [blender, "-b", str(ROOT / "scene" / "stage.blend"),
             "--python", str(ROOT / "scripts" / "env" / "render_worker.py"),
             "--", "--port", str(port), "--device", device],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        self._drain = threading.Thread(target=self._drain_stdout, daemon=True)
        self._drain.start()
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
        self.sock.settimeout(600.0)
        self.sock_f = self.sock.makefile("rw", encoding="utf-8", newline="\n")
        self.cmd({"cmd": "ping"})

    def _drain_stdout(self):
        for line in self.proc.stdout:
            try:
                self._log_f.write(line)
                self._log_f.flush()
            except ValueError:
                break  # log file already closed (close-vs-drain race at shutdown; process is dead anyway)

    def cmd(self, d):
        self.sock_f.write(json.dumps(d) + "\n")
        self.sock_f.flush()
        line = self.sock_f.readline()
        if not line:
            raise RuntimeError("render worker connection dropped (process crashed? see logs/rollout/deploy_worker.log)")
        resp = json.loads(line)
        if not resp.get("ok"):
            raise RuntimeError(f"worker command failed {d.get('cmd')}: {resp.get('error')}")
        return resp

    def close(self):
        try:
            self.cmd({"cmd": "shutdown"})
            self.sock_f.close()
            self.sock.close()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.terminate()
        self._drain.join(timeout=5)  # let the drain thread finish writing the tail before closing the log
        self._log_f.close()


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


@torch.no_grad()
def sample_action(actor, obs, device):
    x = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    a, _, _ = actor.get_action(x)  # first return value = rsample + tanh, exploration-distribution sample
    return a.squeeze(0).cpu().numpy()


@torch.no_grad()
def embed_image(enc_model, enc_prep, path):
    """Frozen ResNet18 → 512-dim render embedding (same convention as refine_env._embed)."""
    from PIL import Image
    return enc_model(enc_prep([Image.open(path).convert("RGB")])).float().cpu().numpy()[0]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sku", required=True)
    p.add_argument("--view", choices=["front", "high"], default="front")
    p.add_argument("--actor", type=Path, default=DEFAULT_ACTOR)
    p.add_argument("--rm-ckpt", type=Path, default=DEFAULT_RM)
    p.add_argument("--n", type=int, default=16, help="number of step-2 sampled candidates (deployment = best-of-16)")
    p.add_argument("--gate-tau", type=float, default=0.15,
                   help="do-no-harm gate threshold: keep the inherited lighting when Δz=z(winner)−z(inherit) ≤ τ "
                        "(pass -9 to disable the gate). τ is RM-calibrated: the shipped rm_v19c pairs with τ=0.15; "
                        "re-validate the threshold when swapping RMs")
    p.add_argument("--port", type=int, default=17873, help="render worker port (outside training 17871 / re-eval 17872)")
    p.add_argument("--device", default="OPTIX", help="render device (OPTIX→CUDA→CPU fallback chain)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True, help="result.json contract output path")
    p.add_argument("--renders-dir", type=Path, default=None,
                   help="directory for candidate renders + winner config (default logs/rollout/deploy_<sku>_<view>)")
    return p.parse_args()


def main():
    args = parse_args()
    sku, view = str(args.sku), args.view
    if not (ROOT / "model" / f"{sku}.blend").is_file():
        raise SystemExit(f"model/{sku}.blend not found")
    emb_path = ROOT / "data" / "embeddings" / f"{sku}.npy"
    if not emb_path.is_file():
        raise SystemExit(f"query embedding missing: {emb_path} (run extract_embeddings.py first)")
    if not args.actor.is_file():
        raise SystemExit(f"--actor not found: {args.actor}")
    if not args.rm_ckpt.is_file():
        raise SystemExit(f"--rm-ckpt not found: {args.rm_ckpt} (explicit RM path required; never silently falls back)")
    renders_dir = args.renders_dir or ROOT / "logs" / "rollout" / f"deploy_{sku}_{view}"
    renders_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # start = raw NN inheritance (deployment does not perturb, no jitter)
    hit = retrieve_mod.retrieve(sku)
    seed_path = hit["seed"] if hit else ROOT / "seeds" / "template_generic.json"
    seed_cfgs = core.load_seed(seed_path)
    inherit = seed_cfgs.get(view) or next(iter(seed_cfgs.values()))
    inherit_tanh = core.to_tanh(inherit).astype(np.float32)

    static = np.concatenate([
        np.load(emb_path),  # 2052
        np.asarray(VIEW_ONEHOT[view], np.float32),
        np.asarray([0.0, 1.0], np.float32),  # no jitter: sin0/cos0
    ])
    assert static.shape == (2056,)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor = load_actor(args.actor, device)
    enc_model, enc_prep = _get_encoder()
    rm_encode, rm_head, rm_mean, rm_std, rm_src = _get_rm(args.rm_ckpt, device)

    worker = Worker(args.port, args.device)
    try:
        worker.cmd({"cmd": "set_episode", "sku": sku, "view": view,
                    "cam_jitter": 0.0, "prod_z": 0.0})

        # step-1: deterministic mean action → preview-tier render → ResNet18 embedding
        obs1 = np.concatenate([static, inherit_tanh, np.zeros(EMB_DIM, np.float32)])
        a1 = np.clip(mean_action(actor, obs1, device), -1.0, 1.0)
        preview_png = renders_dir / "step1_preview.png"
        worker.cmd({"cmd": "render", "config": core.from_tanh(a1),
                    "tier": "preview", "out": str(preview_png)})
        emb = embed_image(enc_model, enc_prep, preview_png).astype(np.float32)

        # step-2: sample N candidates from the same obs2 → one train-tier render each → RM adjudication
        obs2 = np.concatenate([static, a1, emb])
        cand_pngs, cand_cfgs = [], []
        for i in range(args.n):
            a2 = np.clip(sample_action(actor, obs2, device), -1.0, 1.0)
            cfg = core.from_tanh(a2)
            png = renders_dir / f"cand_{i:02d}.png"
            worker.cmd({"cmd": "render", "config": cfg, "tier": "train", "out": str(png)})
            cfg_path = renders_dir / f"cand_{i:02d}_config.json"
            core.save_config(cfg, cfg_path)
            cand_pngs.append(png)
            cand_cfgs.append(cfg_path)
            print(f"[{i + 1}/{args.n}] candidate rendered {png.name}", flush=True)
        raw = rank_mod.score_images(rm_encode, rm_head, cand_pngs, device)
        z = (raw - rm_mean) / max(rm_std, 1e-6)

        # do-no-harm gate (E6 preregistered, validated 2026-09-10): the raw inherited config goes through
        # the same RM; keep the inherited lighting when Δz = z(winner) − z(inherit) does not exceed --gate-tau
        i_win = int(np.argmax(z))
        inherit_png = renders_dir / "inherit.png"
        inherit_cfg_path = renders_dir / "inherit_config.json"
        worker.cmd({"cmd": "render", "config": inherit, "tier": "train", "out": str(inherit_png)})
        core.save_config(inherit, inherit_cfg_path)
        raw_i = rank_mod.score_images(rm_encode, rm_head, [inherit_png], device)
        z_inherit = float((raw_i[0] - rm_mean) / max(rm_std, 1e-6))
    finally:
        worker.close()

    order = np.argsort(-z)
    candidates = [{"index": int(i), "score_z": float(z[i]), "score_raw": float(raw[i]),
                   "png": str(cand_pngs[i]), "config_path": str(cand_cfgs[i])} for i in order]
    best = candidates[0]
    gate_applied = (float(z[i_win]) - z_inherit) <= args.gate_tau
    if gate_applied:  # conservative mode: the correction's relative edge is not enough, keep the inherited lighting
        win = {"index": -1, "score_z": z_inherit, "score_raw": float(raw_i[0]),
               "png": str(inherit_png), "config_path": str(inherit_cfg_path),
               "note": "gate: kept inherited light"}
    else:
        win = best
    result = {"sku": sku, "view": view, "actor": str(args.actor), "rm": rm_src,
              "inherit_from": Path(seed_path).stem, "n": args.n,
              "preview_png": str(preview_png),
              "winner": win,
              "gate": {"tau": args.gate_tau, "applied": bool(gate_applied),
                       "winner_z": float(z[i_win]), "inherit_z": z_inherit,
                       "delta_z": float(z[i_win]) - z_inherit,
                       "inherit_png": str(inherit_png),
                       "inherit_config_path": str(inherit_cfg_path)},
              "candidates": candidates}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=1))
    if gate_applied:
        print(f"winner: kept inherited light (gate: Δz {result['gate']['delta_z']:+.3f} ≤ τ {args.gate_tau}), "
              f"inherited from {result['inherit_from']}")
    else:
        print(f"winner: cand_{win['index']:02d} z={win['score_z']:+.3f}"
              f"(Δz {result['gate']['delta_z']:+.3f} > τ {args.gate_tau}, inherited from {result['inherit_from']})")
    print(f"result.json -> {args.out}")


if __name__ == "__main__":
    main()
