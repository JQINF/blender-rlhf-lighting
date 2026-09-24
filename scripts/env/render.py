"""Render tiers + device selection + single-frame render.

Tiers:
  train    Cycles 256²@25spp+OIDN  — training-pipeline standard; every consumer (record
           backfill / perturb / silver-label / RM / SAC / canary / best-of-N) uses this tier
  preview  Cycles 256²@8spp+OIDN   — step-1 intermediate preview; feeds the embedding
           backbone only, the RM never scores it
  pair512  Cycles 512²@25spp+OIDN  — human-viewing only (blind-label pair images);
           RM encoder input is always 224² — the extra
           resolution is for blind-label detail judging only, not the reward loop
  embed    Cycles 512²@100spp+OIDN — one-off render for embedding extraction (hand-tuned,
           frozen embed_stage.blend rig; Cycles because the rig was tuned via Cycles F12);
           not used inside the training loop
  final    Cycles 2048²@512spp     — delivery render (production tier only after a winner is picked)
All five tiers render through the compositor (film_transparent + AlphaOver onto white,
pinned in stage.blend; the embed tier uses the same setup copied into embed_stage.blend).

Reuse from other scripts (callers import it this way): importlib-load the module, then
  render_mod.setup_tier("train") / render_mod.render_to(path)

Standalone single-frame render (run import_product.py first to mount the
product; both scripts share post-`--` args, so tier/output are flags, not positional):
  blender -b scene/stage.blend --python scripts/env/import_product.py \\
      --python scripts/env/render.py -- 13.blend --tier train --out /tmp/t.png [--device OPTIX|CUDA|CPU]
"""

import sys
import time
from pathlib import Path

import bpy

TIERS = {
    "train":   dict(engine='CYCLES', res=256, samples=25, denoise=True),
    "preview": dict(engine='CYCLES', res=256, samples=8, denoise=True),
    "pair512": dict(engine='CYCLES', res=512, samples=25, denoise=True),
    "embed":   dict(engine='CYCLES', res=512, samples=100, denoise=True),
    "final":   dict(engine='CYCLES', res=2048, samples=512, denoise=True),
}


def set_cycles_device(prefer='OPTIX'):
    """OPTIX → CUDA → CPU fallback chain; returns the actually selected device type.
    Do not call refresh_devices() — it empties the device list."""
    scene = bpy.context.scene
    prefs = bpy.context.preferences.addons['cycles'].preferences
    candidates = {'OPTIX': ['OPTIX', 'CUDA', 'CPU'],
                  'CUDA': ['CUDA', 'CPU'],
                  'CPU': ['CPU']}[prefer.upper()]
    for dev in candidates:
        if dev == 'CPU':
            scene.cycles.device = 'CPU'
            return 'CPU'
        try:
            prefs.compute_device_type = dev
        except TypeError:
            continue
        # prefs.get_devices() returns empty/None; the prefs.devices property
        # is the one that actually lists GPUs
        gpus = [d for d in prefs.devices if d.type == dev]
        if not gpus:
            continue
        for d in gpus:
            d.use = True
        scene.cycles.device = 'GPU'
        return dev
    scene.cycles.device = 'CPU'
    return 'CPU'


def setup_tier(tier, device='OPTIX'):
    """Set render params per tier; returns the actual render device ('n/a' for a non-Cycles
    tier). Camera/lights/product untouched."""
    cfg = TIERS[tier]
    scene = bpy.context.scene
    scene.render.engine = cfg["engine"]
    scene.render.resolution_x = scene.render.resolution_y = cfg["res"]
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.render.film_transparent = True
    if cfg["engine"] != 'CYCLES':
        return 'n/a'
    scene.cycles.samples = cfg["samples"]
    scene.cycles.use_denoising = cfg["denoise"]
    # Pin the integrator noise seed: default seed=0 randomizes every render → same-config
    # re-renders drift in pixels/RM z (can flip a borderline gate). Pinned = byte-reproducible.
    scene.cycles.seed = 1
    try:
        scene.cycles.use_animated_seed = False
    except AttributeError:
        pass
    if cfg["denoise"]:
        try:
            scene.cycles.denoiser = 'OPENIMAGEDENOISE'
        except TypeError:
            pass
    return set_cycles_device(device)


def render_to(path):
    """Render current scene to path (creates parent dirs); returns elapsed seconds."""
    path = str(path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    bpy.context.scene.render.filepath = path
    t0 = time.time()
    bpy.ops.render.render(write_still=True)
    dt = time.time() - t0
    print(f"render_to: {path}  {dt:.2f}s")
    return dt


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []

    def _flag(name, default=None):
        return argv[argv.index(name) + 1] if name in argv else default

    tier = _flag("--tier", "train")
    out = _flag("--out", f"/tmp/render_{tier}.png")
    device = _flag("--device", "OPTIX")
    if tier not in TIERS:
        sys.exit(f"unknown tier {tier!r}, options: {list(TIERS)}")
    chosen = setup_tier(tier, device)
    dt = render_to(out)
    print(f"tier={tier} device={chosen} took={dt:.2f}s -> {out}")
