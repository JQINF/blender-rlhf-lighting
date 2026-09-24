"""ops.py — [Auto-light] modal operator: subprocess-calls scripts/deploy_light.py, polls the contract, applies the winner.

Flow: validate (model/<sku>.blend + data/embeddings/<sku>.npy exist) → Popen the venv deploy script
(env adds HF_HUB_OFFLINE=1; stdout/stderr go to logs/rollout/tmp/deploy_<sku>_<view>.log)
→ modal timer polls at 0.5s (UI not frozen, total ~1.5-2.5 min) → on completion read result.json:
  - if the product in the scene is not the target sku, swap it in via import_product.load_product
    (skipped if lighting_rl already swapped in the same sku)
  - camera switched to the target view (camera.apply_cam, same convention as candidate rendering)
  - lighting_rl.core.load_config(winner config_path) → core.apply (schema validation + anti-clip
    clamping all go through core as the single authority)
  - winner render loaded into bpy.data.images for inspection
"""

import importlib.util
import json
import os
import subprocess
from pathlib import Path

import bpy

from lighting_rl import core  # the 72-dim schema's single authority (both addons live in the same addons dir, direct import)


def _root():
    """Project root. Candidates in order: LRL_PROJECT_ROOT env var → project_root.txt shipped with
    the addon (locates the installed copy) → two levels up from the current .blend → three levels up
    from this file (in-place repo loading). Each candidate is validated by the existence of scripts/env/."""
    candidates = []
    env = os.environ.get("LRL_PROJECT_ROOT")
    if env:
        candidates.append(Path(env))
    txt = Path(__file__).resolve().parent / "project_root.txt"
    if txt.is_file():
        candidates.append(Path(txt.read_text().strip()))
    p = Path(bpy.data.filepath)
    if p.name:
        candidates.append(p.resolve().parent.parent)
    candidates.append(Path(__file__).resolve().parents[2])
    for c in candidates:
        if (c / "scripts" / "env").is_dir():
            return c
    raise FileNotFoundError(
        "Cannot locate the project root (tried: " + ", ".join(str(c) for c in candidates) + "). "
        "Set the LRL_PROJECT_ROOT env var, or check project_root.txt in the addon directory")


def _load(alias, rel):
    spec = importlib.util.spec_from_file_location(alias, _root() / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class LRD_OT_auto_light(bpy.types.Operator):
    bl_idname = "lrd.auto_light"
    bl_label = "Auto Light (best-of-N)"

    _timer = None
    _proc = None
    _log_f = None
    _result_path = None
    _log_path = None
    _sku = None

    def execute(self, context):
        sc = context.scene
        sku = sc.lrd_sku.strip()
        if not sku:
            self.report({'ERROR'}, "Enter a product id first")
            return {'CANCELLED'}
        try:
            root = _root()
        except FileNotFoundError as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        if not (root / "model" / f"{sku}.blend").is_file():
            self.report({'ERROR'}, f"model/{sku}.blend not found")
            return {'CANCELLED'}
        if bpy.data.objects.get("light_01") is None:
            # studio not open (empty scene/other project): opening the file rebuilds the context and
            # terminates this operator, so the user has to press the button once more after it opens
            bpy.ops.wm.open_mainfile(filepath=str(root / "scene" / "stage.blend"))
            return {'FINISHED'}
        if not (root / "data" / "embeddings" / f"{sku}.npy").is_file():
            self.report({'ERROR'},
                        f"Embedding missing: data/embeddings/{sku}.npy (run extract_embeddings first)")
            return {'CANCELLED'}
        venv_py = root / ".venv" / "bin" / "python"
        if not venv_py.is_file():
            self.report({'ERROR'}, f"venv python not found: {venv_py}")
            return {'CANCELLED'}

        tmp = root / "logs" / "rollout" / "tmp"
        tmp.mkdir(parents=True, exist_ok=True)
        self._result_path = tmp / f"deploy_{sku}_{sc.lrd_view}_result.json"
        self._log_path = tmp / f"deploy_{sku}_{sc.lrd_view}.log"
        self._sku = sku
        cmd = [str(venv_py), str(root / "scripts" / "deploy_light.py"),
               "--sku", sku, "--view", sc.lrd_view, "--n", str(sc.lrd_n),
               "--out", str(self._result_path)]
        env = dict(os.environ, HF_HUB_OFFLINE="1")
        self._log_f = open(self._log_path, "w")
        self._proc = subprocess.Popen(cmd, cwd=str(root), env=env,
                                      stdout=self._log_f, stderr=subprocess.STDOUT, text=True)
        sc.lrd_status = f"Running... {sku}_{sc.lrd_view} (~1.5-2.5 min, logs in tmp dir)"
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.5, window=context.window)
        wm.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        if event.type != 'TIMER':
            return {'PASS_THROUGH'}
        rc = self._proc.poll()
        if rc is None:
            return {'PASS_THROUGH'}
        context.window_manager.event_timer_remove(self._timer)
        self._log_f.close()
        sc = context.scene
        if rc != 0:
            tail = "".join(Path(self._log_path).read_text().splitlines(keepends=True)[-5:])
            sc.lrd_status = "Failed"
            self.report({'ERROR'}, f"Deploy script exited with rc={rc}, log tail: {tail}")
            return {'CANCELLED'}
        try:
            result = json.loads(self._result_path.read_text())
            win = result["winner"]
            cfg = core.load_config(win["config_path"])  # schema validation lives in core
            # swap the product in first if the scene has the wrong one (skipped if lighting_rl already swapped in the same sku)
            if getattr(sc, "lrl_loaded_sku", "") != self._sku:
                import_mod = _load("rl_import_product", "scripts/env/import_product.py")
                import_mod.load_product(_root(), self._sku)
                if hasattr(sc, "lrl_loaded_sku"):
                    sc.lrl_loaded_sku = self._sku
            camera_mod = _load("rl_camera", "scripts/env/camera.py")
            camera_mod.apply_cam(result["view"], 0.0)
            core.apply(cfg)
            if hasattr(sc, "lrl_view"):
                sc.lrl_view = result["view"]
            bpy.data.images.load(win["png"], check_existing=True)  # viewable by switching in the image editor
        except Exception as e:
            sc.lrd_status = "Apply failed"
            self.report({'ERROR'}, f"result contract/apply error: {type(e).__name__}: {e}")
            return {'CANCELLED'}
        gate = result.get("gate") or {}
        if gate.get("applied"):
            sc.lrd_status = (f"Conservative mode: keeping inherited light (Δz={gate.get('delta_z', float('nan')):+.3f} "
                             f"≤ τ={gate.get('tau')}; inherited from {result['inherit_from']})")
        else:
            sc.lrd_status = (f"winner #{win['index']} z={win['score_z']:+.3f} "
                             f"(inherited from {result['inherit_from']}, {result['n']} candidates"
                             + (f"，Δz={gate['delta_z']:+.3f}" if gate else "") + "）")
        self.report({'INFO'}, f"Auto light done: winner z={win['score_z']:+.3f}, "
                              f"renders and config in logs/rollout/deploy_{self._sku}_{result['view']}/")
        return {'FINISHED'}


CLASSES = (LRD_OT_auto_light,)
