"""ops.py — recording-workflow operators (buttons for the 5-step flow).

All operators only dispatch into core.py / scripts/env modules + nn_retrieve.retrieve; no business logic:
  [Fetch product]     fetch_product   — NN retrieval (falls back to generic template) + swap product in
  [Apply inherited]   apply_inherited — apply inherited/template seed for the current view via core.apply
  [Load snapshot]     load_snapshot   — explicitly load the on-disk final/before snapshot (current view) and apply,
                                        syncing the session stash so view switches don't resurrect stale state
  [Switch view+prev]  switch_view     — front⇄high + restore target-view lighting (session stash > final snapshot
                                        > before snapshot > inherited config) + low-tier preview render
  [Snapshot]          snapshot        — core.export to disk (records before if absent, else final)
  [Save as seed]      save_seed       — merge both views' final snapshots into seeds/<sku>.json
Safety: fetch_product asks for confirmation first while lrl_dirty (unsaved-seed changes).

_SESSION_STASH: session-level lighting stash {(sku, view): cfg}. Current scene state is stashed automatically
on view switch and restored first when switching back — round-tripping both views never loses unsnapshotted
work. Not persisted to disk; snapshot semantics unchanged; invalidated on product swap / re-apply inherited.
"""

import importlib.util
import json
import os
from pathlib import Path

import bpy
from bpy.props import BoolProperty, EnumProperty, StringProperty

from . import core

_SESSION_STASH = {}


def _root():
    """Project root. Candidates in order: LRL_PROJECT_ROOT env var (explicit override) → project_root.txt
    shipped with the addon (locates the installed copy) → two levels up from the current .blend
    (stage/model projects both sit one level under root) → three levels up from this file
    (when the addon is loaded in-place at <root>/addons/lighting_rl/ops.py).
    Each candidate is validated by the existence of scripts/env/; if all fail, raise — never
    fall back to cwd silently: the cwd of a GUI-launched Blender is unpredictable."""
    candidates = []
    env = os.environ.get("LRL_PROJECT_ROOT")
    if env:
        candidates.append(Path(env))
    txt = Path(__file__).resolve().parent / "project_root.txt"  # locates the installed copy shipped with the addon
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
        "Open scene/stage.blend first, or set the LRL_PROJECT_ROOT env var")


def _load(alias, rel):
    """importlib path-based loader for the project's flat script modules"""
    spec = importlib.util.spec_from_file_location(alias, _root() / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _root_or_report(op):
    """op wrapper for _root(): failure becomes a red error report and None instead of a traceback"""
    try:
        return _root()
    except FileNotFoundError as e:
        op.report({'ERROR'}, str(e))
        return None


def register_props():
    sc = bpy.types.Scene
    sc.lrl_sku = StringProperty(name="Product id", description="id of model/<id>.blend, e.g. 13")
    sc.lrl_inherited_path = StringProperty(name="Inherited from", default="")
    sc.lrl_loaded_sku = StringProperty(default="")  # actually-swapped-in product sku (guards against editing the sku box without fetching)
    sc.lrl_dirty = BoolProperty(default=False)  # unsaved-seed changes pending


def unregister_props():
    sc = bpy.types.Scene
    for p in ("lrl_sku", "lrl_inherited_path", "lrl_loaded_sku", "lrl_dirty"):
        if hasattr(sc, p):
            delattr(sc, p)


class LRL_OT_fetch_product(bpy.types.Operator):
    bl_idname = "lrl.fetch_product"
    bl_label = "Retrieve and load product"
    bl_options = {'REGISTER'}

    def invoke(self, context, event):
        if context.scene.lrl_dirty:
            return context.window_manager.invoke_confirm(
                self, event, title="There are unsaved changes", message="Loading a new product discards unsaved changes. Continue?",
                confirm_text="Load", icon='QUESTION')
        return self.execute(context)

    def execute(self, context):
        sc = context.scene
        sku = sc.lrl_sku.strip()
        if not sku:
            self.report({'ERROR'}, "Enter a product id first")
            return {'CANCELLED'}
        root = _root_or_report(self)
        if root is None:
            return {'CANCELLED'}
        if not (root / "model" / f"{sku}.blend").is_file():
            self.report({'ERROR'}, f"model/{sku}.blend not found (training ids 01-37 + test_01-06)")
            return {'CANCELLED'}
        try:
            hit = _load("rl_nn_retrieve", "scripts/nn_retrieve.py").retrieve(sku)
        except FileNotFoundError as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        if hit:
            sc.lrl_inherited_path = str(hit["seed"])
            src = f"seed {hit['sku']} (distance {hit['distance']:.3f})"
        else:
            sc.lrl_inherited_path = str(root / "seeds" / "template_generic.json")
            src = "generic template (no candidate neighbours)"
        import_mod = _load("rl_import_product", "scripts/env/import_product.py")
        try:
            names = import_mod.load_product(root, sku)
        except Exception as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        _SESSION_STASH.pop((sku, "front"), None)  # product swap = new session; old stash invalidated
        _SESSION_STASH.pop((sku, "high"), None)
        sc.lrl_loaded_sku = sku
        sc.lrl_dirty = False
        self.report({'INFO'}, f"Loaded {sku} ({len(names)} objects), inherited from {src}")
        return {'FINISHED'}


class LRL_OT_apply_inherited(bpy.types.Operator):
    bl_idname = "lrl.apply_inherited"
    bl_label = "Apply inherited config (current view)"
    bl_options = {'REGISTER'}

    def execute(self, context):
        sc = context.scene
        inherited = sc.lrl_inherited_path
        if not inherited:
            root = _root_or_report(self)
            if root is None:
                return {'CANCELLED'}
            inherited = str(root / "seeds" / "template_generic.json")
        path = inherited
        try:
            seed = core.load_seed(path)
            cfg = seed[sc.lrl_view]
            core.apply(cfg)
        except Exception as e:
            self.report({'ERROR'}, f"{path}: {e}")
            return {'CANCELLED'}
        sc.lrl_dirty = True  # lighting session started (product swap now needs confirmation)
        sku = sc.lrl_loaded_sku or sc.lrl_sku.strip()
        if sku:  # explicit return to the inherited starting point: stash reset in sync, so view switches don't resurrect stale state
            _SESSION_STASH[(sku, sc.lrl_view)] = cfg
        self.report({'INFO'}, f"Applied {Path(path).name} for view {sc.lrl_view}")
        return {'FINISHED'}


class LRL_OT_load_snapshot(bpy.types.Operator):
    bl_idname = "lrl.load_snapshot"
    bl_label = "Load snapshot from disk (current view)"
    bl_options = {'REGISTER'}

    kind: EnumProperty(items=[("final", "final", ""), ("before", "before", "")], default="final")

    def execute(self, context):
        sc = context.scene
        sku = sc.lrl_sku.strip()
        if not sku:
            self.report({'ERROR'}, "Enter a product id first")
            return {'CANCELLED'}
        if not sc.lrl_loaded_sku:
            self.report({'ERROR'}, "Press [Retrieve and load product] first — the snapshot's product is not in the scene")
            return {'CANCELLED'}
        if sku != sc.lrl_loaded_sku:
            self.report({'ERROR'},
                        f"id field ({sku}) does not match the loaded product ({sc.lrl_loaded_sku}) — "
                        "press [Retrieve and load product], otherwise the config lands on the wrong product")
            return {'CANCELLED'}
        root = _root_or_report(self)
        if root is None:
            return {'CANCELLED'}
        p = root / "data" / "snapshots" / f"{sku}_{sc.lrl_view}_{self.kind}.json"
        if not p.is_file():
            self.report({'ERROR'}, f"{self.kind} snapshot not found: {p.name}")
            return {'CANCELLED'}
        try:
            cfg = json.loads(p.read_text())["config"]
            core.apply(cfg)
        except Exception as e:
            self.report({'ERROR'}, f"{p.name}: {e}")
            return {'CANCELLED'}
        _SESSION_STASH[(sku, sc.lrl_view)] = cfg  # sync the stash so view switches don't resurrect stale state
        sc.lrl_dirty = True
        if self.kind == "before":
            self.report({'INFO'}, f"Loaded before snapshot: {p.name} (for reference; the before file is never overwritten, "
                                  "recording always writes final)")
        else:
            self.report({'INFO'}, f"Loaded final snapshot: {p.name}")
        return {'FINISHED'}


class LRL_OT_switch_view(bpy.types.Operator):
    bl_idname = "lrl.switch_view"
    bl_label = "Toggle view front⇄high + low-tier preview"
    bl_options = {'REGISTER'}

    def execute(self, context):
        sc = context.scene
        sku = sc.lrl_loaded_sku or sc.lrl_sku.strip()
        root = _root_or_report(self)
        if root is None:
            return {'CANCELLED'}
        cur = sc.lrl_view
        # before switching away: stash the current view's scene lighting (snapshotted or not, no in-session work is lost)
        if sku:
            try:
                stash_cfg = core.export()
                overflow = stash_cfg.pop("_aim_overflow", [])
                if overflow:
                    self.report({'WARNING'},
                                f"{'/'.join(overflow)} aim clamped; pose springs back a few degrees when returning to this view")
                _SESSION_STASH[(sku, cur)] = stash_cfg
            except Exception as e:
                self.report({'WARNING'}, f"Failed to stash the current view state: {e}")
        target = "high" if cur == "front" else "front"
        sc.lrl_view = target
        cam_mod = _load("rl_camera", "scripts/env/camera.py")
        cam_mod.apply_cam(target)  # camera pose set from code for deterministic reproduction
        # restore target-view lighting: session stash > final snapshot > before snapshot > inherited config
        src = None
        if sku and (sku, target) in _SESSION_STASH:
            core.apply(_SESSION_STASH[(sku, target)])
            src = "session stash"
        else:
            snap_dir = root / "data" / "snapshots"
            for kind in ("final", "before"):
                p = snap_dir / f"{sku}_{target}_{kind}.json" if sku else None
                if p is not None and p.is_file():
                    core.apply(json.loads(p.read_text())["config"])
                    src = f"{kind} snapshot"
                    break
            if src is None and sc.lrl_inherited_path:
                try:
                    core.apply(core.load_seed(sc.lrl_inherited_path)[target])
                    src = "inherited config"
                except Exception as e:
                    self.report({'WARNING'}, f"Inherited config not applied: {e}")
        render_mod = _load("rl_render", "scripts/env/render.py")
        render_mod.setup_tier("preview")
        bpy.ops.render.render()
        self.report({'INFO'}, f"Current view: {target} (light source: {src or 'untouched'})")
        return {'FINISHED'}


class LRL_OT_snapshot(bpy.types.Operator):
    bl_idname = "lrl.snapshot"
    bl_label = "Record snapshot (current view)"
    bl_options = {'REGISTER'}

    def execute(self, context):
        sc = context.scene
        sku = sc.lrl_sku.strip()
        if not sku:
            self.report({'ERROR'}, "Enter a product id first")
            return {'CANCELLED'}
        if sc.lrl_loaded_sku and sku != sc.lrl_loaded_sku:
            self.report({'ERROR'},
                        f"id field ({sku}) does not match the loaded product ({sc.lrl_loaded_sku}) — "
                        "press [Retrieve and load product], otherwise the snapshot is recorded under the wrong product")
            return {'CANCELLED'}
        root = _root_or_report(self)
        if root is None:
            return {'CANCELLED'}
        snap_dir = root / "data" / "snapshots"
        kind = "before" if not (snap_dir / f"{sku}_{sc.lrl_view}_before.json").exists() else "final"
        try:
            cfg = core.export()  # soft-off slots auto-canonicalized
            overflow = cfg.pop("_aim_overflow", [])  # for the warning only, never saved
        except Exception as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        if overflow:
            self.report({'WARNING'},
                        f"{'/'.join(overflow)} light axis points outside the aim window (aim clamped to ±1); "
                        "writing back springs a few degrees — aim the light a bit more toward the product; off-target poses cannot be reproduced")
        snap_dir.mkdir(parents=True, exist_ok=True)
        out = snap_dir / f"{sku}_{sc.lrl_view}_{kind}.json"
        overwriting = kind == "final" and out.is_file()
        out.write_text(json.dumps({"sku": sku, "view": sc.lrl_view, "kind": kind,
                                   "config": cfg}, ensure_ascii=False, indent=1))
        if overwriting:
            self.report({'WARNING'}, f"Overwrote an existing final snapshot: {out.name}")
        if kind == "final":
            sc.lrl_dirty = True
        self.report({'INFO'}, f"{kind} snapshot written: {out.name}")
        return {'FINISHED'}


class LRL_OT_save_seed(bpy.types.Operator):
    bl_idname = "lrl.save_seed"
    bl_label = "Save as end-point seed (both views)"
    bl_options = {'REGISTER'}

    def execute(self, context):
        sc = context.scene
        sku = sc.lrl_sku.strip()
        if not sku:
            self.report({'ERROR'}, "Enter a product id first")
            return {'CANCELLED'}
        if sc.lrl_loaded_sku and sku != sc.lrl_loaded_sku:
            self.report({'ERROR'},
                        f"id field ({sku}) does not match the loaded product ({sc.lrl_loaded_sku}) — press [Retrieve and load product] first")
            return {'CANCELLED'}
        root = _root_or_report(self)
        if root is None:
            return {'CANCELLED'}
        snap_dir = root / "data" / "snapshots"
        finals = {}
        for view in ("front", "high"):
            p = snap_dir / f"{sku}_{view}_final.json"
            if not p.is_file():
                self.report({'ERROR'}, f"missing final snapshot for {view} (switch to {view}, tune it, then record final)")
                return {'CANCELLED'}
            finals[view] = json.loads(p.read_text())["config"]
        out = root / "seeds" / f"{sku}.json"
        core.save_seed(out, front=finals["front"], high=finals["high"])
        sc.lrl_dirty = False
        self.report({'INFO'}, f"Seed saved: seeds/{sku}.json (front + high)")
        return {'FINISHED'}


CLASSES = (LRL_OT_fetch_product, LRL_OT_apply_inherited, LRL_OT_load_snapshot, LRL_OT_switch_view,
           LRL_OT_snapshot, LRL_OT_save_seed)
