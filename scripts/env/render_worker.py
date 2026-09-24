"""render_worker.py — resident Blender render worker.

The training loop (scripts/train/refine_env.py) connects over TCP JSON-lines,
one command per line:
  {"cmd":"set_episode","sku":"13","view":"front","cam_jitter":3.2,"prod_z":-5.1}
      swap product (same sku is reused, cur_sku pattern) → position
      camera (apply_cam includes ±10° orbit jitter) → rotate the product collection
      incrementally around origin Z (current angle tracked to avoid accumulation; reset
      on product swap)
  {"cmd":"render","config":<72-dim dict>,"tier":"preview"|"train"|"pair512","out":<path>[, "strip_albedo":true]}
      core.apply → render.setup_tier → render_to; replies {"ok":true,"seconds":...}
      strip_albedo = strip albedo, keep lighting structure (fragmented-light probe only:
      Base Color → mid-gray, specular preserved)
  {"cmd":"ping"} → {"ok":true,"pong":true}; {"cmd":"shutdown"} → reply, then exit
One JSON reply per command; on command error replies {"ok":false,"error":...} and resets
cur_sku (avoid half-swapped state). A dropped connection returns to
accept, so a crashed training side can reconnect for inspection.

Benchmark rule (logs/benchmark.txt): the whole training pipeline reuses this session —
never spawn a per-frame process (cold start ~2.3-2.6s vs ~0.07s/frame amortized).

Usage:
  blender -b scene/stage.blend --python scripts/env/render_worker.py -- [--port 17871] [--device OPTIX|CUDA|CPU]
"""

import importlib.util
import json
import socket
import sys
from math import radians
from pathlib import Path

import bpy
from mathutils import Matrix

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def opt(name, default=None):
    return argv[argv.index(name) + 1] if name in argv else default


PORT = int(opt("--port", "17871"))
DEVICE = opt("--device", "OPTIX").upper()

blend_path = Path(bpy.data.filepath)
project_root = blend_path.resolve().parent.parent if blend_path.name else Path.cwd()


def load_env_mod(alias, name):
    """importlib-load a flat script from scripts/env/ (these only self-execute
    inside their __main__ guard)"""
    spec = importlib.util.spec_from_file_location(alias, project_root / "scripts" / "env" / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


camera_mod = load_env_mod("rl_camera", "camera.py")
render_mod = load_env_mod("rl_render", "render.py")
import_mod = load_env_mod("rl_import_product", "import_product.py")

sys.path.insert(0, str(project_root / "addons"))
try:
    from lighting_rl import core
except ImportError:
    sys.exit("missing dependency: addons/lighting_rl/core.py (72-dim config core) not in place; cannot apply configs")

chosen = render_mod.setup_tier("train", DEVICE)  # device selected once at startup (OPTIX→CUDA→CPU fallback chain)
print(f"render_worker: device {chosen}, port {PORT}", flush=True)

cur_sku = None
cur_prod_z = 0.0  # product collection's current Z rotation (deg); reset on product swap (newly linked objects are in original pose)


def set_product_z(deg):
    """Rotate the product collection incrementally around origin Z to deg
    (product center is at origin — spins in place)."""
    global cur_prod_z
    delta = radians(deg - cur_prod_z)
    if abs(delta) < 1e-12:
        return
    rot = Matrix.Rotation(delta, 4, 'Z')
    coll = bpy.context.scene.collection.children.get('product')
    for ob in (list(coll.objects) if coll else []):
        ob.matrix_world = rot @ ob.matrix_world
    cur_prod_z = deg


def handle(cmd):
    global cur_sku, cur_prod_z
    c = cmd.get("cmd")
    if c == "ping":
        return {"ok": True, "pong": True}
    if c == "set_episode":
        sku = str(cmd["sku"])
        if sku != cur_sku:
            try:
                import_mod.load_product(project_root, sku)
            except Exception:
                cur_sku = None  # force re-swap on next command after failure (avoid half-swapped state)
                raise
            cur_sku = sku
            cur_prod_z = 0.0
        # Newly linked objects have dirty matrix_world — refresh before reading; otherwise
        # set_product_z applies on identity and wipes ob.scale (product scales up → black frame)
        bpy.context.view_layer.update()
        camera_mod.apply_cam(cmd["view"], float(cmd.get("cam_jitter", 0.0)))
        set_product_z(float(cmd.get("prod_z", 0.0)))
        return {"ok": True}
    if c == "render":
        core.apply(cmd["config"])
        render_mod.setup_tier(cmd["tier"], DEVICE)  # plain attribute assignment, idempotent
        # strip_albedo: albedo → mid-gray (Base Color unlinked), specular kept alive — fragmented
        # light is a specular artifact; high-frequency albedo (foil/labels) is the confound. Restore after render.
        saved_vals, saved_links = [], []
        if cmd.get("strip_albedo"):
            for mat in bpy.data.materials:
                if not mat.use_nodes:
                    continue
                for n in mat.node_tree.nodes:
                    if n.type == 'BSDF_PRINCIPLED':
                        inp = n.inputs.get("Base Color")
                        if inp is None:
                            continue
                        saved_vals.append((inp, tuple(inp.default_value)))
                        inp.default_value = (0.5, 0.5, 0.5, 1.0)
                        for lk in list(inp.links):
                            saved_links.append((mat.node_tree, lk.from_socket, inp))
                            mat.node_tree.links.remove(lk)
        try:
            dt = render_mod.render_to(cmd["out"])
        finally:
            for inp, val in saved_vals:
                inp.default_value = val
            for nt, fsock, inp in saved_links:
                nt.links.new(fsock, inp)
        # Black-frame guard: RGB all ≈0 = fail fast (a dirty matrix_world once banished the
        # lights → all-black frame; any normal white-bg image has RGB max well above this)
        img = bpy.data.images.load(cmd["out"], check_existing=True)
        import numpy as np
        px = np.asarray(img.pixels[:], dtype=np.float32).reshape(-1, 4)
        bpy.data.images.remove(img)
        if float(px[:, :3].max()) < 2.0 / 255.0:
            return {"ok": False, "error": f"black-frame check: {cmd['out']} is all black (RGB max < 2/255)"}
        return {"ok": True, "seconds": dt}
    if c == "render_mask":
    # True product mask: render the product collection as pure-white emissive, everything else
    # hidden, world black, compositor off. Cached per product+view on the training side.
        scene = bpy.context.scene
        coll = scene.collection.children.get('product')
        prod = set(coll.all_objects) if coll else set()
        white = bpy.data.materials.get("_mask_mat")
        if white is None:
            white = bpy.data.materials.new("_mask_mat")
            white.use_nodes = True
            bsdf = white.node_tree.nodes.get("Principled BSDF")
            bsdf.inputs["Base Color"].default_value = (1, 1, 1, 1)
            bsdf.inputs["Roughness"].default_value = 1.0
            if "Emission Color" in bsdf.inputs:
                bsdf.inputs["Emission Color"].default_value = (1, 1, 1, 1)
                bsdf.inputs["Emission Strength"].default_value = 1.0
        vl = bpy.context.view_layer
        saved_hide, saved_world = {}, None
        rx = scene.render.resolution_x
        ry = scene.render.resolution_y
        use_comp = scene.render.use_compositing
        if scene.world and scene.world.use_nodes:
            bg = scene.world.node_tree.nodes.get("Background")
            if bg is not None:
                saved_world = bg.inputs["Strength"].default_value
        try:
            for ob in scene.objects:
                if ob not in prod and ob.type != 'CAMERA':
                    saved_hide[ob.name] = ob.hide_render
                    ob.hide_render = True
            vl.material_override = white
            if saved_world is not None:
                scene.world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.0
            scene.render.resolution_x = scene.render.resolution_y = 64
            scene.render.use_compositing = False
            render_mod.setup_tier("preview", DEVICE)
            dt = render_mod.render_to(cmd["out"])
        finally:
            for name, v in saved_hide.items():
                ob = bpy.data.objects.get(name)
                if ob is not None:
                    ob.hide_render = v
            vl.material_override = None
            if saved_world is not None:
                scene.world.node_tree.nodes["Background"].inputs["Strength"].default_value = saved_world
            scene.render.resolution_x, scene.render.resolution_y = rx, ry
            scene.render.use_compositing = use_comp
        img = bpy.data.images.load(cmd["out"], check_existing=True)
        import numpy as np
        px = np.asarray(img.pixels[:], dtype=np.float32).reshape(-1, 4)
        bpy.data.images.remove(img)
        if float(px[:, :3].max()) < 2.0 / 255.0:
            return {"ok": False, "error": f"empty mask: {cmd['out']} ('product' collection has no objects?)"}
        return {"ok": True, "seconds": dt}
    if c == "shutdown":
        return {"ok": True, "bye": True}
    if c == "probe":
        # state snapshot for troubleshooting: render config + product/camera/light energies
        scene = bpy.context.scene
        coll = scene.collection.children.get('product')
        return {"ok": True, "engine": scene.render.engine, "film": scene.render.film_transparent,
                "res": scene.render.resolution_x, "samples": scene.cycles.samples,
                "device": scene.cycles.device, "comp": bool(scene.compositing_node_group),
                "view_transform": scene.view_settings.view_transform,
                "product_objs": [ob.name for ob in coll.objects] if coll else [],
                "prod_z": cur_prod_z, "cur_sku": cur_sku,
                "cam_loc": [round(v, 3) for v in bpy.data.objects['Camera'].location],
                "light_E": [round(bpy.data.objects[n].data.energy, 2) for n in core.SLOTS]}
    return {"ok": False, "error": f"unknown command {c!r}"}


def main():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", PORT))
    srv.listen(1)
    print(f"READY {PORT}", flush=True)  # readiness = port connectable (stdout is log-only)
    while True:
        conn, addr = srv.accept()
        print(f"render_worker: connection from {addr}", flush=True)
        f = conn.makefile("rw", encoding="utf-8", newline="\n")
        shutdown = False
        while True:
            line = f.readline()
            if not line:
                break  # peer closed → back to accept, allow reconnect
            try:
                cmd = json.loads(line)
                resp = handle(cmd)
            except Exception as e:
                resp = {"ok": False, "error": f"{type(e).__name__}: {e}"}
            f.write(json.dumps(resp) + "\n")
            f.flush()
            if cmd.get("cmd") == "shutdown":
                shutdown = True
                break
        conn.close()
        if shutdown:
            break
    srv.close()
    print("render_worker: exit", flush=True)


main()
