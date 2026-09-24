"""Blender-side render helper for embedding extraction (invoked as a subprocess by
scripts/extract_embeddings.py — do not run by hand).

Does the following against the current embed_stage:
  1. Swap in the product from model/<sku>.blend (reuses import_product.load_product; the
     placeholder product pre-mounted in embed_stage gets replaced automatically)
  2. Never touch lights/world — scene/embed_stage.blend IS the embedding rig (hand-tuned in
     the GUI and frozen; changing it invalidates every cached embedding and forces re-extraction).
     Do NOT reuse stage.blend: its world Background Color input is is_linked (occupied by a
     Hue/Sat ← environment-map chain), so script-side assignments are silent no-ops and batch
     renders pick up directional light from the environment map (front bright, sides dark)
  3. Render 4 images at Cycles 512²@100spp+OIDN (render.py "embed" tier, matching the hand-tuned
     rig; clay via in-process view_layer.material_override, reset to None before the neutral
     render) into data/embeddings/img/:
       <sku>_clay_{front,side,top}.png  — material-neutral override (clay), 3-view shape channel
       <sku>_neutral.png                — real materials, front view, material/appearance channel
  4. Write data/embeddings/img/<sku>_bbox.json (3-dim bbox size, for the state vector)

Usage: blender -b scene/embed_stage.blend --python scripts/env/embed_views.py -- <sku>
"""

import importlib.util
import json
import sys
from math import radians
from pathlib import Path

import bpy

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
sku = argv[0].removesuffix(".blend") if argv else "13"

blend_path = Path(bpy.data.filepath)
project_root = blend_path.resolve().parent.parent if blend_path.name else Path.cwd()


def load_env_mod(alias, name):
    spec = importlib.util.spec_from_file_location(alias, project_root / "scripts" / "env" / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


import_mod = load_env_mod("rl_import_product", "import_product.py")
render_mod = load_env_mod("rl_render", "render.py")

names = import_mod.load_product(project_root, sku)
print(f"loaded product {sku}: {names}")

# ---- bbox size (for the state vector; world coordinates) ----
bpy.context.view_layer.update()
coll = bpy.context.scene.collection.children['product']
sys.path.insert(0, str(project_root / "addons"))
from lighting_rl import core

center, R, bottom_z = core.get_product_sphere()
# get_product_sphere returns only center/R/bottom; compute size separately
# (the 3-view embedding state needs the 3-dim bbox size)
from mathutils import Vector
pts = [ob.matrix_world @ Vector(c) for ob in coll.objects if ob.type == 'MESH' for c in ob.bound_box]
size = [max(p[i] for p in pts) - min(p[i] for p in pts) for i in range(3)]

out_dir = project_root / "data" / "embeddings" / "img"
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / f"{sku}_bbox.json").write_text(json.dumps({"bbox_size": size}))

# ---- lights/world: untouched — embed_stage.blend itself is the rig (frozen contract) ----

# ---- clay material override ----
clay = bpy.data.materials.new("clay_override")
clay.use_nodes = True
bsdf = clay.node_tree.nodes["Principled BSDF"]
bsdf.inputs["Base Color"].default_value = (0.8, 0.8, 0.8, 1)
bsdf.inputs["Roughness"].default_value = 0.9
# IOR 1.45 default specular leaves bright spots on clay; suppress it for a closer matte-clay look
if "Specular IOR Level" in bsdf.inputs:
    bsdf.inputs["Specular IOR Level"].default_value = 0.0

view_layer = bpy.context.view_layer
cam = bpy.data.objects['Camera']

# clay 3-view poses: front / side / top (same 10m radius, 280mm lens, aimed at origin —
# framing matches CAM_FRONT; keep in sync when CAM_FRONT's constant changes)
CLAY_VIEWS = {
    "front": ((0, -10.0, 0), (radians(90), 0, 0)),
    "side":  ((10.0, 0, 0), (radians(90), 0, radians(90))),
    "top":   ((0, 0, 10.0), (0, 0, 0)),
}

render_mod.setup_tier("embed")  # embed tier: Cycles 512²@100spp+OIDN, compositor white-bg output unchanged

view_layer.material_override = clay
for view, (loc, rot) in CLAY_VIEWS.items():
    cam.location = loc
    cam.rotation_mode = 'XYZ'
    cam.rotation_euler = rot
    cam.data.lens = 280  # same focal length as CAM_FRONT
    render_mod.render_to(out_dir / f"{sku}_clay_{view}.png")

# ---- neutral render: real materials + front pose ----
view_layer.material_override = None
cam.location = CLAY_VIEWS["front"][0]
cam.rotation_euler = CLAY_VIEWS["front"][1]
render_mod.render_to(out_dir / f"{sku}_neutral.png")

print(f"embed_views done: {sku} -> {out_dir}")
