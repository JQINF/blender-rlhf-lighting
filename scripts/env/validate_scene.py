"""Scene-contract assertion net (scene spec + intake acceptance).

Usage (headless):
  # single product (run after import_product.py):
  blender -b scene/stage.blend --python-exit-code 1 \
      --python scripts/env/import_product.py --python scripts/env/validate_scene.py -- 13
  # batch-validate every model/*.blend:
  blender -b scene/stage.blend --python-exit-code 1 \
      --python scripts/env/validate_scene.py -- --all
  # add --skip-render to skip the fast-render check (fast, but the white-background chain goes unverified)

Assertions:
  1. product collection exists and contains mesh objects
  2. product world bbox: longest edge 1.0±0.01 and center at origin (±0.01m);
     size/bbox exemptions are registry-driven — validated per entry against the optional
     model/scale_override.json (absent in a fresh clone: every product is then checked strictly)
     (size ±0.01m, bottom z ±0.02m)
  3. product unrotated: hard assert for single-object products (|rotation_euler| < 1e-3 rad
     per axis); multi-part products (straw/pump-head parts etc.) are downgraded to a WARN
     for visual orientation check; skus in ROTATION_OK exempt (orientation compensated in
     geometry, front visually confirmed facing -Y)
  4. light_01..light_08 exact names present (catches .001 duplicate-name accidents), all
     AREA lights with SQUARE shape
  5. scene.camera set, film_transparent on, compositor has an AlphaOver onto-white node
  6. low-tier fast render at 64²: corner pixels >=0.9 and mutually consistent (stage color
     management is ACES 1.3, pure white compresses to ~0.94; black/transparent leak <0.5 —
     thresholds separate the cases) — proves the transparent-film + white-composite chain
     actually works (under --all this runs once after all products pass)
"""

import importlib.util
import json
import re
import sys
import tempfile
from pathlib import Path

import bpy
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ALL = "--all" in argv
SKIP_RENDER = "--skip-render" in argv
pos = [a for a in argv if not a.startswith("--")]
sku_arg = pos[0].removesuffix(".blend") if pos else None

# project root derivation, same rule as import_product.py: stage.blend lives in <root>/scene/
blend_path = Path(bpy.data.filepath)
project_root = blend_path.resolve().parent.parent if blend_path.name else Path.cwd()

# reuse load_product from import_product.py (same import logic, don't duplicate)
_spec = importlib.util.spec_from_file_location(
    "rl_import_product", project_root / "scripts" / "env" / "import_product.py")
import_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(import_mod)

# Intake exemption registry (optional): entries let a product skip the strict size/bbox checks.
# A fresh clone has no registry file — that is fine, all products are then checked strictly.
_ov_path = project_root / "model" / "scale_override.json"
if _ov_path.is_file():
    overrides = json.loads(_ov_path.read_text())
    overrides.pop("_readme", None)
else:
    overrides = {}

    # Rotation exemption list (orientation fixed in the geometry): front must face the -Y camera;
    # the Z rotation is intentional — do not "fix" the source files.
ROTATION_OK = {"09", "18", "25", "28"}

failures = []


def check(ok, label, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""))
    if not ok:
        failures.append(label)


def import_product(source):
    import_mod.load_product(project_root, Path(source).name)


def world_bbox(meshes):
    corners = [ob.matrix_world @ Vector(c) for ob in meshes for c in ob.bound_box]
    xs, ys, zs = [c.x for c in corners], [c.y for c in corners], [c.z for c in corners]
    size = Vector((max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)))
    center = Vector(((max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2, (max(zs) + min(zs)) / 2))
    return size, center, min(zs)


def check_product(sku):
    bpy.context.view_layer.update()  # newly linked objects have dirty matrix_world — refresh before measuring bbox
    coll = bpy.context.scene.collection.children.get('product')
    check(coll is not None, "product collection exists")
    if coll is None:
        return
    objs = list(coll.objects)
    meshes = [ob for ob in objs if ob.type == 'MESH']
    check(len(meshes) > 0, f"{sku} product collection has meshes", f"{len(meshes)} mesh / {len(objs)} obj")
    if not meshes:
        return
    size, center, bottom_z = world_bbox(meshes)
    if sku in overrides:
        exp = overrides[sku]
        ok_size = all(abs(s - e) <= 0.01 for s, e in zip(size, exp["bbox_size_m"]))
        ok_z = abs(bottom_z - exp["bbox_bottom_z"]) <= 0.02
        check(ok_size, f"{sku} exempt bbox size (scale_override entry)",
              f"measured {tuple(round(v, 4) for v in size)} expected {exp['bbox_size_m']}")
        check(ok_z, f"{sku} exempt bottom z",
              f"measured {bottom_z:.4f} expected {exp['bbox_bottom_z']}")
    else:
        check(abs(max(size) - 1.0) <= 0.01, f"{sku} longest edge = 1.0±0.01",
              f"size = {tuple(round(v, 4) for v in size)}")
        check(center.length <= 0.01, f"{sku} bbox centre at origin (±0.01m)",
              f"center = ({center.x:.4f}, {center.y:.4f}, {center.z:.4f})")
    bad_rot = [ob.name for ob in objs if max(abs(a) for a in ob.rotation_euler) > 1e-3]
    if len(objs) == 1:
        if bad_rot and sku in ROTATION_OK:
            print(f"  [PASS] {sku} rotation exempt (front faces -Y verified by eye, rotation = geometry orientation fix): "
                  f"{bad_rot}")
        else:
            # single-object product: object rotation = whole-product rotation, hard assert
            check(not bad_rot, f"{sku} product objects have no rotation", f"non-zero rotation: {bad_rot}" if bad_rot else "")
    elif bad_rot:
        # multi-part product: part angles (straw/pump head) are geometry per se and can't be
        # told apart from "whole product misrotated" — degrade to WARN for visual orientation check
        print(f"  [WARN] {sku} multi-part model with rotated parts (verify overall orientation by eye): {bad_rot}")


def check_stage():
    names = {ob.name: ob for ob in bpy.data.objects}
    want = [f"light_{i:02d}" for i in range(1, 9)]
    missing = [n for n in want if n not in names]
    renamed = sorted(n for n in names if re.fullmatch(r"light_\d+\.\d+", n))
    check(not missing and not renamed, "light_01-08 exactly named",
          f"missing={missing} duplicate.001={renamed}" if (missing or renamed) else "")
    bad = []
    for n in want:
        ob = names.get(n)
        if ob is None:
            continue
        if ob.type != 'LIGHT' or ob.data.type != 'AREA' or ob.data.shape != 'SQUARE':
            bad.append(f"{n}(type={ob.type}"
                       f"{', ' + ob.data.type + '/' + ob.data.shape if ob.type == 'LIGHT' else ''})")
    check(not bad, "all 8 lights are AREA + SQUARE", ", ".join(bad))
    scene = bpy.context.scene
    check(scene.camera is not None, "scene.camera present",
          scene.camera.name if scene.camera else "")
    check(scene.render.film_transparent, "film_transparent on")
    ng = scene.compositing_node_group  # Blender 5.x: compositor node tree lives here (old scene.node_tree removed)
    has_alphaover = ng is not None and any(
        n.bl_idname == 'CompositorNodeAlphaOver' for n in ng.nodes)
    check(has_alphaover, "compositor has an AlphaOver-white node")


def check_white_bg():
    # headless Render Result is not kept in memory (size=0) — write to file, read back
    scene = bpy.context.scene
    res = 64
    scene.render.resolution_x = scene.render.resolution_y = res
    scene.render.resolution_percentage = 100
    scene.cycles.samples = 4
    tmp = Path(tempfile.gettempdir()) / "validate_bg_check.png"
    scene.render.filepath = str(tmp)
    bpy.ops.render.render(write_still=True)
    img = bpy.data.images.load(str(tmp))
    px = img.pixels
    vals = []
    for x, y in ((0, 0), (res - 1, 0), (0, res - 1), (res - 1, res - 1)):
        i = (y * res + x) * 4
        vals.append(tuple(round(v, 3) for v in px[i:i + 3]))
    bpy.data.images.remove(img)
    tmp.unlink()
    ok = all(min(v) >= 0.9 for v in vals) and max(max(v) for v in vals) - min(min(v) for v in vals) <= 0.02
    check(ok, "quick-render corners are pure white (>=0.9 and all four corners consistent; stage uses ACES 1.3, white point ~0.94, so 0.99 would be wrong)",
          f"corners={vals}")


if ALL:
    blends = sorted((project_root / "model").glob("*.blend"))
    print(f"== batch validation of {len(blends)} files ==")
    for b in blends:
        print(f"[{b.stem}]")
        try:
            import_product(b)
        except Exception as e:
            check(False, f"{b.stem} append", str(e))
            continue
        check_product(b.stem)
    print("[stage]")
    check_stage()
else:
    print(f"== single-product validation sku={sku_arg or '(none, default normalisation rules)'} ==")
    check_product(sku_arg)
    check_stage()

if not SKIP_RENDER:
    check_white_bg()

print("=" * 40)
if failures:
    print(f"VALIDATE: FAIL ({len(failures)} items): {failures}")
    sys.exit(1)
print("VALIDATE: PASS")
