"""normalize_product.py — normalize a source .blend (onboarding contract: longest edge 1.0m
+ bbox center at origin).

Normalization convention: per mesh
object, obj.scale *= s, obj.location = (location - c) * s — scale+translate about the world
center, NO transform_apply (applying on multi-part projects misaligns the parts; object-level
scale/location travels via matrix_world and reads identically across render/embed/validate).

Orientation is NOT judged — front-facing -Y is a semantic call; verify visually after running.

Usage:
  blender -b model/55.blend --python-exit-code 1 --python scripts/env/normalize_product.py
"""

import sys
from pathlib import Path

import bpy
from mathutils import Vector


def main():
    coll = bpy.data.collections.get("product")
    if coll is None:
        sys.exit("source file has no 'product' collection (intake contract)")
    objs = [ob for ob in coll.all_objects if ob.type == "MESH"]
    if not objs:
        sys.exit("'product' collection contains no mesh")
    others = [ob.name for ob in coll.all_objects if ob.type != "MESH"]
    if others:
        print(f"WARNING: 'product' collection contains non-mesh objects (excluded from normalisation; verify by eye): {others}")
    bpy.context.view_layer.update()

    def world_bbox():
        pts = [ob.matrix_world @ Vector(c) for ob in objs for c in ob.bound_box]
        xs = [p.x for p in pts]
        ys = [p.y for p in pts]
        zs = [p.z for p in pts]
        size = (max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))
        center = Vector(((max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2, (max(zs) + min(zs)) / 2))
        return size, center

    size, center = world_bbox()
    s = 1.0 / max(size)
    for ob in objs:
        ob.scale = Vector(ob.scale) * s
        ob.location = (Vector(ob.location) - center) * s
    bpy.context.view_layer.update()

    new_size, new_center = world_bbox()
    assert abs(max(new_size) - 1.0) <= 1e-4, f"longest edge after normalisation {max(new_size)}"
    assert new_center.length <= 1e-4, f"centre after normalisation {tuple(new_center)}"
    bpy.ops.wm.save_mainfile()
    print(f"NORMALIZE: {Path(bpy.data.filepath).name} original size={tuple(round(v, 3) for v in size)} "
          f"-> 1.0m, centre at origin, saved (verify the front faces -Y by eye)")


main()
