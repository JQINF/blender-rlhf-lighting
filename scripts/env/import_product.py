"""Swap in a product: clear the resident product collection + append from model/<sku>.blend
+ orphans purge (prevents mesh-orphan buildup).

No normalization here — source .blend files must already be normalized
(see normalize_product.py); this module only moves objects across.

Usage (headless):
  blender -b scene/stage.blend --python scripts/env/import_product.py -- 13.blend
Reuse from other scripts (validate_scene.py calls it this way):
  importlib-load this file, then call load_product(project_root, "13")  # sku or filename with .blend both work
"""

import sys
from pathlib import Path

import bpy


def load_product(project_root, model_name):
    """Clear the resident product collection, append objects from the 'product' collection
    of model/<model_name>, purge orphans; returns the swapped-in object names."""
    source = Path(project_root) / "model" / model_name
    if not source.suffix:
        source = source.with_suffix(".blend")
    if not source.is_file():
        raise FileNotFoundError(f"model file not found: {source}")
    with bpy.data.libraries.load(str(source)) as (data_from, data_to):
        data_to.collections = ['product']
    imported_coll = data_to.collections[0]
    if imported_coll is None:
        raise RuntimeError(f"{source} has no collection named 'product'")
    scene_col = bpy.context.scene.collection.children['product']
    for ob in list(scene_col.objects):
        bpy.data.objects.remove(ob)
    for ob in imported_coll.objects:
        scene_col.objects.link(ob)
    bpy.data.collections.remove(imported_coll)
    bpy.data.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)
    return [ob.name for ob in scene_col.objects]


def project_root_from_blend():
    """Derive project root from the open stage.blend path: stage.blend lives at
    <project_root>/scene/stage.blend, so the root is two levels up.
    Falls back to cwd when no .blend is open (plain headless --python)."""
    blend_path = Path(bpy.data.filepath)
    if not blend_path or blend_path.name == "":
        return Path.cwd()
    return blend_path.resolve().parent.parent


if __name__ == "__main__":
    # default model filename; a CLI arg after -- wins
    # only take post-`--` args — with no args, sys.argv[-1] is this script's own path
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    model_name = argv[0] if argv else "13.blend"
    names = load_product(project_root_from_blend(), model_name)
    print(f"loaded product {model_name}: {names}")
