"""lighting_rl — recording-workflow addon shell.

Structure: core.py (config I/O, no UI deps) + ops.py (5 operators) + panel.py (N panel) + hotkey.py.
Buttons only dispatch into core.py and the env modules under scripts/env; no business logic here.

Install: Blender Preferences → Add-ons → Install from Disk on the addons/lighting_rl directory (or symlink it into the addons dir).
Headless self-test: blender -b scene/stage.blend --python-expr "import sys; sys.path.insert(0,'addons'); import lighting_rl; lighting_rl.register()"
"""

bl_info = {
    "name": "Lighting RL recording workflow",
    "author": "local",
    "version": (0, 1, 0),
    "blender": (5, 2, 0),
    "location": "3D View > N panel > Lighting RL",
    "description": "Nearest-neighbour inheritance + dual-view snapshots/seed recording (buttons only call core.py)",
    "category": "Lighting",
}

try:
    import bpy
except ImportError:  # no bpy on the training side (plain venv python): only core.py's pure-numeric parts are needed; addon registration is skipped
    bpy = None

if bpy is not None:
    from . import hotkey, ops, panel

CLASSES = (ops.CLASSES + panel.CLASSES) if bpy is not None else []


def register():
    assert bpy is not None, "register/unregister only works inside a Blender process"
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    ops.register_props()
    panel.register_props()
    hotkey.register()


def unregister():
    assert bpy is not None, "register/unregister only works inside a Blender process"
    hotkey.unregister()
    panel.unregister_props()
    ops.unregister_props()
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
