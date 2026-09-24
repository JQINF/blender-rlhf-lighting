"""lighting_deploy — deployment-form "auto-lighting" addon shell.

Structure: ops.py (modal operator) + panel.py (N panel). All business logic lives in scripts/deploy_light.py
(venv side: NN inherit → policy 2 correction steps → sample N step-2 candidates → RM adjudication); the addon only
handles subprocess scheduling + result.json contract readback + applying the winner config via core.apply.

Requires the lighting_rl addon to be installed (the 72-dim schema authority core.py is imported from there, not copied).

Install: Blender Preferences → Add-ons → Install from Disk on the addons/lighting_deploy directory.
Headless self-test: blender -b scene/stage.blend --python-expr "import sys; sys.path.insert(0,'addons'); import lighting_deploy; lighting_deploy.register()"
"""

bl_info = {
    "name": "Lighting Deploy auto-lighting",
    "author": "local",
    "version": (0, 1, 0),
    "blender": (5, 2, 0),
    "location": "3D View > N panel > Lighting Deploy",
    "description": "One-click deploy pipeline: NN inheritance -> 2-step policy refine -> best-of-N RM arbitration -> apply winner",
    "category": "Lighting",
}

try:
    import bpy
except ImportError:  # no bpy on the venv side: same guard as lighting_rl
    bpy = None

if bpy is not None:
    from . import ops, panel

CLASSES = (ops.CLASSES + panel.CLASSES) if bpy is not None else []


def register():
    assert bpy is not None, "register/unregister only works inside a Blender process"
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    panel.register_props()


def unregister():
    assert bpy is not None, "register/unregister only works inside a Blender process"
    panel.unregister_props()
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
