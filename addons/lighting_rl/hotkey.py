"""hotkey.py — Ctrl+Alt+S = take snapshot (inside the 3D view)."""

import bpy

_keys = []


def register():
    wm = bpy.context.window_manager
    kc = wm.keyconfigs.addon
    if kc is None:  # headless has no keyconfig; skip silently
        return
    km = kc.keymaps.new(name="3D View", space_type='VIEW_3D')
    kmi = km.keymap_items.new("lrl.snapshot", 'S', 'PRESS', ctrl=True, alt=True)
    _keys.append((km, kmi))


def unregister():
    for km, kmi in _keys:
        km.keymap_items.remove(kmi)
    _keys.clear()
