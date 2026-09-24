"""panel.py — 3D-viewport N panel (buttons and status only; all logic lives in ops.py / core.py)."""

import bpy
from bpy.props import EnumProperty


def register_props():
    bpy.types.Scene.lrl_view = EnumProperty(
        name="Current view",
        items=[("front", "Front 0°", ""), ("high", "High 45° right", "")],
        default="front")


def unregister_props():
    if hasattr(bpy.types.Scene, "lrl_view"):
        del bpy.types.Scene.lrl_view


class LRL_PT_panel(bpy.types.Panel):
    bl_label = "Lighting RL"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Lighting RL"

    def draw(self, context):
        sc = context.scene
        col = self.layout.column(align=True)
        col.prop(sc, "lrl_sku")
        col.operator("lrl.fetch_product")
        col.operator("lrl.apply_inherited")
        row = col.row(align=True)
        row.operator("lrl.load_snapshot", text=f"Load final ({sc.lrl_view})").kind = "final"
        row.operator("lrl.load_snapshot", text=f"Load before ({sc.lrl_view})").kind = "before"
        col.operator("lrl.switch_view",
                     text=f"Toggle view (current: {sc.lrl_view}) ⇄ + low-tier preview")
        col.operator("lrl.snapshot", text=f"Record snapshot ({sc.lrl_view}, hotkey Ctrl+Alt+S)")
        col.operator("lrl.save_seed")
        if sc.lrl_loaded_sku:
            col.label(text=f"Loaded: {sc.lrl_loaded_sku}", icon='MESH_DATA')
        if sc.lrl_inherited_path:
            col.label(text=f"Inherited: {sc.lrl_inherited_path.split('/')[-1]}", icon='FILE')
        if sc.lrl_dirty:
            col.label(text="● unsaved changes", icon='ERROR')


CLASSES = (LRL_PT_panel,)
