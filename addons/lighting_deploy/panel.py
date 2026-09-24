"""panel.py — 3D-viewport N panel (buttons and status only; all logic lives in ops.py)."""

import bpy
from bpy.props import EnumProperty, IntProperty, StringProperty


def register_props():
    sc = bpy.types.Scene
    sc.lrd_sku = StringProperty(name="Product id", description="id of model/<id>.blend, e.g. 54")
    sc.lrd_view = EnumProperty(
        name="View",
        items=[("front", "Front 0°", ""), ("high", "High 45° right", "")],
        default="front")
    sc.lrd_n = IntProperty(name="Candidates N", description="step-2 sample count (deploy = best-of-16)",
                           default=16, min=1, max=64)
    sc.lrd_status = StringProperty(default="")


def unregister_props():
    sc = bpy.types.Scene
    for p in ("lrd_sku", "lrd_view", "lrd_n", "lrd_status"):
        if hasattr(sc, p):
            delattr(sc, p)


class LRD_PT_panel(bpy.types.Panel):
    bl_label = "Lighting Deploy"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Lighting Deploy"

    def draw(self, context):
        sc = context.scene
        col = self.layout.column(align=True)
        col.prop(sc, "lrd_sku")
        col.prop(sc, "lrd_view")
        col.prop(sc, "lrd_n")
        running = bool(sc.lrd_status.startswith("Running"))
        if bpy.data.objects.get("light_01") is None:
            col.label(text="Stage not open: the button will open stage.blend, then press it again", icon='INFO')
        row = col.row()
        row.enabled = not running
        row.operator("lrd.auto_light", text=f"Auto Light ({sc.lrd_view}, best-of-{sc.lrd_n})",
                     icon='LIGHT_AREA')
        if sc.lrd_status:
            icon = 'TIME' if running else 'CHECKMARK'
            col.label(text=sc.lrd_status, icon=icon)


CLASSES = (LRD_PT_panel,)
