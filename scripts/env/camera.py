"""Standard dual views (view spec: deliverable = 2 images per product —
front 0° + upper-right 45° high).

Both poses live in the top-level constants CAM_FRONT / CAM_HIGH; render code sets the camera
actively on every render for deterministic reproduction. Change a pose = change the constant.
Axis convention: camera sits at -Y looking toward +Y → the product front (label side) faces -Y.

Jitter: apply_cam(pose, jitter_deg) orbits ±10° around origin Z on top of the pose (rollout
use; recording/backfill passes omit it = 0). Product Z ±10° jitter is NOT handled here —
that belongs to the import/view logic.

Usage:
  blender -b scene/stage.blend --python scripts/env/camera.py -- [front|high] [jitter_deg]
Other scripts: importlib-load, then call apply_cam("front" | "high", jitter_deg=0.0)
"""

import sys
from math import radians

import bpy
from mathutils import Matrix

CAM_FRONT = dict(loc=(0, -10.0, 0), rot=(radians(90), 0, 0), lens=280)  # pose captured from the viewport
CAM_HIGH = dict(loc=(8.3, -8.3, 3.15), rot=(radians(75), 0, radians(45)), lens=280)  # pose captured from the viewport

POSES = {"front": CAM_FRONT, "high": CAM_HIGH}


def apply_cam(pose_name, jitter_deg=0.0):
    """Move the camera to a standard pose. jitter_deg = orbit angle around origin Z (±10°
    jitter): position rotates about Z, yaw gets the same angle, aim at origin unchanged
    (exact, since both poses have rot.y == 0 under euler XYZ)."""
    pose = POSES[pose_name]
    camera = bpy.data.objects['Camera']
    camera.location = pose["loc"]
    camera.rotation_euler = pose["rot"]
    camera.data.lens = pose["lens"]
    if jitter_deg:
        camera.location = Matrix.Rotation(radians(jitter_deg), 4, 'Z') @ camera.location
        camera.rotation_euler.z += radians(jitter_deg)


if __name__ == "__main__":
    # chained with import_product.py etc. sharing post-`--` args: pose name must be a POSES key; a bare number is taken as jitter
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    pose_name = next((a for a in argv if a in POSES), "front")
    jitter = next((float(a) for a in argv if a.lstrip("+-.").replace(".", "", 1).isdigit()), 0.0)
    apply_cam(pose_name, jitter)
    cam = bpy.data.objects['Camera']
    print(f"view {pose_name} jitter={jitter}°: loc={tuple(round(v, 3) for v in cam.location)} "
          f"lens={cam.data.lens}")
