"""lighting_rl/core.py — 72-dim lighting config core.

Pure logic, no UI deps: shared by the addon shell (ops/panel/hotkey) and headless scripts.
No intra-package relative imports; can run directly as a self-test:
  blender -b scene/stage.blend --python addons/lighting_rl/core.py        # incl. scene round-trip
  .venv/bin/python addons/lighting_rl/core.py                             # pure-numeric round-trip only

Schema (8 slots x 9 dims = 72; slot order = light_01..light_08, fixed intra-slot order):
  0-2 pos x/y/z  tanh → linear ±POS_RANGE m (relative to product center; POS_RANGE=5.0 —
                 stage-measured hard lights reach 4.4m; the old ±2m was a leftover from the
                 5-25cm-scale era and was relaxed proportionally after normalization)
  3-4 aim u/v    tanh → ±1 (= ±AIM_RANGE·R offset on the per-light tangent plane; R = product
                 bbox bounding-sphere radius. Plane: through product center, normal
                 a = normalize(center − light_pos), e1 = normalize(ẑ×a), e2 = a×e1, with
                 x̂ fallback when a is near-vertical. Window widened 2R → 3.5R because rim-light
                 setups hit raw offsets up to 3.3R and got clamped; legacy data was migrated with aim ×2/3.5)
  5   roll       tanh → ±90° (about the local light axis; softbox orientation period is 180°,
                 so ±90° has no wrap and no gap)
  6   energy     tanh → x=(t+1)/2 → 10^(4x−1), 0.1-1000W log scale (0.1W = soft off)
  7-8 size       tanh → linear 0.01-12, = object scale.x/scale.y (viewport S-key value, read/written
                 directly on export/apply; rig convention keeps data.size = 1, so the number is the
                 effective size in m). scale.z is not stored and not touched — the AREA emitter face
                 is the local XY plane, so z scaling is optically meaningless.

JSON on disk = physical quantities (m / W / deg / aim dimensionless); the network side enters tanh
space via to_tanh/from_tanh:
  single config = {"schema": "lighting_rl.v1",
               "slots": {"light_01": {"pos": [x,y,z], "aim": [u,v], "roll": deg,
                                      "energy": W, "size": [sx,sy]}, ...}}
  seed seeds/<sku>.json = {"front": <config>, "high": <config>};
  snapshots data/snapshots/<sku>_<view>_<kind>.json wrap an extra {"sku","view","kind","config"} layer.

Env-side clamping (in apply, before render): light pos clamped outside the product bounding sphere
(+2cm) and z ≥ -1.2 (suspended mode has no ground geometry; the "no underground" bound is set from the
stage-measured bottom light light_05 z=-0.94 with margin); aim u/v clamped to ±1. Hand-tuned seeds
satisfy these naturally; clamping only bites RL exploration.
Anti-clip: center-point clamping cannot guard the emitter rectangle itself (size up to 12m) — when the
emitting plane slices the product bounding sphere at a grazing angle, it is pushed radially outward in
2cm steps until 2cm clearance (_avoid_product_intersection; aim target unchanged; seeds are pushed at
most 14cm, visually imperceptible). Extreme configs that still fail after 5m of push-out (huge emitter
grazing the sphere at a shallow angle, produced under relaxed RL sigma) no longer hard-crash: the slot
is soft-parked at POSE_OFF for a degraded render (the low RM score penalizes it naturally).
Soft-off slots (energy ≤ 0.1W + 1e-6) are canonicalized to POSE_OFF on export (high rear position
behind/above the camera, aim at center, size 0.5, roll 0) — in training data the other 8 dims of an
"off" light are constants, so BC doesn't learn noise.
"""

import json
import math
from pathlib import Path

import numpy as np

try:
    import bpy
    from mathutils import Quaternion, Vector
except ImportError:  # no bpy on the training side (plain venv python)
    bpy = None

SCHEMA = "lighting_rl.v1"
SLOTS = [f"light_{i:02d}" for i in range(1, 9)]
N_SLOTS = 8
DIM_PER_SLOT = 9
N_DIM = N_SLOTS * DIM_PER_SLOT

POS_RANGE = 5.0       # pos linear mapping, ±m
SIZE_MIN, SIZE_MAX = 0.01, 12.0  # size = object scale.x/y; stage-measured max 11.5m, with margin
ENERGY_MIN, ENERGY_MAX = 0.1, 1000.0  # log scale; ENERGY_MIN = soft off
AIM_RANGE = 3.5       # aim physical offset = u × AIM_RANGE × R (widened from 2.0 to fit rim-light setups)
OFF_EPS = 1e-6

# canonical pose for soft-off slots: high rear position behind the camera (-Y side), aim at product center, default size, roll 0
POSE_OFF = dict(pos=[0.0, -1.9, 1.8], aim=[0.0, 0.0], roll=0.0, energy=ENERGY_MIN,
                size=[0.5, 0.5])

FLOOR_Z = -1.2  # "no underground" proxy for suspended mode (stage bottom light measured at -0.94, with margin)

EMITTER_MARGIN = 0.02  # min clearance (m) between emitter rectangle and product bounding sphere, same spherical margin as clamp_pos


# ---------- pure numeric: per-dim mapping (tanh space t ∈ [-1,1] ↔ physical quantity) ----------

def _pos_to_tanh(p): return np.clip(np.asarray(p, float) / POS_RANGE, -1, 1)
def _pos_from_tanh(t): return np.asarray(t, float) * POS_RANGE
def _aim_to_tanh(uv): return np.clip(np.asarray(uv, float), -1, 1)
def _aim_from_tanh(t): return np.asarray(t, float)
def _roll_to_tanh(deg): return float(np.clip(deg / 90.0, -1, 1))
def _roll_from_tanh(t): return float(t) * 90.0

def _energy_to_tanh(w):
    x = (math.log10(max(w, ENERGY_MIN)) + 1.0) / 4.0
    return float(np.clip(2 * x - 1, -1, 1))

def _energy_from_tanh(t):
    return 10.0 ** (4.0 * ((float(t) + 1) / 2) - 1)

def _size_to_tanh(m):
    return np.clip(2 * (np.asarray(m, float) - SIZE_MIN) / (SIZE_MAX - SIZE_MIN) - 1, -1, 1)

def _size_from_tanh(t):
    return SIZE_MIN + (np.asarray(t, float) + 1) / 2 * (SIZE_MAX - SIZE_MIN)


# ---------- pure numeric: config(dict) ↔ 72-dim tanh vector ----------

def empty_config():
    return {"schema": SCHEMA,
            "slots": {n: dict(pos=[0, 0, 1], aim=[0, 0], roll=0.0,
                              energy=ENERGY_MIN, size=[0.5, 0.5]) for n in SLOTS}}


def to_tanh(config):
    """config → (72,) tanh vector. Out-of-range physical values are clipped (BC target saturation = out-of-spec data; callers beware)."""
    vec = np.zeros(N_DIM)
    for i, name in enumerate(SLOTS):
        s = config["slots"][name]
        j = i * DIM_PER_SLOT
        vec[j:j + 3] = _pos_to_tanh(s["pos"])
        vec[j + 3:j + 5] = _aim_to_tanh(s["aim"])
        vec[j + 5] = _roll_to_tanh(s["roll"])
        vec[j + 6] = _energy_to_tanh(s["energy"])
        vec[j + 7:j + 9] = _size_to_tanh(s["size"])
    return vec


def from_tanh(vec):
    """(72,) tanh vector → config. Pure decode, no scene dependency (aim stored normalized, R-independent)."""
    vec = np.clip(np.asarray(vec, float).reshape(-1), -1, 1)
    assert vec.shape[0] == N_DIM, f"expected {N_DIM} dims, got {vec.shape[0]}"
    cfg = {"schema": SCHEMA, "slots": {}}
    for i, name in enumerate(SLOTS):
        j = i * DIM_PER_SLOT
        cfg["slots"][name] = dict(
            pos=_pos_from_tanh(vec[j:j + 3]).tolist(),
            aim=_aim_from_tanh(vec[j + 3:j + 5]).tolist(),
            roll=_roll_from_tanh(vec[j + 5]),
            energy=_energy_from_tanh(vec[j + 6]),
            size=_size_from_tanh(vec[j + 7:j + 9]).tolist(),
        )
    return cfg


def canonical_off_slots(config):
    """Replace soft-off slots (energy ≤ 0.1W) wholesale with POSE_OFF; return a new config. Used for BC target cleanup."""
    cfg = json.loads(json.dumps(config))
    for name, s in cfg["slots"].items():
        if s["energy"] <= ENERGY_MIN + OFF_EPS:
            cfg["slots"][name] = json.loads(json.dumps(POSE_OFF))
    return cfg


# ---------- JSON I/O ----------

def save_config(config, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    clean = {k: v for k, v in config.items() if not k.startswith("_")}  # private keys (e.g. _aim_overflow) are never written to disk
    Path(path).write_text(json.dumps(clean, ensure_ascii=False, indent=1))


def load_config(path):
    cfg = json.loads(Path(path).read_text())
    assert cfg.get("schema") == SCHEMA, f"{path} schema mismatch: {cfg.get('schema')}"
    assert set(cfg["slots"]) == set(SLOTS), f"{path} has missing slots"
    return cfg


def save_seed(path, front=None, high=None):
    """Seed = {front, high} configs; when only one side is passed, the other side already in the file is kept."""
    path = Path(path)
    seed = {}
    if path.exists():
        seed = json.loads(path.read_text())
    if front is not None:
        seed["front"] = {k: v for k, v in front.items() if not k.startswith("_")}  # private keys not written to disk
    if high is not None:
        seed["high"] = {k: v for k, v in high.items() if not k.startswith("_")}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(seed, ensure_ascii=False, indent=1))


def load_seed(path):
    seed = json.loads(Path(path).read_text())
    out = {}
    for view in ("front", "high"):
        if view in seed:
            cfg = seed[view]
            assert cfg.get("schema") == SCHEMA and set(cfg["slots"]) == set(SLOTS)
            out[view] = cfg
    return out


# ---------- bpy side: scene apply / export ----------

def bpy_available():
    """True only when actually running inside Blender (if the venv has the standalone bpy package, context.scene is None)."""
    return bpy is not None and bpy.context is not None and bpy.context.scene is not None


def _require_bpy():
    if not bpy_available():
        raise RuntimeError("apply/export only runs inside Blender (use to_tanh/from_tanh on the training side)")


def get_product_sphere():
    """Product bbox center and bounding-sphere radius R (world coords; normalized products should be ≈ origin / R ≤ 0.87)."""
    _require_bpy()
    bpy.context.view_layer.update()  # matrix_world of a freshly linked object is stale
    coll = bpy.context.scene.collection.children.get('product')
    meshes = [ob for ob in coll.objects if ob.type == 'MESH'] if coll else []
    if not meshes:
        raise RuntimeError("product collection is empty; cannot determine product centre/bounding sphere")
    pts = [ob.matrix_world @ Vector(c) for ob in meshes for c in ob.bound_box]
    lo = Vector([min(p[i] for p in pts) for i in range(3)])
    hi = Vector([max(p[i] for p in pts) for i in range(3)])
    return (lo + hi) / 2.0, (hi - lo).length / 2.0, lo.z


def clamp_pos(pos, center, R, floor_z):
    """Env-side clamping: outside the bounding sphere (+2cm), no underground (z ≥ max(FLOOR_Z, product bottom - 5cm))."""
    p = Vector(pos)
    v = p - center
    r_min = R + 0.02
    if v.length < r_min:
        direction = v.normalized() if v.length > 1e-6 else Vector((0, -1, 0.3)).normalized()
        p = center + direction * r_min
    p.z = max(p.z, min(FLOOR_Z, floor_z - 0.05))  # take the lower: with normalized product bottom -0.5 the floor is -1.2, so existing bottom lights are unaffected
    return p


def _aim_basis(center, light_pos):
    """Per-light tangent-plane basis."""
    a = center - light_pos
    if a.length < 1e-6:
        raise RuntimeError("light position coincides with product centre; cannot define the tangent plane")
    a = a.normalized()
    up = Vector((0, 0, 1))
    if abs(a.dot(up)) > 0.999:
        up = Vector((1, 0, 0))
    e1 = up.cross(a).normalized()
    e2 = a.cross(e1)
    return a, e1, e2


def _pose_quaternion(light_pos, center, R, u, v, roll_deg):
    """Forward compose: aim point = center + AIM_RANGE·R(u·e1 + v·e2), to_track_quat('-Z','Y') right-multiplied by roll."""
    _, e1, e2 = _aim_basis(center, light_pos)
    target = center + AIM_RANGE * R * (u * e1 + v * e2)
    return (target - light_pos).to_track_quat('-Z', 'Y') @ Quaternion((0, 0, 1), math.radians(roll_deg))


def _emitter_clearance(L, q, hx, hy, center, R):
    """Clearance between the emitter rectangle (center L, pose q, half extents hx/hy) and the product
    bounding sphere (center, R) (<0 = intersecting). The emitter is a planar rectangle, not a volume:
    dist² = squared normal component from sphere center to the plane + squared in-plane components
    beyond the rect edges. A large softbox facing the product up close (studio norm) is not a false
    positive — when the plane is parallel to the product, the normal component ≈ d > R."""
    n = (q @ Vector((0, 0, -1))).normalized()  # emitter face normal (toward the aim point)
    e1, e2 = q @ Vector((1, 0, 0)), q @ Vector((0, 1, 0))
    w = Vector(center) - L
    dn = w.dot(n)
    wp = w - dn * n
    da = max(0.0, abs(wp.dot(e1)) - hx)
    db = max(0.0, abs(wp.dot(e2)) - hy)
    return math.sqrt(dn * dn + da * da + db * db) - R


def _avoid_product_intersection(L, center, R, floor_z, u, v, roll_deg, hx, hy):
    """When the emitter rectangle cuts into the product bounding sphere, push it radially (center→L)
    in 2cm steps until clearance reaches EMITTER_MARGIN. Radial push leaves _aim_basis unchanged
    (a = -radial is invariant), so the aim target's world position is kept; only distance grows.
    The floor is re-clamped every step (radial-down push may lower z); convergence: as d grows, the
    normal component ≈ d monotonically exceeds R."""
    for _ in range(250):  # cap: 5m of push-out; normally a few cm suffice
        q = _pose_quaternion(L, center, R, u, v, roll_deg)
        if _emitter_clearance(L, q, hx, hy, center, R) >= EMITTER_MARGIN:
            return L, q
        radial = L - center
        radial = radial.normalized() if radial.length > 1e-6 else Vector((0, -1, 0.3)).normalized()
        L = L + radial * 0.02
        L.z = max(L.z, min(FLOOR_Z, floor_z - 0.05))  # same rule as clamp_pos
    raise RuntimeError(f"anti-clip push-out did not clear after 5m (R={R:.2f}), bad config: pos={tuple(L)}")


def _decompose_pose(light, center, R):
    """Inverse decomposition: intersect the light axis with the tangent plane to recover (u,v); swing-twist decomposition recovers roll (period-normalized to ±90°)."""
    mw = light.matrix_world
    L = mw.translation
    q = mw.to_quaternion()
    d = (q @ Vector((0, 0, -1))).normalized()
    a, e1, e2 = _aim_basis(center, L)
    denom = d.dot(a)
    if abs(denom) < 1e-6:
        raise RuntimeError(f"{light.name}: light axis is parallel to the tangent plane, aim cannot be decomposed")
    s = (center - L).dot(a) / denom
    if s < 0:
        raise RuntimeError(f"{light.name} points away from the product (hit distance s={s:.2f}<0), aim cannot be decomposed")
    off = (L + s * d) - center
    u_raw = off.dot(e1) / (AIM_RANGE * R)
    v_raw = off.dot(e2) / (AIM_RANGE * R)
    # overflow warning threshold 1.02 (= 2% over): report only once writeback loss reaches ~0.3-0.6 deg;
    # edge-hugging/slightly-over cases (e.g. template light_01 raw u≈1.00x, loss <0.1 deg) stay silent to avoid alert fatigue
    overflowed = abs(u_raw) > 1.02 or abs(v_raw) > 1.02
    u = float(np.clip(u_raw, -1, 1))
    v = float(np.clip(v_raw, -1, 1))
    q_swing = d.to_track_quat('-Z', 'Y')
    q_twist = q_swing.inverted() @ q
    roll = 2.0 * math.atan2(q_twist.z, q_twist.w)
    while roll > math.pi / 2:   # softbox orientation period 180° → [-90°, 90°]
        roll -= math.pi
    while roll < -math.pi / 2:
        roll += math.pi
    return list(L), [u, v], math.degrees(roll), overflowed


def apply(config):
    """Apply config to scene light_01-08 (env-side clamping included; node tree/color and other fixed rig properties untouched)."""
    _require_bpy()
    center, R, bottom_z = get_product_sphere()
    for name in SLOTS:
        s = config["slots"].get(name)
        if s is None:
            raise KeyError(f"config missing slot {name}")
        ob = bpy.data.objects.get(name)
        if ob is None:
            raise KeyError(f"scene missing light {name} (duplicate .001 name? run validate_scene.py)")
        L = clamp_pos(s["pos"], center, R, bottom_z)
        u, v = (float(x) for x in _aim_to_tanh(s["aim"]))  # clamp to ±1; np scalar × Vector yields ndarray, convert to float
        # size written directly to object scale.x/scale.y (viewport S-key value); data.size/shape and scale.z untouched
        sx = float(np.clip(s["size"][0], SIZE_MIN, SIZE_MAX))
        sy = float(np.clip(s["size"][1], SIZE_MIN, SIZE_MAX))
        # anti-clip: clamp_pos only guards the lamp center; the emitter rect (up to 12m) can still
        # graze into the product — RL outputs hit this in practice; conservative seeds get ≤14cm push, visually imperceptible
        energy = min(max(s["energy"], ENERGY_MIN), ENERGY_MAX)
        try:
            L, q = _avoid_product_intersection(L, center, R, bottom_z, u, v, float(s["roll"]),
                                               sx / 2, sy / 2)
        except RuntimeError:
    # Push-out still stuck after 5m (extreme config): soft-park at POSE_OFF and render degraded —
    # the image still renders, the RM scores it low, and the policy learns to avoid it.
            L = Vector(POSE_OFF["pos"])
            q = _pose_quaternion(L, center, R, 0.0, 0.0, 0.0)
            sx, sy = float(POSE_OFF["size"][0]), float(POSE_OFF["size"][1])
            energy = ENERGY_MIN
            print(f"WARNING: {name} anti-clip push-out did not clear after 5m; rendering in the soft-off parked pose", flush=True)
        ob.location = L
        ob.rotation_mode = 'QUATERNION'
        ob.rotation_quaternion = q
        ob.data.energy = energy
        ob.scale.x = sx
        ob.scale.y = sy


def export(canonicalize=True):
    """Export config from scene light_01-08. canonicalize=True parks soft-off slots at POSE_OFF.

    Slots with genuinely overflowing aim (raw u/v beyond ±1 got clamped, costing several degrees of
    pose on writeback) make cfg carry the private key "_aim_overflow": [slot names...] — ops should
    pop it after reading and never save it (save_seed/save_config already filter underscore keys as
    a backstop). Soft-off slots are not counted here (their pose is canonicalized anyway)."""
    _require_bpy()
    center, R, _ = get_product_sphere()
    cfg = {"schema": SCHEMA, "slots": {}}
    overflow = []
    for name in SLOTS:
        ob = bpy.data.objects.get(name)
        if ob is None:
            raise KeyError(f"scene missing light {name} (duplicate .001 name? run validate_scene.py)")
        pos, aim, roll, overflowed = _decompose_pose(ob, center, R)
        if overflowed and ob.data.energy > ENERGY_MIN + OFF_EPS:
            overflow.append(name)
        # size read directly from object scale.x/scale.y (rig convention data.size = 1, so the value is the effective size in m)
        cfg["slots"][name] = dict(pos=pos, aim=aim, roll=roll,
                                  energy=float(ob.data.energy),
                                  size=[float(ob.scale.x), float(ob.scale.y)])
    if overflow:
        cfg["_aim_overflow"] = overflow
    return canonical_off_slots(cfg) if canonicalize else cfg


# ---------- self-test ----------

def _flat(cfg):
    return [(n, tuple(round(x, 9) for x in s["pos"]),
             tuple(round(x, 9) for x in s["aim"]), round(s["roll"], 9),
             round(s["energy"], 9), tuple(round(x, 9) for x in s["size"]))
            for n, s in cfg["slots"].items()]


def selftest():
    """Round-trip asserts: zero-error pure-numeric tanh space; inside Blender also runs scene export→apply→export."""
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(200):
        v = rng.uniform(-1, 1, N_DIM)
        err = float(np.max(np.abs(to_tanh(from_tanh(v)) - v)))
        worst = max(worst, err)
    assert worst < 1e-12, f"tanh round-trip error out of bounds: {worst}"
    print(f"tanh round-trip: max error over 200 random configs {worst:.1e} (float64 precision) ✓")

    off = canonical_off_slots(from_tanh(rng.uniform(-1, 1, N_DIM)))
    for name, s in off["slots"].items():
        if s["energy"] <= ENERGY_MIN + OFF_EPS:
            assert s == {**POSE_OFF, "size": list(POSE_OFF["size"])} or \
                   s["pos"] == POSE_OFF["pos"] and s["aim"] == POSE_OFF["aim"], \
                f"{name} soft-off pose not canonical"
    print("soft-off canonicalisation ✓")

    coll = bpy.context.scene.collection.children.get('product') if bpy_available() else None
    if coll and any(ob.type == 'MESH' for ob in coll.objects):
        cfg1 = export()
        apply(cfg1)
        bpy.context.view_layer.update()
        cfg2 = export()
        # per-slot quantized residual (pos m / aim dimensionless / roll deg / energy W / size m)
        dmax = 0.0
        for n in SLOTS:
            a, b = cfg1["slots"][n], cfg2["slots"][n]
            d = max(max(abs(x - y) for x, y in zip(a["pos"], b["pos"])),
                    max(abs(x - y) for x, y in zip(a["aim"], b["aim"])),
                    abs(a["roll"] - b["roll"]), abs(a["energy"] - b["energy"]),
                    max(abs(x - y) for x, y in zip(a["size"], b["size"])))
            if d > 1e-6:
                print(f"  {n} residual {d:.2e}: pos {np.array(a['pos']) - np.array(b['pos'])}, "
                      f"aim {np.array(a['aim']) - np.array(b['aim'])}, "
                      f"roll {a['roll'] - b['roll']:+.2e}, E {a['energy'] - b['energy']:+.2e}")
            dmax = max(dmax, d)
        assert dmax < 1e-4, f"scene export->apply->export residual out of bounds (max {dmax:.2e})"
        print(f"scene round-trip: export->apply->export max residual {dmax:.2e} (quaternion float noise level ✓)")
    print("CORE SELFTEST: PASS")


if __name__ == "__main__":
    selftest()
