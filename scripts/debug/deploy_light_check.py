"""deploy_light_check.py — deployment-pipeline smoke check (needs GPU, ~2 min, run on the user terminal).

Runs the full scripts/deploy_light.py pipeline twice as a subprocess (sku 13 + front + n=2):
  ① --gate-tau -9 (gate disabled) → assert legacy semantics: winner = candidate with the highest z
  ② --gate-tau 9 (gate forced) → assert conservative semantics: winner = inherit lighting (index=-1), config=inherit_config

Both passes assert:
  1. result.json contract keys present (sku/view/inherit_from/n/winner/gate/candidates; winner carries config_path)
  2. winner_config passes core.load_config schema validation
  3. render count = preview 1 + n candidates + 1 inherit (png and config, n + 1 each)
  4. candidates sorted by z descending; the gate block's Δz = winner_z − inherit_z is self-consistent

Usage:
  HF_HUB_OFFLINE=1 .venv/bin/python scripts/debug/deploy_light_check.py
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "addons"))
from lighting_rl import core  # noqa: E402


def run_once(tag, gate_tau, sku, view, n):
    tmp = Path(tempfile.mkdtemp(prefix=f"deploy_check_{tag}_"))
    out = tmp / "result.json"
    cmd = [sys.executable, str(ROOT / "scripts" / "deploy_light.py"),
           "--sku", str(sku), "--view", view, "--n", str(n), "--gate-tau", str(gate_tau),
           "--out", str(out), "--renders-dir", str(tmp / "renders")]
    print(f"[{tag}] running deploy pipeline (sku {sku} / {view} / n={n} / gate-tau={gate_tau})...", flush=True)
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout[-2000:])
        print(proc.stderr[-2000:])
        raise SystemExit(f"deploy_light exited with rc={proc.returncode}")

    result = json.loads(out.read_text())
    for k in ("sku", "view", "inherit_from", "n", "preview_png", "winner", "gate", "candidates"):
        assert k in result, f"result.json missing key {k}"
    assert result["sku"] == str(sku) and result["view"] == view and result["n"] == n
    win, gate = result["winner"], result["gate"]
    for k in ("index", "score_z", "score_raw", "png", "config_path"):
        assert k in win, f"winner missing key {k}"
    for k in ("tau", "applied", "winner_z", "inherit_z", "delta_z",
              "inherit_png", "inherit_config_path"):
        assert k in gate, f"gate missing key {k}"
    core.load_config(win["config_path"])  # raises here on schema mismatch

    renders = tmp / "renders"
    pngs = sorted(p.name for p in renders.glob("cand_*.png"))
    cfgs = sorted(p.name for p in renders.glob("cand_*_config.json"))
    assert pngs == ["cand_00.png", "cand_01.png"], pngs
    assert cfgs == ["cand_00_config.json", "cand_01_config.json"], cfgs
    assert Path(result["preview_png"]).is_file()
    assert Path(gate["inherit_png"]).is_file() and Path(gate["inherit_config_path"]).is_file()
    core.load_config(gate["inherit_config_path"])

    cands = result["candidates"]
    assert len(cands) == 2
    assert cands[0]["score_z"] >= cands[1]["score_z"], "candidates not sorted by z descending"
    assert abs(gate["delta_z"] - (gate["winner_z"] - gate["inherit_z"])) < 1e-9, "gate Δz inconsistent"
    assert gate["applied"] == (gate["delta_z"] <= gate["tau"]), "gate decision inconsistent with Δz/τ"

    if gate["applied"]:
        assert win["index"] == -1, "when the gate fires, winner must point at the inherited light (index=-1)"
        assert Path(win["config_path"]) == Path(gate["inherit_config_path"])
        assert win["score_z"] == gate["inherit_z"]
    else:
        assert win["index"] == cands[0]["index"] and win["score_z"] == cands[0]["score_z"], \
            "when the gate does not fire, winner must be the highest-z candidate"
    wname = "inherited light (gated)" if gate["applied"] else f"cand_{win['index']:02d}"
    print(f"[{tag}] winner={wname} z={win['score_z']:+.3f}  Δz={gate['delta_z']:+.3f}  τ={gate['tau']}",
          flush=True)
    return result


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Deploy-pipeline smoke test: result.json contract, gate "
                                             "self-consistency, and same-args rerun reproducibility (3 pipeline runs).")
    ap.add_argument("--sku", default="52", help="product id to test with (default: a shipped sample)")
    ap.add_argument("--view", default="front", choices=["front", "high"])
    ap.add_argument("--n", type=int, default=2, help="candidate count per run (keep small: 3 runs total)")
    args = ap.parse_args()
    sku, view, n = args.sku, args.view, args.n
    for need in (ROOT / "model" / f"{sku}.blend", ROOT / "data" / "embeddings" / f"{sku}.npy"):
        if not need.is_file():
            raise SystemExit(f"missing {need.relative_to(ROOT)} — unpack samples.zip into model/ and "
                             f"data/embeddings/ first (or pass --sku of a product you have)")
    r_on = run_once("gate-off", -9, sku, view, n)    # τ=-9: Δz always > τ → always keep the correction
    assert r_on["gate"]["applied"] is False
    r_on2 = run_once("gate-off-rerun", -9, sku, view, n)  # reproducibility: second run with identical params
    # Tolerance: cross-process float noise after a system change — PNG pixels ±1/255 (identical seed/OPTIX/config)
    # amplify to RM z ≈ ±0.005 and candidate config ≈ ±2e-3; bitwise equality is unattainable. Semantic repro = same winner, z/config within tolerance; τ-boundary flips within ±0.005z are inherent noise.
    z1 = [c["score_z"] for c in r_on["candidates"]] + [r_on["gate"]["inherit_z"]]
    z2 = [c["score_z"] for c in r_on2["candidates"]] + [r_on2["gate"]["inherit_z"]]
    assert all(abs(a - b) <= 0.05 for a, b in zip(z1, z2)), \
        f"render reproducibility beyond tolerance ±0.05z (if this fires: check Cycles seed pinning / device fallback chain): {z1} vs {z2}"
    assert r_on["winner"]["index"] == r_on2["winner"]["index"], \
        f"two runs with identical args disagree on winner: {r_on['winner']['index']} vs {r_on2['winner']['index']}"

    # Config comparison in tanh space (72-dim policy-native representation): physical axes have mixed units —
    # on the energy log axis, 0.25W@480W is only a 5e-4 relative jitter, so absolute tolerance false-alarms; tanh-space noise is uniform (measured max ≈4e-4).
    cfg1 = json.loads(Path(r_on["winner"]["config_path"]).read_text())
    cfg2 = json.loads(Path(r_on2["winner"]["config_path"]).read_text())
    dmax = max(abs(float(d)) for d in (core.to_tanh(cfg1) - core.to_tanh(cfg2)))
    assert dmax <= 2e-3, \
        f"two runs with identical args exceed config tolerance (tanh max|Δ|={dmax:.2e} > 2e-3; preview embedding drift?)"
    dzmax = max(abs(a - b) for a, b in zip(z1, z2))
    print(f"[repro] two identical runs: winner matches + z max|Δ|={dzmax:.4f} + config max|Δ|={dmax:.2e} PASS", flush=True)
    r_gate = run_once("gate-forced", 9, sku, view, n)  # τ=9: Δz always ≤ τ → always keep the inherit lighting
    assert r_gate["gate"]["applied"] is True
    print(f"inherited from {r_gate['inherit_from']}; DEPLOY LIGHT CHECK: PASS")


if __name__ == "__main__":
    main()
