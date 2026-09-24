"""make_blind200.py — acceptance gate: sample a 200-pair blind-labeling set + generate the keyboard labeling page.

Stratified sampling of 200 pairs from the "hard" pairs in pairs.jsonl (2 pairs per (sku, view) as guaranteed coverage first, rest filled randomly),
writes a blind-labeling directory data/pairs/blind200/:
  blind200.jsonl  {"blind_id","pair_id","sku","view","a","b"} (no winner — blind)
  manifest.json   image list for the labeling page (URLs relative to project root)
  index.html      keyboard labeling page (A/← = left, B/→ = right, S/↓ = tie, Backspace = undo;
                  "save to server" button + autosave (800ms debounce after an answer, save on tab switch) —
                  scripts/label_server.py POST /save writes the answers file directly into this directory;
                  "export JSON" kept as fallback for when the server is not running)

Usage: .venv/bin/python scripts/train/make_blind200.py [--n 200] [--seed 202]
Serve: from project root run `.venv/bin/python scripts/label_server.py` (static + answers to disk),
      open http://localhost:8000/data/pairs/blind200/ in a browser
Score: score_blind_eval.py reconciles the answers JSON against rm.pt (pairwise agreement >=70% passes the gate).
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data/pairs/blind200"


INDEX_HTML = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>Blind labeling (200 pairs)</title>
<style>
  body { font-family: sans-serif; background: #222; color: #eee; text-align: center; margin: 0; }
  #bar { height: 6px; background: #4caf50; width: 0; transition: width .15s; }
  #info { margin: 8px; font-size: 14px; color: #aaa; }
  #row { display: flex; justify-content: center; gap: 12px; margin: 12px; }
  .cell { background: #fff; padding: 6px; border-radius: 4px; border: 3px solid transparent; }
  .cell.sel { border-color: #4caf50; }
  .cell img { display: block; width: 384px; height: 384px; }
  .tag { font-size: 18px; font-weight: bold; padding: 4px; }
  #keys { color: #888; font-size: 13px; margin: 8px; }
  button { font-size: 16px; padding: 8px 16px; margin: 4px; cursor: pointer; }
</style>
</head>
<body>
<div id="bar"></div>
<div id="info">Loading…</div>
<div id="row">
  <div class="cell" id="ca"><div class="tag">A (left)</div><img id="ia"></div>
  <div class="cell" id="cb"><div class="tag">B (right)</div><img id="ib"></div>
</div>
<div id="keys">A or ← = left is better | B or → = right is better | S or ↓ = tie | Backspace = undo last pair</div>
<button id="save">Save to server</button>
<button id="export">Export JSON (fallback)</button>
<span id="savestate" style="color:#888; font-size:13px"></span>
<span id="done" style="color:#4caf50"></span>
<script>
let pairs = [], idx = 0, answers = {}, dirty = false, saveTimer = null;
const LS_KEY = "blind200_ans_" + location.pathname;  // isolated per page path; v1/v2 do not pollute each other
const ANSWERS_NAME = "blind200_answers.json";  // answers filename for this batch (server save + export fallback)
const SAVE_DIR = location.pathname.replace(/[^/]*$/, "");  // this page's directory = batch directory
try { answers = JSON.parse(localStorage.getItem(LS_KEY) || "{}"); } catch (e) {}
function setState(t, c) { const el = document.getElementById("savestate"); el.textContent = t; el.style.color = c || "#888"; }
fetch("manifest.json").then(r => r.json()).then(async m => {
  pairs = m;
  let srv = {};
  try {  // answers already on the server (empty object if none)
    const r = await fetch(ANSWERS_NAME, { cache: "no-store" });
    if (r.ok) srv = await r.json();
  } catch (e) {}
  const n_loc = Object.keys(answers).length, n_srv = Object.keys(srv).length;
  if (n_loc === 0 && n_srv > 0) {          // local empty: restore from server
    answers = srv; localStorage.setItem(LS_KEY, JSON.stringify(answers));
  } else if (n_loc > n_srv) {              // local has more: autosave on open (guards against "labeled but never saved")
    setState("local " + n_loc + " > server " + n_srv + ", autosaving…");
    dirty = true; saveServer();
  }
  while (idx < pairs.length && answers[pairs[idx].blind_id]) idx++;
  show();
});
function scheduleSave() {
  dirty = true; setState("unsaved…", "#ffb74d");
  clearTimeout(saveTimer); saveTimer = setTimeout(saveServer, 800);
}
async function saveServer() {
  clearTimeout(saveTimer);
  if (!Object.keys(answers).length) { setState("nothing to save yet"); return; }
  try {
    const r = await fetch("/save", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ dir: SAVE_DIR, filename: ANSWERS_NAME, answers }) });
    const j = await r.json();
    if (!j.ok) throw new Error(j.error || ("HTTP " + r.status));
    dirty = false;
    setState("saved " + j.n + " entries " + j.mtime.slice(11), "#4caf50");
  } catch (e) {
    setState("save failed: " + e.message + " (server down? use Export JSON as fallback)", "#e57373");
  }
}
document.addEventListener("visibilitychange", () => { if (document.hidden && dirty) saveServer(); });
function show() {
  document.getElementById("bar").style.width = (100 * Object.keys(answers).length / pairs.length) + "%";
  if (idx >= pairs.length) {
    document.getElementById("info").textContent = "All labeled! (autosaving)";
    document.getElementById("ia").removeAttribute("src");
    document.getElementById("ib").removeAttribute("src");
    document.getElementById("done").textContent = " labeled " + Object.keys(answers).length + "/" + pairs.length;
    return;
  }
  const p = pairs[idx];
  document.getElementById("info").textContent =
    "pair " + (idx + 1) + "/" + pairs.length + " (labeled " + Object.keys(answers).length + ")";
  document.getElementById("ia").src = "/" + p.a;
  document.getElementById("ib").src = "/" + p.b;
  document.getElementById("ca").className = "cell";
  document.getElementById("cb").className = "cell";
}
function answer(w) {
  if (idx >= pairs.length) return;
  answers[pairs[idx].blind_id] = w;
  localStorage.setItem(LS_KEY, JSON.stringify(answers));
  scheduleSave();
  const cell = document.getElementById(w === "a" ? "ca" : "cb");
  if (w !== "tie") cell.className = "cell sel";
  setTimeout(() => { idx++; show(); }, 120);
}
document.addEventListener("keydown", e => {
  if (e.key === "a" || e.key === "A" || e.key === "ArrowLeft") answer("a");
  else if (e.key === "b" || e.key === "B" || e.key === "ArrowRight") answer("b");
  else if (e.key === "s" || e.key === "S" || e.key === "ArrowDown") answer("tie");
  else if (e.key === "Backspace" && idx > 0) { idx--; delete answers[pairs[idx].blind_id]; localStorage.setItem(LS_KEY, JSON.stringify(answers)); scheduleSave(); show(); e.preventDefault(); }
});
document.getElementById("save").onclick = saveServer;
document.getElementById("export").onclick = () => {
  const blob = new Blob([JSON.stringify(answers, null, 1)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = ANSWERS_NAME;
  a.click();
};
</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=202)
    ap.add_argument("--pairs-file", type=Path, default=ROOT / "data/pairs/pairs.jsonl",
                    help="pair source (default main pairs.jsonl; ppo flywheel batch: data/pairs/ppo_v1/pairs.jsonl)")
    ap.add_argument("--kinds", nargs="+", default=["hard"],
                    help="kinds eligible for sampling (default hard; ppo batch: policy_vs_human policy_vs_step1 policy_vs_perturb human_vs_perturb)")
    ap.add_argument("--answers-name", default="blind200_answers.json",
                    help="answers filename for the page export button (per-batch, prevents overwriting)")
    ap.add_argument("--exclude-pairs", type=Path, nargs="+", default=None,
                    help="exclusion files (jsonl with pair_id field, can pass several): their pair_ids are excluded from sampling")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR, help="output dir (default data/pairs/blind200)")
    args = ap.parse_args()
    out_dir = args.out_dir

    exclude = set()
    for f in args.exclude_pairs or []:
        ids = {json.loads(l)["pair_id"] for l in f.read_text().splitlines() if l.strip()}
        exclude |= ids
        print(f"excluding {len(ids)} pair_ids ({f})")
    if exclude:
        print(f"cumulative exclusions {len(exclude)}")

    pairs = [json.loads(l) for l in args.pairs_file.read_text().splitlines() if l.strip()]
    hard = [p for p in pairs if p["kind"] in args.kinds and p["pair_id"] not in exclude]
    by_view = defaultdict(list)
    for p in hard:
        by_view[(p["sku"], p["view"])].append(p)

    rng = np.random.default_rng(args.seed)
    picked, rest = [], []
    for key in sorted(by_view):
        rows = by_view[key]
        take = rng.choice(len(rows), size=min(2, len(rows)), replace=False)
        for i, r in enumerate(rows):
            (picked if i in take else rest).append(r)
    n = min(args.n, len(hard))
    if n < args.n:
        print(f"candidates {len(hard)} pairs, fewer than --n {args.n}, sampling {n}")
    if n < len(picked):  # when n is below the guaranteed coverage count, randomly keep n of them
        idx = rng.choice(len(picked), size=n, replace=False)
        picked = [picked[i] for i in idx]
    need = n - len(picked)
    assert need >= 0 and need <= len(rest), f"sample size out of range: need {need}, have {len(rest)}"
    extra = rng.choice(len(rest), size=need, replace=False)
    picked += [rest[i] for i in extra]
    rng.shuffle(picked)

    out_dir.mkdir(parents=True, exist_ok=True)
    rows_out, manifest = [], []
    for i, p in enumerate(picked):
        rows_out.append({"blind_id": i, "pair_id": p["pair_id"], "sku": p["sku"], "view": p["view"],
                         "a": p["a"], "b": p["b"]})
        manifest.append({"blind_id": i, "a": p["a"], "b": p["b"]})
    (out_dir / "blind200.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows_out) + "\n")
    (out_dir / "manifest.json").write_text(json.dumps(manifest))
    html = INDEX_HTML.replace("blind200_answers.json", args.answers_name).replace(
        "Blind labeling (200 pairs)", f"Blind labeling ({len(rows_out)} pairs)")
    (out_dir / "index.html").write_text(html)
    n_views = len({(r["sku"], r["view"]) for r in rows_out})
    print(f"blind set {len(rows_out)} pairs (covering {n_views} views) -> {out_dir}")
    try:
        rel = out_dir.resolve().relative_to(ROOT)
        print(f"serve: run `python3 -m http.server 8000` at the project root, open http://localhost:8000/{rel}/")
    except ValueError:
        print(f"output dir is outside the project ({out_dir}); http.server cannot reach it, sampling check only")


if __name__ == "__main__":
    main()
