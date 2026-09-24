"""label_server.py — local server for blind-labeling pages: static files + answers written straight to disk.

Replaces `python3 -m http.server 8000`: adds POST /save on top of static serving — the labeling page's
"Save to server" button / autosave writes the answers JSON directly back into the batch directory
(previously a manual download-and-copy every batch).

Rules:
  - listens on 127.0.0.1 only; writes allowed only under data/pairs/ inside the project root, filename reduced to basename (blocks path traversal)
  - keeps the previous version as <stem>.prev.json before overwriting (mis-saves stay rollback-able)
  - on startup, lists labeling-page URLs under data/pairs/*/index.html with saved-answer counts

Usage:
  .venv/bin/python scripts/label_server.py [--port 8000]
  (if 8000 is held by an old http.server: `pkill -f "http.server 8000"` first)
"""

import argparse
import json
import re
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parents[1]
PAIRS = ROOT / "data" / "pairs"


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def log_message(self, fmt, *args):  # mute per-request logging; save events stay
        pass

    def _json(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if urlparse(self.path).path.rstrip("/") != "/save":
            self._json(404, {"ok": False, "error": "unknown endpoint"})
            return
        try:
            n = int(self.headers.get("Content-Length") or 0)
            payload = json.loads(self.rfile.read(n) or b"{}")
            answers = payload["answers"]
            assert isinstance(answers, dict) and answers, "answers are empty"
            name = Path(str(payload["filename"])).name  # basename: blocks path traversal
            assert name.endswith(".json"), f"bad filename: {name}"
            tdir = (ROOT / unquote(str(payload["dir"])).strip("/")).resolve()
            tdir.relative_to(PAIRS)  # writes allowed only under data/pairs
            assert tdir.is_dir(), f"{tdir} is not a directory"
            out = tdir / name
            if out.is_file():
                (tdir / (out.stem + ".prev.json")).write_bytes(out.read_bytes())
            out.write_text(json.dumps(answers, ensure_ascii=False, indent=1))
        except Exception as e:  # noqa: BLE001 —— any failure returns 400; the page shows the reason, with export fallback
            self._json(400, {"ok": False, "error": f"{type(e).__name__}: {e}"})
            return
        print(f"[save] {out.relative_to(ROOT)}  {len(answers)} entries  {time.strftime('%H:%M:%S')}",
              flush=True)
        self._json(200, {"ok": True, "file": str(out.relative_to(ROOT)), "n": len(answers),
                         "mtime": time.strftime("%Y-%m-%d %H:%M:%S")})


def list_pages(host, port):
    for p in sorted(PAIRS.glob("*/index.html")):
        title = re.search(r"<title>([^<]*)</title>", p.read_text())
        m = re.search(r'const ANSWERS_NAME = "([^"]+)"', p.read_text())
        note = ""
        if m:
            ans = p.parent / m.group(1)
            note = f"  [{sum(1 for l in ans.read_text().splitlines() if l.strip())} saved]" \
                if ans.is_file() else "  [not labeled]"
        print(f"  http://{host}:{port}/{p.parent.relative_to(ROOT)}/   "
              f"{title.group(1) if title else p.parent.name}{note}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    try:
        srv = ThreadingHTTPServer((args.host, args.port), Handler)
    except OSError as e:
        raise SystemExit(f"port {args.port} unavailable ({e}) — an old server still holding it? "
                         f"kill it first or use another --port")
    print(f"label_server up: http://{args.host}:{args.port}/  (static root {ROOT}, answers written straight to data/pairs/)",
          flush=True)
    print("Labeling page (save button + autosave; answers land on disk, no download step):", flush=True)
    list_pages(args.host, args.port)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
