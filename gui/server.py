"""2048-konsey GUI gateway: serves the game UI and streams a model's live game over SSE.

    python3 gui/server.py [--port 8048] [--models DIR ...]

Binds to 127.0.0.1 only. Models are any K2048NT1 network files found under runs/, models/
and the --models directories; the browser may only pick from that list (no arbitrary paths).
Standard library only.
"""

import argparse
import json
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent.parent
GUI = ROOT / "gui"
PLAY = ROOT / "build" / "play"
STATIC = {"/": ("index.html", "text/html; charset=utf-8"), "/engine.js": ("engine.js", "text/javascript; charset=utf-8")}


def read_header(path: Path) -> dict | None:
    """Text header of a network file, or None if it is not one."""
    try:
        with open(path, "rb") as f:
            if f.readline() != b"K2048NT1\n":
                return None
            h = {}
            for _ in range(64):
                line = f.readline().decode("utf-8", "replace").strip()
                if line == "end":
                    return h
                k, _, v = line.partition("=")
                h[k] = v
    except OSError:
        pass
    return None


def find_models(dirs: list[Path]) -> dict[str, dict]:
    models = {}
    for d in dirs:
        if not d.is_dir():
            continue
        for p in sorted(d.rglob("*.bin")):
            h = read_header(p)
            if h is None:
                continue
            mid = str(p.relative_to(d.parent)) if d.parent != p else p.name
            st = p.stat()
            models[mid] = {
                "id": mid,
                "path": str(p),
                "mb": round(st.st_size / 1e6),
                "mtime": int(st.st_mtime),
                "games": int(h.get("games", 0) or 0),
                "tuples": len(h.get("patterns", "").split(";")),
                "stages": len(h.get("stages", "").split(";")),
            }
    return models


class Handler(BaseHTTPRequestHandler):
    model_dirs: list[Path] = []

    def log_message(self, fmt, *args):
        sys.stderr.write("[gateway] " + fmt % args + "\n")

    def send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def error(self, code: int, msg: str):
        self.send(code, json.dumps({"error": msg}).encode(), "application/json")

    def do_GET(self):
        url = urlparse(self.path)
        if url.path in STATIC:
            name, ctype = STATIC[url.path]
            return self.send(200, (GUI / name).read_bytes(), ctype)
        if url.path == "/api/models":
            models = sorted(find_models(self.model_dirs).values(), key=lambda m: -m["mtime"])
            for m in models:
                m.pop("path")
            return self.send(200, json.dumps({"models": models, "play_built": PLAY.exists()}).encode(), "application/json")
        if url.path == "/api/play":
            return self.play(parse_qs(url.query))
        self.error(404, "bulunamadi")

    def play(self, q: dict):
        models = find_models(self.model_dirs)
        mid = q.get("model", [""])[0]
        if mid not in models:
            return self.error(400, "bilinmeyen model")
        try:
            depth = min(3, max(1, int(q.get("depth", ["2"])[0])))
            seed = int(q.get("seed", ["1"])[0])
            if not 0 <= seed < 2**63:
                raise ValueError
        except ValueError:
            return self.error(400, "derinlik/tohum gecersiz")
        if not PLAY.exists():
            return self.error(500, "build/play yok: cmake --build build")
        cmd = [str(PLAY), "--agent", "greedy" if depth == 1 else "expectimax", "--net", models[mid]["path"], "--seed", str(seed)]
        if depth > 1:
            cmd += ["--depth", str(depth)]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            for line in proc.stdout:
                self.wfile.write(f"data: {line.strip()}\n\n".encode())
                self.wfile.flush()
            err = proc.stderr.read().strip()
            if proc.wait() != 0:
                self.wfile.write(f"data: {json.dumps({'t': 'error', 'msg': err or 'play hatasi'})}\n\n".encode())
        except (BrokenPipeError, ConnectionResetError):
            pass  # browser stopped watching
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.wait()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8048)
    ap.add_argument("--models", nargs="*", default=[], help="extra model directories")
    args = ap.parse_args()
    Handler.model_dirs = [ROOT / "runs", ROOT / "models"] + [Path(m).expanduser().resolve() for m in args.models]
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"2048 konsey arayuzu: http://127.0.0.1:{args.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
