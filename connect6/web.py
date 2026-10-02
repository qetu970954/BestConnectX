"""Loopback-only dashboard. No remote accounts, telemetry, or CDN dependencies."""
import json
import math
from pathlib import Path
import secrets
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from .game import Game
from .storage import busy, load_json, usage


ROOT = Path(__file__).resolve().parent.parent


def serve(port, data, open_browser=False):
    data = Path(data).resolve()
    data.mkdir(parents=True, exist_ok=True)
    token = secrets.token_urlsafe(32)
    origin = f"http://127.0.0.1:{port}"
    game = Game()
    human = 1
    process = None
    bot_result = None
    play_strategy = None

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def send(self, code, payload, content_type="application/json"):
            body = json.dumps(payload).encode() if content_type == "application/json" else payload
            self.send_response(code)
            self.send_header("Content-Type", content_type + "; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(body)

        def valid_host(self):
            return self.headers.get("Host") in (f"127.0.0.1:{port}", f"localhost:{port}")

        def do_GET(self):
            if not self.valid_host():
                return self.send(403, {"error": "Loopback host required."})
            if self.path == "/":
                html = (ROOT / "connect6" / "static" / "index.html").read_text(encoding="utf-8")
                return self.send(200, html.replace("__TOKEN__", token).encode(), "text/html")
            assets = {"/app.js": "text/javascript", "/style.css": "text/css"}
            if self.path in assets:
                return self.send(200, (ROOT / "connect6" / "static" / self.path[1:]).read_bytes(), assets[self.path])
            if self.path == "/api/status":
                state = load_json(data / "status.json", {"phase": "not_started", "games": 0, "updates": 0})
                state["incumbent"] = load_json(data / "incumbent.json", {"kind": "heuristic"})
                if not state.get("last_evaluation") and state["incumbent"].get("gate"):
                    report = load_json(data / state["incumbent"]["gate"], {})
                    state["last_evaluation"] = {k: v for k, v in report.items() if k != "matches"}
                state.update(running=busy(data) or (process is not None and process.poll() is None), artifact_bytes=usage(data),
                             game={"board": game.board.tolist(), "player": game.player,
                                   "left": game.left, "done": game.done, "winner": game.winner,
                                   "moves": game.moves, "human": human}, bot_result=bot_result,
                             candidate_available=(data / "latest.pt").is_file())
                if process is not None and process.poll() not in (None, 0):
                    state["worker_error"] = "Training worker exited with an error. See data/training.log."
                return self.send(200, state)
            self.send(404, {"error": "Not found."})

        def do_POST(self):
            nonlocal game, human, process, bot_result, play_strategy
            if (not self.valid_host() or self.headers.get("X-Connect6-Token") != token
                    or self.headers.get("Origin") not in (origin, f"http://localhost:{port}")):
                return self.send(403, {"error": "Invalid local request token or origin."})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 16384:
                    raise ValueError("Request body must be between 1 and 16384 bytes.")
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError("Expected a JSON object.")
                if self.path == "/api/train":
                    if busy(data) or (process is not None and process.poll() is None):
                        raise ValueError("A training or play worker is already active.")
                    hours = float(payload.get("hours", 2))
                    disk = float(payload.get("disk_gib", 20))
                    if not math.isfinite(hours) or not 0 < hours <= 168 or not math.isfinite(disk) or not .1 <= disk <= 10000:
                        raise ValueError("Invalid session duration or artifact cap.")
                    (data / "stop").unlink(missing_ok=True)
                    with open(data / "training.log", "wb") as log:
                        process = subprocess.Popen([sys.executable, "-m", "connect6", "train",
                            "--data", str(data), "--hours", str(hours), "--disk-gib", str(disk), "--keep-stop"],
                            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                    return self.send(200, {"message": "Starting a bounded session. Initialization can take a few seconds."})
                if self.path == "/api/stop":
                    (data / "stop").touch()
                    return self.send(200, {"message": "Stop requested. Waiting for the current operation and checkpoint."})
                if self.path == "/api/new":
                    color = payload.get("human", 1)
                    if type(color) is not int or color not in (-1, 1):
                        raise ValueError("Human color must be 1 (black) or -1 (white).")
                    game, human, bot_result, play_strategy = Game(), color, None, None
                elif self.path == "/api/move":
                    if game.player != human:
                        raise ValueError("It is the bot's turn.")
                    game.play(payload.get("cell"))
                elif self.path == "/api/bot":
                    if game.done or game.player == human:
                        raise ValueError("It is not the bot's turn.")
                    if busy(data) or (process is not None and process.poll() is None):
                        raise ValueError("Pause training before playing the bot; both share the GPU.")
                    seconds = float(payload.get("seconds", 5))
                    if not math.isfinite(seconds) or not .05 <= seconds <= 30:
                        raise ValueError("Thinking time must be between 0.05 and 30 seconds.")
                    candidate = payload.get("candidate", False)
                    if type(candidate) is not bool:
                        raise ValueError("Candidate selection must be true or false.")
                    if candidate and not (data / "latest.pt").is_file():
                        raise ValueError("No candidate checkpoint yet. Train a session first.")
                    command = [sys.executable, "-m", "connect6", "play", "--data", str(data),
                               "--seconds", str(seconds)] + (["--candidate"] if candidate else [])
                    request = {"moves": game.moves, "strategy": play_strategy}
                    result = subprocess.run(command, cwd=ROOT, input=json.dumps(request),
                                            capture_output=True, text=True, timeout=120)
                    if result.returncode:
                        raise RuntimeError("Bot worker failed: " + result.stderr[-1200:])
                    reply = json.loads(result.stdout)
                    updated = game.copy()
                    for move in reply["moves"]:
                        updated.play(move)
                    game, bot_result = updated, reply
                    play_strategy = reply.get("strategy")
                else:
                    return self.send(404, {"error": "Not found."})
                self.send(200, {"message": "OK"})
            except (ValueError, TypeError, KeyError) as exc:
                self.send(400, {"error": str(exc)})
            except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
                self.send(503, {"error": str(exc)})

    server = HTTPServer(("127.0.0.1", port), Handler)
    print(f"Connect6 dashboard: {origin}", flush=True)
    print("Local access only. No weights are downloaded. Ctrl+C closes the dashboard and requests training stop.", flush=True)
    try:
        if open_browser:
            import webbrowser
            webbrowser.open(origin)
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if process is not None and process.poll() is None:
            (data / "stop").touch()
            print("Training stop requested; allow the worker to finish its checkpoint.", flush=True)
        server.server_close()
