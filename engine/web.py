"""Local connection-game board and results; training itself uses the CLI."""
from collections import Counter
from functools import lru_cache
import json
import math
from pathlib import Path
from statistics import mean, median
import secrets
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from .game import DEFAULT_RULES, Game, Rules
from .storage import busy, load_json, save_json, usage

ROOT = Path(__file__).resolve().parent.parent
ASSETS = Path(__file__).resolve().parent / "static"


class _DashboardServer(HTTPServer):
    # Windows SO_REUSEADDR lets two dashboards silently share the same port.
    allow_reuse_address = sys.platform != "win32"
    allow_reuse_port = False


@lru_cache(maxsize=1)
def _selfplay_stats(paths):
    """Summarize immutable terminal exports; unfinished and evaluation games stay out."""
    records = []
    for path in reversed(paths):
        record = load_json(path, {})
        if record.get("complete") is True:
            records.append(record)
            if len(records) == 1000:
                break
    count = len(records)
    lengths = [len(row["moves"]) for row in records]
    turns = []
    for row, length in zip(records, lengths):
        rules = Rules(**row["rule_config"])
        turns.append(1 + max(0, math.ceil((length - rules.starter_stones) / rules.stones_per_turn)))
    wins = Counter(row["winner"] for row in records)
    sources = Counter(turn["source"] for row in records for turn in row.get("turns", []))
    return {"window": 1000, "games": count,
        "first_game": records[-1]["game"] if count else None,
        "last_game": records[0]["game"] if count else None,
        "black_wins": wins[1], "white_wins": wins[-1], "draws": wins[0],
        "black_win_rate": wins[1] / count if count else None,
        "white_win_rate": wins[-1] / count if count else None,
        "draw_rate": wins[0] / count if count else None,
        "mean_placements": mean(lengths) if count else None,
        "mean_turns": mean(turns) if count else None,
        "median_placements": median(lengths) if count else None,
        "min_placements": min(lengths, default=None), "max_placements": max(lengths, default=None),
        "source_counts": dict(sources)}


def history(data):
    # ponytail: recent 200 metric batches bound dashboard reads; all raw history stays on disk.
    rows = []
    for path in sorted((data / "metrics").glob("updates-*.json"))[-200:]:
        rows.extend(load_json(path, []))
    if len(rows) > 500:
        rows = [rows[i * (len(rows) - 1) // 499] for i in range(500)]
    gates = []
    for path in sorted(data.glob("gate-model-*.json")):
        report = load_json(path, {})
        gates.append({**{key: report.get(key, 0) for key in
                         ("games", "wins", "draws", "losses", "complete", "promoted")},
            "candidate": report.get("candidate", {}).get("id", path.stem),
            "incumbent": report.get("opponent", {}).get("id", report.get("incumbent")),
            "score": report.get("score"), "win_rate": report.get("win_rate")})
    models = [{"file": p.name, "games": int(p.stem.rsplit("-", 1)[-1])}
              for p in sorted((data / "models").glob("model-*.pt"))]
    # ponytail: scan archive names, but cache unchanged immutable records; index only if scans become slow.
    selfplay = _selfplay_stats(tuple(sorted((data / "selfplay").glob("game-*.json"))))
    return {"metrics": rows, "gates": gates, "models": models, "selfplay": selfplay}


def serve(port, data, open_browser=False, rules=DEFAULT_RULES):
    data = Path(data).resolve()
    data.mkdir(parents=True, exist_ok=True)
    identity = load_json(data / "run.json") or load_json(data / "rules.json")
    if identity:
        rules = Rules(**identity["rule_config"])
    else:
        if (data / "latest.pt").exists() or (data / "incumbent.json").exists():
            raise ValueError("Unrecognized data directory. Do not use legacy Connect6 artifacts here.")
        save_json(data / "run.json", {"rules": rules.id, "rule_config": rules.to_dict()})
    token = secrets.token_urlsafe(32)
    origin = f"http://127.0.0.1:{port}"
    game, human, strategy, model_used = Game(rules=rules), 1, None, None

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def send(self, code, value, content_type="application/json"):
            body = json.dumps(value, allow_nan=False).encode() if content_type == "application/json" else value
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
                html = (ASSETS / "index.html").read_text(encoding="utf-8").replace("__TOKEN__", token)
                return self.send(200, html.encode(), "text/html")
            static = {"/app.js": "app.js", "/style.css": "style.css"}
            if self.path in static:
                return self.send(200, (ASSETS / static[self.path]).read_bytes(),
                                 "text/javascript" if self.path.endswith(".js") else "text/css")
            if self.path == "/api/status":
                status = load_json(data / "status.json", {"phase": "not_started", "games": 0, "updates": 0})
                status["incumbent"] = load_json(data / "incumbent.json", {"kind": "heuristic", "id": "heuristic"})
                status["running"] = busy(data)
                status["artifact_bytes"] = usage(data)
                status["game"] = {"board": game.board.tolist(), "player": game.player,
                    "left": game.left, "done": game.done, "winner": game.winner,
                    "moves": game.moves, "human": human, "size": game.size}
                status["rule_config"] = rules.to_dict()
                status["candidate_available"] = (data / "latest.pt").is_file()
                status["data_directory"] = data.as_posix()
                return self.send(200, status)
            if self.path == "/api/history":
                return self.send(200, history(data))
            return self.send(404, {"error": "Not found."})

        def do_POST(self):
            nonlocal game, human, strategy, model_used
            if (not self.valid_host() or self.headers.get("X-Engine-Token") != token
                    or self.headers.get("Origin") not in (origin, f"http://localhost:{port}")):
                return self.send(403, {"error": "Invalid local request token or origin."})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 16384:
                    raise ValueError("Request body must be between 1 and 16384 bytes.")
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError("Expected a JSON object.")
                if self.path == "/api/new":
                    color = payload.get("human", 1)
                    if type(color) is not int or color not in (-1, 1):
                        raise ValueError("Choose black or white.")
                    game, human, strategy, model_used = Game(rules=rules), color, None, None
                elif self.path == "/api/move":
                    if game.done or game.player != human:
                        raise ValueError("It is not your turn.")
                    game.play(payload.get("cell"))
                elif self.path == "/api/bot":
                    if game.done or game.player == human:
                        raise ValueError("It is not the bot's turn.")
                    if busy(data):
                        raise ValueError("Pause the CLI training process before GPU-assisted bot play.")
                    seconds = float(payload.get("seconds", 5))
                    if not math.isfinite(seconds) or not .02 <= seconds <= 30:
                        raise ValueError("Thinking time must be between 0.02 and 30 seconds.")
                    model = payload.get("model", "best")
                    if not isinstance(model, str) or len(model) > 80:
                        raise ValueError("Invalid model selection.")
                    request = {"moves": game.moves, "strategy": strategy if model == model_used else None}
                    result = subprocess.run([sys.executable, "-m", "engine", "play", "--data", str(data),
                        "--model", model, "--seconds", str(seconds)], cwd=ROOT,
                        input=json.dumps(request), capture_output=True, text=True, timeout=120)
                    if result.returncode:
                        raise RuntimeError("Bot failed: " + result.stderr[-1200:])
                    reply = json.loads(result.stdout)
                    updated = game.copy()
                    color = updated.player
                    for move in reply["moves"]:
                        if updated.player != color:
                            raise ValueError("Bot reply crosses a turn boundary.")
                        updated.play(move)
                    if not updated.done and updated.player == color:
                        raise ValueError("Bot did not finish its turn.")
                    game, strategy, model_used = updated, reply.get("strategy"), model
                else:
                    return self.send(404, {"error": "Not found."})
                return self.send(200, {})
            except (ValueError, TypeError, KeyError) as exc:
                self.send(400, {"error": str(exc)})
            except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
                self.send(503, {"error": str(exc)})

    try:
        server = _DashboardServer(("127.0.0.1", port), Handler)
    except OSError as exc:
        raise OSError(exc.errno, f"Cannot start dashboard at {origin}: {exc}. "
                      "Close the existing dashboard or choose another --port.") from exc
    print(f"Local connection-game dashboard: {origin}. Run: {rules.id} ({data}). "
          "Training is controlled by train.py in another terminal.", flush=True)
    try:
        if open_browser:
            import webbrowser
            webbrowser.open(origin)
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
