"""Optional real-browser check: uv run --group browser python tests/ui_smoke.py.
Uses installed Chrome; set CHROME_EXECUTABLE to override. No browser download.
"""
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.request
from playwright.sync_api import sync_playwright, expect

ROOT = Path(__file__).resolve().parent.parent
chrome = os.environ.get("CHROME_EXECUTABLE", str(Path(os.environ.get("LOCALAPPDATA", "")) / "Google/Chrome/Application/chrome.exe"))
if not Path(chrome).is_file():
    raise SystemExit("Set CHROME_EXECUTABLE to an installed Chrome/Chromium executable.")
with socket.socket() as sock:
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
server = subprocess.Popen([sys.executable, "-m", "connect6", "web", "--port", str(port),
                           "--data", str(ROOT / "data" / "ui-test")], cwd=ROOT,
                          stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
try:
    url = f"http://127.0.0.1:{port}"
    for _ in range(120):
        try:
            urllib.request.urlopen(url, timeout=1).close()
            break
        except OSError:
            if server.poll() is not None:
                raise RuntimeError(server.stderr.read().decode())
            time.sleep(.25)
    else:
        raise RuntimeError("Dashboard failed to start.")
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=chrome, headless=True)
        page = browser.new_page(viewport={"width": 1400, "height": 1300}, device_scale_factor=1)
        errors = []
        page.on("pageerror", lambda exc: errors.append(str(exc)))
        page.goto(url)
        expect(page.locator('.cell')).to_have_count(361)
        expect(page.locator('#turn')).to_contain_text('your turn')
        page.locator('.cell').nth(180).click()
        expect(page.locator('.black, .white')).to_have_count(3, timeout=60000)
        assert 'your turn' in page.locator('#turn').inner_text()
        page.locator('.cell').nth(0).click()
        expect(page.locator('.black, .white')).to_have_count(4)
        assert '1 stone left' in page.locator('#placements').inner_text()
        page.locator('.cell').nth(1).click()
        expect(page.locator('.black, .white')).to_have_count(7, timeout=60000)
        page.screenshot(path=str(ROOT / 'docs' / 'dashboard.png'), full_page=True)
        # No token: same-origin mutations must still be rejected.
        assert page.evaluate("fetch('/api/new',{method:'POST',body:'{}'}).then(r=>r.status)") == 403
        page.set_viewport_size({"width": 390, "height": 844})
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        # Keyboard operation and mobile layout both retain all intersections.
        page.locator('.cell').nth(180).focus()
        page.keyboard.press('ArrowRight')
        assert page.locator('.cell').nth(181).evaluate('(el) => el === document.activeElement')
        # Exercise the real trainer start/stop boundary, including checkpoint persistence.
        page.locator('#hours').fill('0.01')
        await_start = page.locator('#train')
        await_start.click()
        expect(page.locator('#stop')).to_be_enabled(timeout=30000)
        page.locator('#stop').click()
        expect(page.locator('#train')).to_be_enabled(timeout=45000)
        assert (ROOT / 'data' / 'ui-test' / 'latest.pt').is_file()
        page.locator('#opponent').select_option('candidate')
        page.locator('#seconds').select_option('1')
        page.locator('#new-game').click()
        expect(page.locator('.black, .white')).to_have_count(0)
        page.locator('.cell').nth(180).click()
        expect(page.locator('.black, .white')).to_have_count(3, timeout=60000)
        latest = page.evaluate("fetch('/api/status').then(r=>r.json())")
        assert latest['bot_result']['kind'] == 'candidate'
        assert latest['incumbent']['kind'] == 'heuristic'  # viewing a candidate never promotes it
        expect(page.locator('#game-detail')).to_contain_text('has not passed')
        assert not errors, errors
        print("Browser checks passed: play, candidate preview without promotion, CSRF, mobile, keyboard, training start/stop/checkpoint, no JS errors.")
        browser.close()
finally:
    server.terminate()
    server.wait(timeout=15)
