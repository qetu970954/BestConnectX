"""Optional Chrome check in disposable CPU runs; never trains from the browser.
Run: uv run --group browser python tests/connection_ui_smoke.py
"""
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request

# Keep this check within CPU permission, including bot subprocesses.
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from connect6.storage import run_lock
from playwright.sync_api import sync_playwright, expect

ROOT = Path(__file__).resolve().parent.parent
chrome = os.environ.get('CHROME_EXECUTABLE', str(Path(os.environ.get('LOCALAPPDATA', '')) / 'Google/Chrome/Application/chrome.exe'))
if not Path(chrome).is_file():
    raise SystemExit('Set CHROME_EXECUTABLE to an installed Chrome/Chromium executable. No browser will be downloaded.')


def check(data, flags, connect6, browser):
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    server = subprocess.Popen([sys.executable, 'dashboard.py', '--port', str(port), '--data', str(data), *flags],
                              cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        url = f'http://127.0.0.1:{port}'
        for _ in range(120):
            try:
                urllib.request.urlopen(url, timeout=1).close()
                break
            except OSError:
                if server.poll() is not None:
                    raise RuntimeError(server.stderr.read().decode())
                time.sleep(.1)
        else:
            raise RuntimeError('Dashboard did not start.')
        page = browser.new_page(viewport={'width': 1920, 'height': 1080}, device_scale_factor=1)
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.goto(url)
        expect(page.locator('.cell')).to_have_count(169 if connect6 else 81)
        expect(page.locator('#turn')).to_contain_text('你的回合')
        assert page.locator('#train').count() == 0
        assert page.locator('#stop').count() == 0
        for height in (1080, 960):  # Also allow room for browser chrome on a 1080p display.
            page.set_viewport_size({'width': 1920, 'height': height})
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth && document.documentElement.scrollHeight <= innerHeight')
            for selector in ('#board', '.play-controls', '.training-summary', '#loss-chart', '#gate-chart', '.incumbent-card', 'footer'):
                bounds = page.locator(selector).bounding_box()
                assert bounds and bounds['y'] >= 0 and bounds['y'] + bounds['height'] <= height, (selector, bounds)
            board = page.locator('#board').bounding_box()
            assert abs(board['width'] - board['height']) < 1
        page.set_viewport_size({'width': 1920, 'height': 1080})
        assert page.evaluate("action('/api/move', {cell:-1})") is False
        expect(page.locator('#notice')).to_be_visible()
        expect(page.locator('#notice')).to_have_class('error')
        page.evaluate("notice('AI 正在思考…')")
        expect(page.locator('#notice')).to_be_visible()
        assert page.evaluate('document.documentElement.scrollHeight <= innerHeight')
        page.evaluate("notice('')")
        page.locator('#seconds').fill('.02')
        center = 84 if connect6 else 40
        page.locator('.cell').nth(center).click()
        expect(page.locator('.black, .white')).to_have_count(3 if connect6 else 2, timeout=60000)
        expect(page.locator('#turn')).to_contain_text('你的回合')
        expect(page.locator('#notice')).to_be_empty()
        expect(page.locator('#notice')).to_be_hidden()
        if connect6:
            page.locator('.cell').nth(0).click()
            expect(page.locator('.black, .white')).to_have_count(4)
            expect(page.locator('#placements')).to_contain_text('剩 1 子')
            page.locator('.cell').nth(1).click()
            expect(page.locator('.black, .white')).to_have_count(7, timeout=60000)
        else:
            expect(page.locator('#loss-chart polyline')).to_have_count(3)
            expect(page.locator('#gate-chart polyline')).to_have_count(2)
            expect(page.locator('#gate-results')).to_contain_text('100/100')
            best = page.evaluate("fetch('/api/status').then(r=>r.json()).then(s=>s.incumbent.file)")
            page.locator('#model').select_option('latest')
            page.locator('#new-game').click()
            expect(page.locator('.black, .white')).to_have_count(0)
            page.locator('.cell').nth(center).click()
            expect(page.locator('.black, .white')).to_have_count(2, timeout=60000)
            assert page.evaluate("fetch('/api/status').then(r=>r.json()).then(s=>s.incumbent.file)") == best
            page.screenshot(path=str(ROOT / 'docs' / 'connection-dashboard.png'))
            # Long model history stays accessible inside its card, not below the viewport.
            page.evaluate("document.getElementById('gate-results').textContent = 'model-00002000 · 100/100 場 · 已升級；'.repeat(100)")
            assert page.evaluate('document.documentElement.scrollHeight <= innerHeight')
            assert page.locator('#gate-results').evaluate('(node) => node.scrollHeight > node.clientHeight')
            page.locator('#gate-results').focus()
            page.keyboard.press('End')
            page.wait_for_function("() => document.getElementById('gate-results').scrollTop > 0", timeout=1500)
            page.evaluate('refresh()')
        assert page.evaluate("fetch('/api/new',{method:'POST',body:'{}'}).then(r=>r.status)") == 403
        # Authenticated, same-origin requests still cannot launch or stop training.
        assert page.evaluate("""fetch('/api/train',{method:'POST',headers:{'Content-Type':'application/json',
            'X-Gomoku-Token':document.querySelector('meta[name="gomoku-token"]').content},body:'{}'}).then(r=>r.status)""") == 404
        assert page.evaluate("""fetch('/api/new',{method:'POST',headers:{'Content-Type':'application/json',
            'X-Gomoku-Token':document.querySelector('meta[name="gomoku-token"]').content},body:'{"human":true}'}).then(r=>r.status)""") == 400
        page.set_viewport_size({'width': 390, 'height': 844})
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
        page.locator('.cell').nth(center).focus()
        page.keyboard.press('ArrowRight')
        assert page.locator('.cell').nth(center + 1).evaluate('(cell) => cell === document.activeElement')
        # The data-directory lock blocks competing play while CLI work owns it.
        page.locator('#new-game').click()
        expect(page.locator('.black, .white')).to_have_count(0)
        with run_lock(data):
            assert page.evaluate("""fetch('/api/move',{method:'POST',headers:{'Content-Type':'application/json',
                'X-Gomoku-Token':document.querySelector('meta[name="gomoku-token"]').content},body:JSON.stringify({cell:0})}).then(r=>r.status)""") == 200
            assert page.evaluate("""fetch('/api/bot',{method:'POST',headers:{'Content-Type':'application/json',
                'X-Gomoku-Token':document.querySelector('meta[name="gomoku-token"]').content},body:'{"seconds":0.02}'}).then(r=>r.status)""") == 400
        assert not errors, errors
        page.close()
    finally:
        server.terminate()
        server.wait(timeout=15)


with tempfile.TemporaryDirectory() as name:
    root = Path(name)
    data = root / 'gomoku'
    # Short milestone/turn budgets test artifacts and charts, not playing strength.
    result = subprocess.run([sys.executable, 'train.py', '--data', str(data), '--device', 'cpu', '--hours', '.02',
        '--parallel', '2', '--batch', '2', '--simulations', '2', '--snapshot-every', '8', '--seconds', '.02',
        '--max-games', '16'], cwd=ROOT, capture_output=True, text=True, timeout=90)
    if result.returncode:
        raise RuntimeError(result.stderr)
    # Normal training intentionally leaves evaluation pending when its 20% credit is exhausted.
    result = subprocess.run([sys.executable, '-m', 'gomoku', 'evaluate', '--data', str(data),
        '--report', 'gate-model-00000016.json', '--device', 'cpu', '--hours', '.02'],
        cwd=ROOT, capture_output=True, text=True, timeout=90)
    if result.returncode:
        raise RuntimeError(result.stderr)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=chrome, headless=True)
        check(data, [], False, browser)
        check(root / 'connect6-square', ['--connect', '6', '--board_size', '13*13',
              '--stones_per_turn', '2', '--starter-stones', '1'], True, browser)
        browser.close()
    print('Browser checks passed: single viewport at 1920x1080/960, no OK banner, accessible history, 9x9/13x13 boards, full turns, charts, keyboard, mobile, CSRF, no browser training, run lock, no JS errors.')
