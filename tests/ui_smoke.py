"""Optional Chrome check in disposable CPU runs; never trains from the browser.
Run: uv run --group browser python tests/ui_smoke.py
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
from engine.game import Game, Rules
from engine.storage import load_json, run_lock, save_json
from playwright.sync_api import sync_playwright, expect

ROOT = Path(__file__).resolve().parent.parent
chrome = os.environ.get('CHROME_EXECUTABLE', str(Path(os.environ.get('LOCALAPPDATA', '')) / 'Google/Chrome/Application/chrome.exe'))
if not Path(chrome).is_file():
    raise SystemExit('Set CHROME_EXECUTABLE to an installed Chrome/Chromium executable. No browser will be downloaded.')


def layout(page):
    return {selector: page.locator(selector).bounding_box() for selector in
            ('#board', '.play-controls', '.training-summary', '.selfplay-card',
             '#loss-chart', '#gate-chart', '.incumbent-card', 'footer')}


def assert_layout(before, after):
    for selector, bounds in before.items():
        assert bounds and after[selector], selector
        assert all(abs(after[selector][key] - value) <= 1 for key, value in bounds.items()), (
            selector, bounds, after[selector])


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
        page = browser.new_page(viewport={'width': 1920, 'height': 1080}, device_scale_factor=1, locale='zh-TW')
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.goto(url)
        status = page.request.get(url + '/api/status').json()
        size = status['rule_config']['height']
        expect(page.locator('.cell')).to_have_count(size * size)
        expect(page.locator('#turn')).to_contain_text('你的回合')
        expect(page.locator('#undo-move')).to_have_text('悔棋')
        expect(page.locator('[data-i18n="thinkingSimulations"]')).to_have_text('MCTS／子')
        stats = page.request.get(url + '/api/history').json()['selfplay']
        expect(page.locator('#selfplay-window')).to_have_text(f"{stats['games']:,} / 1,000")
        for selector, key, scale in (
            ('#selfplay-length', 'mean_placements', 1), ('#selfplay-turns', 'mean_turns', 1),
            ('#selfplay-median', 'median_placements', 1), ('#black-rate', 'black_win_rate', 100),
            ('#white-rate', 'white_win_rate', 100), ('#draw-rate', 'draw_rate', 100)):
            text = page.locator(selector).inner_text()
            if stats[key] is None:
                assert text == '—', (selector, text)
            else:
                assert abs(float(text.rstrip('%')) - scale * stats[key]) <= .050000001, (selector, text, stats[key])
        if stats['games']:
            expect(page.locator('#black-wins')).to_contain_text(f"{stats['black_wins']:,} 盤")
            expect(page.locator('#white-wins')).to_contain_text(f"{stats['white_wins']:,} 盤")
            expect(page.locator('#selfplay-draws')).to_contain_text(f"{stats['draws']:,} 盤")
        if connect6:
            # Live summaries and the very first loss point must not need a checkpoint.
            original_status = (data / 'status.json').read_bytes() if (data / 'status.json').exists() else None
            live = {'games': 10001, 'updates': 1, 'selfplay_summaries': [
                {'game': 10001, 'winner': 1, 'placements': 12, 'turns': 7, 'source_counts': {}}],
                'loss_history': [{'step': 1, 'games': 10001, 'loss': 3., 'policy_loss': 2., 'value_loss': 1.}]}
            save_json(data / 'status.json', live)
            page.evaluate('refresh()')
            expect(page.locator('#selfplay-range')).to_contain_text('10,001–10,001')
            expect(page.locator('#loss-chart polyline')).to_have_count(3)
            live['games'], live['updates'] = 10002, 2
            live['selfplay_summaries'].append({'game': 10002, 'winner': -1, 'placements': 14, 'turns': 8, 'source_counts': {}})
            live['loss_history'].append({'step': 2, 'games': 10002, 'loss': 2., 'policy_loss': 1.5, 'value_loss': .5})
            save_json(data / 'status.json', live)
            expect(page.locator('#selfplay-range')).to_contain_text('10,001–10,002')
            expect(page.locator('#loss-chart')).to_contain_text('2')
            expect(page.locator('#loss-value')).to_contain_text('2.0000')
            if original_status is None:
                (data / 'status.json').unlink()
            else:
                (data / 'status.json').write_bytes(original_status)
            page.evaluate('refresh()')
        if not connect6:
            assert stats['games'] == 16  # The separate 100-game evaluation must stay out.
        assert page.locator('#train').count() == 0
        assert page.locator('#stop').count() == 0
        for height in (1080, 960):  # Also allow room for browser chrome on a 1080p display.
            page.set_viewport_size({'width': 1920, 'height': height})
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth && document.documentElement.scrollHeight <= innerHeight')
            for selector in ('#board', '.play-controls', '.training-summary', '.selfplay-card', '#loss-chart', '#gate-chart', '.incumbent-card', 'footer'):
                bounds = page.locator(selector).bounding_box()
                assert bounds and bounds['y'] >= 0 and bounds['y'] + bounds['height'] <= height, (selector, bounds)
            card = page.locator('.selfplay-card').bounding_box()
            for selector in ('#selfplay-length', '#selfplay-turns', '#black-rate', '#white-rate', '#draw-rate', '#selfplay-median'):
                bounds = page.locator(selector).bounding_box()
                assert bounds and bounds['y'] + bounds['height'] <= card['y'] + card['height'], (selector, bounds, card)
            board = page.locator('#board').bounding_box()
            assert abs(board['width'] - board['height']) < 1
        page.set_viewport_size({'width': 1920, 'height': 1080})
        assert page.locator('#simulations option').evaluate_all('(options) => options.map(option => option.value)') == ['32', '64', '128', '256', '512', '1024', '2048', '4096']
        page.locator('#simulations').select_option('256')
        page.locator('#language').select_option('en')
        expect(page.locator('html')).to_have_attribute('lang', 'en')
        expect(page.locator('#turn')).to_contain_text('Your turn')
        expect(page.locator('#bot-move')).to_have_text('AI move')
        expect(page.locator('#undo-move')).to_have_text('Undo')
        expect(page.locator('#simulations')).to_have_value('256')
        assert page.evaluate("!/[\\u3400-\\u9fff]/u.test(document.body.innerText.replace('繁體中文', ''))")
        page.reload()
        expect(page.locator('#language')).to_have_value('en')
        expect(page.locator('#turn')).to_contain_text('Your turn')
        page.locator('#language').select_option('zh-Hant')
        expect(page.locator('html')).to_have_attribute('lang', 'zh-Hant')
        expect(page.locator('#turn')).to_contain_text('你的回合')
        expect(page.locator('#bot-move')).to_have_text('AI 落子')
        baseline = layout(page)
        assert page.evaluate("action('/api/move', {cell:-1})") is False
        expect(page.locator('#undo-move')).to_be_disabled()
        expect(page.locator('#notice')).to_be_visible()
        expect(page.locator('#notice')).to_have_class('error')
        assert_layout(baseline, layout(page))
        page.evaluate("notice('AI 正在思考…')")
        expect(page.locator('#notice')).to_be_visible()
        assert_layout(baseline, layout(page))
        assert page.evaluate('document.documentElement.scrollHeight <= innerHeight')
        page.evaluate("notice('')")
        assert_layout(baseline, layout(page))
        page.locator('#simulations').select_option('32')
        page.locator('.selfplay-card .stats-details summary').focus()
        page.keyboard.press('Enter')
        expect(page.locator('.selfplay-card .stats-details')).to_have_attribute('open', '')
        if stats['games']:
            expect(page.locator('#selfplay-length-range')).to_contain_text(f"最短 {stats['min_placements']} 子")
            if stats['source_counts']:
                expect(page.locator('#selfplay-sources')).not_to_contain_text('尚無落子來源資料')
        assert page.evaluate('document.documentElement.scrollHeight <= innerHeight')
        page.keyboard.press('Enter')
        expect(page.locator('.selfplay-card .stats-details')).not_to_have_attribute('open', '')
        with run_lock(data):
            page.evaluate('refresh()')
            elapsed = page.evaluate("""async () => {
                while (polling) await new Promise(resolve => setTimeout(resolve, 10));
                const original = window.fetch;
                window.fetch = async (...args) => {
                    if (args[0] === '/api/history') await new Promise(resolve => setTimeout(resolve, 1500));
                    return original(...args);
                };
                const started = performance.now();
                try { await action('/api/move', {cell: 0}); return performance.now() - started; }
                finally { window.fetch = original; }
            }""")
            expect(page.locator('.black')).to_have_count(1)
            assert elapsed < 800, f'Stone feedback waited for chart history: {elapsed:.0f}ms'
            print(f'Stone feedback: {elapsed:.0f}ms with history artificially delayed 1500ms')
            page.locator('#undo-move').click()
            expect(page.locator('.black, .white')).to_have_count(0)
            # A status request started before a placement must not erase that placement.
            assert page.evaluate("""async () => {
                while (polling) await new Promise(resolve => setTimeout(resolve, 10));
                const original = window.fetch;
                let release, ready;
                const delayed = new Promise(resolve => { release = resolve; });
                const captured = new Promise(resolve => { ready = resolve; });
                window.fetch = async (...args) => {
                    const response = await original(...args);
                    if (args[0] !== '/api/status') return response;
                    const old = await response.json(); ready(); await delayed;
                    return new Response(JSON.stringify(old));
                };
                try {
                    const poll = refresh(); await captured;
                    await action('/api/move', {cell: 0}); release(); await poll;
                    return state.game.moves.length === 1 && state.game.can_undo;
                } finally { release(); window.fetch = original; }
            }""")
            expect(page.locator('.black')).to_have_count(1)
            page.locator('#undo-move').click()
            expect(page.locator('.black, .white')).to_have_count(0)
        page.evaluate('async () => { while (polling) await new Promise(resolve => setTimeout(resolve, 10)); await refresh(); }')
        center = size * size // 2
        baseline = layout(page)
        thinking_layout = []
        def capture_thinking(route):
            thinking_layout.append(layout(page))
            route.continue_()
        page.route('**/api/bot', capture_thinking)
        page.locator('.cell').nth(center).click()
        expect(page.locator('.black, .white')).to_have_count(3 if connect6 else 2, timeout=60000)
        page.unroute('**/api/bot', capture_thinking)
        assert thinking_layout
        assert_layout(baseline, thinking_layout[0])
        assert_layout(baseline, layout(page))
        expect(page.locator('#turn')).to_contain_text('你的回合')
        expect(page.locator('#notice')).to_be_empty()
        expect(page.locator('#notice')).to_be_hidden()
        if connect6:
            page.locator('.cell').nth(0).click()
            expect(page.locator('.black, .white')).to_have_count(4)
            expect(page.locator('#placements')).to_contain_text('剩 1 子')
            page.locator('#undo-move').click()
            expect(page.locator('.black, .white')).to_have_count(3)
            expect(page.locator('#placements')).to_contain_text('剩 2 子')
            page.locator('.cell').nth(0).click()
            expect(page.locator('.black, .white')).to_have_count(4)
            page.locator('.cell').nth(1).click()
            expect(page.locator('.black, .white')).to_have_count(7, timeout=60000)
            page.locator('#undo-move').click()
            expect(page.locator('.black, .white')).to_have_count(4)
            expect(page.locator('#placements')).to_contain_text('剩 1 子')
            page.locator('#undo-move').click()
            expect(page.locator('.black, .white')).to_have_count(3)
            expect(page.locator('#placements')).to_contain_text('剩 2 子')
        else:
            expect(page.locator('#loss-chart polyline')).to_have_count(3)
            expect(page.locator('#gate-chart polyline')).to_have_count(2)
            expect(page.locator('.gate-card')).to_contain_text('55%')
            expect(page.locator('.incumbent-card')).to_contain_text('55%')
            threshold = page.locator('#gate-chart line[stroke-dasharray]').get_attribute('y1')
            assert abs(float(threshold) - 98.7) < .000001  # 55% on the 0..1 score axis.
            expect(page.locator('#gate-results')).to_contain_text('100/100')
            page.locator('#undo-move').click()
            expect(page.locator('.black, .white')).to_have_count(0)
            expect(page.locator('#undo-move')).to_be_disabled()
            best = page.evaluate("fetch('/api/status').then(r=>r.json()).then(s=>s.incumbent.file)")
            page.locator('#model').select_option('latest')
            page.locator('#new-game').click()
            expect(page.locator('.black, .white')).to_have_count(0)
            page.locator('.cell').nth(center).click()
            expect(page.locator('.black, .white')).to_have_count(2, timeout=60000)
            assert page.evaluate("fetch('/api/status').then(r=>r.json()).then(s=>s.incumbent.file)") == best
            (ROOT / '.native-cache').mkdir(exist_ok=True)
            page.screenshot(path=str(ROOT / '.native-cache' / 'ui-dashboard.png'))
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
            'X-Engine-Token':document.querySelector('meta[name="engine-token"]').content},body:'{}'}).then(r=>r.status)""") == 404
        assert page.evaluate("""fetch('/api/new',{method:'POST',headers:{'Content-Type':'application/json',
            'X-Engine-Token':document.querySelector('meta[name="engine-token"]').content},body:'{"human":true}'}).then(r=>r.status)""") == 400
        page.set_viewport_size({'width': 390, 'height': 844})
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
        baseline = layout(page)
        page.evaluate("notice('AI 正在思考…')")
        assert_layout(baseline, layout(page))
        page.evaluate("notice('Bot failed: ' + 'long error message '.repeat(50), true)")
        assert_layout(baseline, layout(page))
        assert page.locator('#notice').evaluate('(node) => node.scrollHeight > node.clientHeight')
        page.evaluate("notice('')")
        assert_layout(baseline, layout(page))
        page.locator('.cell').nth(center).focus()
        page.keyboard.press('ArrowRight')
        assert page.locator('.cell').nth(center + 1).evaluate('(cell) => cell === document.activeElement')
        # The data-directory lock blocks competing play while CLI work owns it.
        page.locator('#new-game').click()
        expect(page.locator('.black, .white')).to_have_count(0)
        expect(page.locator('#undo-move')).to_be_disabled()
        assert page.evaluate("action('/api/undo', {})") is False
        assert page.evaluate("action('/api/new', {human: -1})") is True
        for invalid in (True, 0, -1, 4097, 1.5, '32', None):
            assert page.evaluate("value => action('/api/bot', {simulations: value})", invalid) is False
        assert page.evaluate('state.game.moves.length') == 0
        assert page.evaluate("action('/api/new', {human: 1})") is True
        with run_lock(data):
            assert page.evaluate("""fetch('/api/move',{method:'POST',headers:{'Content-Type':'application/json',
                'X-Engine-Token':document.querySelector('meta[name="engine-token"]').content},body:JSON.stringify({cell:0})}).then(r=>r.status)""") == 200
            assert page.evaluate("""fetch('/api/bot',{method:'POST',headers:{'Content-Type':'application/json',
                'X-Engine-Token':document.querySelector('meta[name="engine-token"]').content},body:'{"simulations":32}'}).then(r=>r.status)""") == 400
        assert not errors, errors
        page.close()
    finally:
        server.terminate()
        server.wait(timeout=15)


if '--live-only' in sys.argv:
    with tempfile.TemporaryDirectory() as name, sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=chrome, headless=True, args=['--disable-gpu'])
        check(Path(name), ['--preset', 'connect6'], True, browser)
        browser.close()
    print('Live dashboard and responsive placement checks passed.')
    raise SystemExit(0)

with tempfile.TemporaryDirectory() as name:
    root = Path(name)
    data = root / 'gomoku'
    # Short milestone/turn budgets test artifacts and charts, not playing strength.
    result = subprocess.run([sys.executable, 'train.py', '--data', str(data), '--device', 'cpu', '--hours', '.02',
        '--parallel', '2', '--batch', '2', '--simulations', '2', '--snapshot-every', '8', '--seconds', '.02',
        '--max-games', '16', '--channels', '4', '--blocks', '1', '--tactical-ms', '0'], cwd=ROOT, capture_output=True, text=True, timeout=90)
    if result.returncode:
        raise RuntimeError(result.stderr)
    # The milestone tournament finishes before training returns at the game limit.
    report = 'gate-model-00000016.json'
    assert load_json(data / report)['decision_recorded']
    result = subprocess.run([sys.executable, '-m', 'engine', 'evaluate', '--data', str(data),
        '--report', report, '--device', 'cpu', '--hours', '.02'],
        cwd=ROOT, capture_output=True, text=True, timeout=90)
    if result.returncode:
        raise RuntimeError(result.stderr)
    # Legal statistics-only fixtures verify distinct placement/turn lengths and both colors on 19x19.
    standard = root / 'connect6-standard'
    rules = Rules(19, 19, 6, 2, 1)
    summaries = []
    for number, moves in enumerate((
        [0, 19, 20, 1, 2, 21, 22, 3, 4, 23, 40, 5],
        [200, 0, 1, 201, 202, 2, 3, 219, 220, 4, 359, 203, 204, 5]), 1):
        game = Game.from_moves(moves, rules=rules)
        assert game.done and game.winner == (1 if number == 1 else -1)
        summaries.append({'game': number, 'placements': len(moves), 'winner': game.winner,
            'turns': 1 + (len(moves)-rules.starter_stones+rules.stones_per_turn-1)//rules.stones_per_turn,
            'source_counts': {'mcts': len(moves)-1}})
    save_json(standard / 'selfplay-stats.json', {'summaries': summaries})
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=chrome, headless=True, args=['--disable-gpu'])
        check(data, [], False, browser)
        check(root / 'connect6-square', ['--connect', '6', '--board_size', '13*13',
              '--stones_per_turn', '2', '--starter-stones', '1'], True, browser)
        check(standard, ['--connect', '6', '--board_size', '19*19',
              '--stones_per_turn', '2', '--starter-stones', '1'], True, browser)
        browser.close()
    print('Browser checks passed: single viewport at 1920x1080/960, self-play stats/empty states, accessible details/history, 15x15/13x13/19x19 boards, full turns, charts, keyboard, mobile, CSRF, no browser training, run lock, no JS errors.')
