'use strict';
const $ = id => document.getElementById(id), NS = 'http://www.w3.org/2000/svg';
const token = document.querySelector('meta[name="engine-token"]').content;
let state = null, history = {metrics: [], gates: [], models: []}, pending = false, polling = false;
let size = 0;
const cells = [];
function notice(text, error = false) { $('notice').textContent = text; $('notice').classList.toggle('error', error); }
function svg(name, attrs = {}, text = '') { const node = document.createElementNS(NS, name); for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value); node.textContent = text; return node; }
function plot(id, rows, keys, percent = false) {
  const root = $(id), w = 600, h = 210, pad = 42;
  root.replaceChildren();
  const values = rows.flatMap(row => keys.map(key => row[key])).filter(value => Number.isFinite(value));
  if (!values.length) { root.append(svg('text', {x: pad, y: h / 2, fill: '#9aa9a7'}, '尚無資料')); return; }
  const low = percent ? 0 : Math.min(0, ...values), high = percent ? 1 : Math.max(.01, ...values);
  const y = value => h - pad - (value - low) / (high - low) * (h - 2 * pad);
  const x = i => pad + i * (w - 2 * pad) / Math.max(1, rows.length - 1);
  for (let i = 0; i < 5; i++) {
    const value = low + i * (high - low) / 4;
    root.append(svg('line', {x1: pad, x2: w - pad, y1: y(value), y2: y(value), stroke: '#344246'}));
    root.append(svg('text', {x: 2, y: y(value) + 4, fill: '#9aa9a7', 'font-size': 11}, percent ? `${Math.round(value * 100)}%` : value.toFixed(2)));
  }
  if (percent) root.append(svg('line', {x1: pad, x2: w - pad, y1: y(.5), y2: y(.5), stroke: '#d3b477', 'stroke-dasharray': '6 5'}));
  const colors = ['#cff394', '#78c9f0', '#ef9565'];
  keys.forEach((key, series) => {
    const points = rows.flatMap((row, i) => Number.isFinite(row[key]) ? [`${x(i)},${y(row[key])}`] : []);
    root.append(svg('polyline', {points: points.join(' '), fill: 'none', stroke: colors[series], 'stroke-width': 2}));
    if (rows.length < 80) rows.forEach((row, i) => {
      if (!Number.isFinite(row[key])) return;
      const dot = svg('circle', {cx: x(i), cy: y(row[key]), r: 3, fill: colors[series]});
      dot.append(svg('title', {}, `${row.step ?? row.candidate}: ${key} = ${row[key].toFixed(4)}`)); root.append(dot);
    });
  });
  root.append(svg('text', {x: pad, y: h - 8, fill: '#9aa9a7', 'font-size': 11}, String(rows[0].step ?? rows[0].candidate)));
  root.append(svg('text', {x: w - pad, y: h - 8, fill: '#9aa9a7', 'font-size': 11, 'text-anchor': 'end'}, String(rows.at(-1).step ?? rows.at(-1).candidate)));
}
function buildBoard(game) {
  if (size === game.size) return;
  size = game.size; cells.length = 0; $('board').replaceChildren();
  $('board').style.gridTemplateColumns = `repeat(${size}, minmax(0,1fr))`;
  $('board').setAttribute('aria-label', `${size}乘${size}棋盤；方向鍵移動，Enter下棋`);
  for (let i = 0; i < size * size; i++) {
    const button = document.createElement('button'), r = Math.floor(i / size), c = i % size;
    button.className = 'cell'; button.dataset.row = r; button.dataset.col = c;
    button.classList.toggle('edge-right', c === size - 1); button.classList.toggle('edge-bottom', r === size - 1);
    if (r === Math.floor(size / 2) && c === Math.floor(size / 2)) button.classList.add('star');
    button.tabIndex = r === Math.floor(size / 2) && c === Math.floor(size / 2) ? 0 : -1;
    const stone = document.createElement('span'); stone.className = 'stone'; button.append(stone);
    button.addEventListener('click', async () => {
      if (!state || pending || state.game.done || state.game.player !== state.game.human || state.game.board[i]) return;
      if (await action('/api/move', {cell: i}) && !state.game.done && state.game.player !== state.game.human && !state.running) await bot();
    });
    button.addEventListener('keydown', event => {
      const delta = {ArrowLeft: -1, ArrowRight: 1, ArrowUp: -size, ArrowDown: size}[event.key];
      if (delta === undefined) return;
      event.preventDefault(); const next = Math.max(0, Math.min(size * size - 1, i + delta));
      cells.forEach(cell => cell.tabIndex = -1); cells[next].tabIndex = 0; cells[next].focus();
    });
    cells.push(button); $('board').append(button);
  }
}
function render() {
  if (!state) return;
  const game = state.game, rules = state.rule_config, last = game.moves.at(-1);
  buildBoard(game);
  $('rule-title').textContent = `${size}×${size} · 連${rules.connect}棋`;
  $('rule-detail').textContent = `黑棋開局 ${rules.starter_stones} 子，之後每回合 ${rules.stones_per_turn} 子。訓練採用 8 種旋轉／反射增強。`;
  $('train-command').textContent = `uv run python train.py --connect ${rules.connect} --board_size "${size}*${size}" --stones_per_turn ${rules.stones_per_turn} --starter-stones ${rules.starter_stones} --data "${state.data_directory}"`;
  cells.forEach((button, i) => {
    button.classList.toggle('black', game.board[i] === 1); button.classList.toggle('white', game.board[i] === -1); button.classList.toggle('last', i === last);
    const who = game.board[i] === 1 ? '黑棋' : game.board[i] === -1 ? '白棋' : '空格';
    button.setAttribute('aria-label', `${i % size + 1},${Math.floor(i / size) + 1} ${who}`);
    button.setAttribute('aria-disabled', String(pending || game.done || game.player !== game.human || !!game.board[i]));
  });
  $('turn').textContent = game.done ? (game.winner ? `${game.winner === 1 ? '黑棋' : '白棋'}獲勝` : '和棋') : `${game.player === 1 ? '黑棋' : '白棋'}${game.player === game.human ? ' · 你的回合' : ' · AI 的回合'}`;
  $('placements').textContent = game.done ? '對局結束' : `本回合剩 ${game.left} 子`;
  $('move-count').textContent = `${game.moves.length} 手`;
  $('game-detail').textContent = state.running ? 'CLI 訓練執行中；暫停後可請求 AI 落子。' : `橫、直或斜向連成 ${rules.connect} 子以上即勝。`;
  $('bot-move').disabled = pending || state.running || game.done || game.player === game.human;
  $('new-game').disabled = pending; $('model').disabled = pending;
  $('phase').textContent = state.running ? String(state.phase).replaceAll('_', ' ') : state.phase === 'error' ? 'error' : state.phase === 'not_started' ? '尚未開始' : '已暫停';
  for (const [id, key] of [['games', 'games'], ['updates', 'updates'], ['replay', 'replay_positions']]) $(id).textContent = Number(state[key] || 0).toLocaleString();
  $('storage').textContent = `${(Number(state.artifact_bytes || 0) / 2**30).toFixed(2)} GiB`;
  $('incumbent').textContent = state.incumbent?.label || state.incumbent?.id || '尚無里程碑模型；啟發式預覽';
  $('message').textContent = state.message || 'Loss 僅供診斷；是否升級以 100 場實戰得分為準。';
  $('milestone').textContent = `下一個模型：${Number(state.next_milestone || 1000).toLocaleString()} 場 · ${Math.ceil((state.remaining_seconds || 0) / 60)} 分鐘剩餘 · 時間目標 80/20（評估已用 ${(100 * Number(state.evaluation_share || 0)).toFixed(1)}%）· 同時 ${state.parallel || 64} 盤`;
  const selfplay = history.selfplay || {window: 1000, games: 0, source_counts: {}};
  $('selfplay-window').textContent = `${selfplay.games.toLocaleString()} / ${selfplay.window.toLocaleString()}`;
  $('selfplay-range').textContent = selfplay.games ? `第 ${selfplay.first_game.toLocaleString()}–${selfplay.last_game.toLocaleString()} 盤摘要；不保存自我對弈棋譜` : '尚無完整棋局摘要';
  for (const [id, key] of [['selfplay-length', 'mean_placements'], ['selfplay-turns', 'mean_turns'], ['selfplay-median', 'median_placements']]) {
    $(id).textContent = Number.isFinite(selfplay[key]) ? selfplay[key].toFixed(1) : '—';
  }
  for (const [id, key] of [['black-rate', 'black_win_rate'], ['white-rate', 'white_win_rate'], ['draw-rate', 'draw_rate']]) {
    $(id).textContent = Number.isFinite(selfplay[key]) ? `${(100 * selfplay[key]).toFixed(1)}%` : '—';
  }
  for (const [id, key, label] of [['black-wins', 'black_wins', '黑棋勝'], ['white-wins', 'white_wins', '白棋勝'], ['selfplay-draws', 'draws', '和棋']]) {
    $(id).textContent = `${label} · ${Number(selfplay[key] || 0).toLocaleString()} 盤`;
  }
  $('selfplay-length-range').textContent = selfplay.games ? `最短 ${selfplay.min_placements} 子 · 最長 ${selfplay.max_placements} 子；開局與終局的未滿回合皆計入長度。` : '尚無長度資料';
  const sources = Object.entries(selfplay.source_counts), decisions = sources.reduce((total, [, count]) => total + count, 0);
  const sourceNames = {mcts: 'MCTS', heuristic: '啟發式', forced: '單一候選', proof_move: '短程證明', tss_move: 'TSS 證明'};
  $('selfplay-sources').textContent = decisions ? `${sources.map(([source, count]) => `${sourceNames[source] || source} ${count.toLocaleString()}（${(100 * count / decisions).toFixed(1)}%）`).join(' · ')}。不含預設開局子。` : '尚無落子來源資料';
  const config = state.network, timing = state.timings || {};
  $('model-runtime').textContent = config ? `C++ / LibTorch · ${state.device || '尚未選擇裝置'} · ${config.channels} 通道 × ${config.blocks} 殘差區塊 · ${state.workers || 6} CPU 工作執行緒` : 'C++ / LibTorch · 尚無訓練模型；可使用原生啟發式對戰';
  $('native-times').textContent = Object.keys(timing).length ? `累計實測：原生搜尋 ${(timing.cpu_search_seconds || 0).toFixed(2)}s · 推論 ${(timing.inference_seconds || 0).toFixed(2)}s · 學習 ${(timing.learning_seconds || 0).toFixed(2)}s · 保存 ${(timing.checkpoint_seconds || 0).toFixed(2)}s。不是硬體使用率。` : '搜尋／推論／學習／保存時間會在 CLI 檢查點更新；未量測值不代表零成本。';
  const metric = state.loss_metrics || history.metrics.at(-1);
  plot('loss-chart', history.metrics, ['loss', 'policy_loss', 'value_loss']);
  $('loss-value').textContent = metric ? `總 loss ${metric.loss.toFixed(4)} · 策略 ${metric.policy_loss.toFixed(4)} · 價值 ${metric.value_loss.toFixed(4)}` : '尚無訓練更新';
  plot('gate-chart', history.gates, ['score', 'win_rate'], true);
  $('gate-results').textContent = history.gates.length ? history.gates.map(row => `${row.candidate} vs ${row.incumbent}: ${row.games}/100 場 · ${row.wins}勝 ${row.draws}和 ${row.losses}敗 · 得分 ${row.score == null ? '—' : (100 * row.score).toFixed(1) + '%'}${row.promoted ? ' · 已升級' : row.complete ? ' · 保留舊版' : ' · 評估中'}`).join('；') : '第一個里程碑作為初始比較基準；後續候選挑戰目前最佳版本。';
  const selector = $('model'), selected = selector.value;
  const choices = [{id: 'best', label: '目前最佳'}, {id: 'heuristic', label: '啟發式基準'},
    ...(state.candidate_available ? [{id: 'latest', label: '最新訓練模型 · 未驗證'}] : []),
    ...history.models.map(model => ({id: model.file, label: `${model.games.toLocaleString()} 場模型${state.incumbent?.file === `models/${model.file}` ? ' · 目前最佳' : ''}`}))];
  selector.replaceChildren(...choices.map(choice => { const option = document.createElement('option'); option.value = choice.id; option.textContent = choice.label; return option; }));
  if (choices.some(choice => choice.id === selected)) selector.value = selected;
  if (state.phase === 'error') notice(state.message, true);
}
async function refresh() {
  if (polling) return; polling = true;
  try {
    const [status, records] = await Promise.all([fetch('/api/status'), fetch('/api/history')]);
    if (!status.ok || !records.ok) throw new Error('Dashboard connection failed.');
    state = await status.json(); history = await records.json(); render();
  } catch (error) { notice(error.message, true); }
  finally { polling = false; }
}
async function action(path, body) {
  if (pending) return false; pending = true; render(); let success = false;
  try {
    const response = await fetch(path, {method: 'POST', headers: {'Content-Type': 'application/json', 'X-Engine-Token': token}, body: JSON.stringify(body)});
    const reply = await response.json(); if (!response.ok) throw new Error(reply.error || 'Request failed.');
    notice(''); success = true;
  } catch (error) { notice(error.message, true); }
  finally { pending = false; await refresh(); render(); }
  return success;
}
async function bot() { notice('AI 正在思考…'); await action('/api/bot', {model: $('model').value, seconds: Number($('seconds').value)}); }
$('bot-move').addEventListener('click', bot);
$('new-game').addEventListener('click', async () => {
  if (await action('/api/new', {human: Number($('color').value)}) && state.game.player !== state.game.human && !state.running) await bot();
});
refresh(); setInterval(refresh, 2000);
