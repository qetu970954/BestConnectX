'use strict';
const $ = id => document.getElementById(id), NS = 'http://www.w3.org/2000/svg';
const token = document.querySelector('meta[name="engine-token"]').content;
const TEXT = {
  en: {
    title: 'Connection AI Lab', brand: 'CONNECTION AI', dashboard: '/ Results Dashboard', language: 'Language',
    languageLabel: 'Dashboard language', localOnly: 'LOCAL ONLY', ruleTitleDefault: 'CONFIGURABLE CONNECTION GAME',
    introLead: 'Self-play, ', introAccent: 'keeps improving.',
    ruleDetailDefault: 'AlphaZero-style policy/value learning with threat-space search. Train from the CLI; play and inspect measured results here.',
    boardRegion: 'Connection game board', board: 'BOARD', turnBlackFirst: 'Black moves first', cellsDefault: '81 cells',
    boardAriaDefault: 'Board; use arrow keys to move and Enter to play', gameDetailDefault: 'Connect five or more stones to win.',
    color: 'Your color', black: 'Black', white: 'White', model: 'Opponent', best: 'Current best', heuristic: 'Heuristic baseline',
    thinkingSimulations: 'MCTS/stone', undoMove: 'Undo', newGame: 'New game', aiMove: 'AI move',
    pauseTraining: 'Pause CLI training before bot play. MCTS limit is per stone; tactical moves can finish sooner. Model startup adds to the wait.', trainingStatusAria: 'Training status and CLI command',
    trainingStatus: 'TRAINING STATUS', notStarted: 'Not started', selfplayGames: 'Self-play games', optimizationUpdates: 'Learning updates',
    replayPositions: 'Replay positions', artifacts: 'Saved artifacts',
    trainCommandDefault: 'uv run python train.py · Run again to resume; Ctrl+C saves safely.', nativeEngine: 'Native C++ / LibTorch engine',
    runtime: 'Runtime breakdown', noTiming: 'No timing data yet',
    lossDiagnostic: 'Loss is a training diagnostic; tournaments decide provisional promotion.',
    recentSelfplay: 'LATEST 1,000 TRAINING SELF-PLAY GAMES', noCompletedSummary: 'No completed-game summaries yet',
    meanPlacements: 'Mean placements', meanTurns: 'Mean turns', medianPlacements: 'Median placements',
    lengthAndSources: 'Length range and move sources', noLength: 'No length data yet', noSources: 'No move-source data yet',
    selfplayDisclaimer: 'Updated during training (about every 3 seconds); evaluation games are excluded. Rates use all games, including draws, and do not measure playing strength.',
    trainingLoss: 'TRAINING LOSS', lossFormula: 'L = policy cross-entropy + value mean-square error', totalLoss: 'Total loss',
    policy: 'Policy', value: 'Value', chartSampling: '(sampled curves; complete raw data stays on disk)',
    lossChartAria: 'Training loss chart', noTrainingUpdates: 'No learning updates yet',
    gateTitle: 'CANDIDATE VS CURRENT BEST · 100 GAMES EACH', scoreLegend: 'Score: win 1, draw 0.5', winRate: 'Win rate',
    thresholdLegend: 'dashed line = provisional 55% promotion threshold', gateChartAria: 'Candidate score and win-rate chart',
    gateResultsAria: 'Model evaluation results',
    noGateResults: 'No milestone evaluation yet. The first milestone is the baseline; later tournaments use the newest frozen milestone against current best.',
    currentBest: 'CURRENT BEST', noAcceptedModel: 'No evaluated model yet',
    incumbentDisclaimer: 'A candidate needs at least 55% score and must pass timing checks for provisional promotion. This is not proof of strength; failed models and evaluation histories stay on disk.',
    footerTraining: 'FREESTYLE CONNECTION GAMES · LOCALLY TRAINED WEIGHTS',
    footerRanking: 'STRENGTH IS REPORTED FROM LOCAL PAIRED GAMES · NO EXTERNAL RANK CLAIM',
    noData: 'No data', gameOver: 'Game over', yourTurn: 'Your turn', aiTurn: 'AI turn', draw: 'Draw',
    trainingRunning: 'CLI training is running; pause it before asking the AI to move.',
    noMilestonePreview: 'No milestone model yet; heuristic preview',
    defaultStatus: 'Loss is diagnostic; a candidate needs at least 55% over 100 games for provisional promotion.',
    noSelfplaySummary: 'No completed-game summaries yet', sourceSuffix: 'Opening stones are excluded.',
    noModelRuntime: 'C++ / LibTorch · no trained model yet; native heuristic play is available',
    noTimingDetail: 'Search, inference, learning, and save times update during training; an unmeasured value is not zero cost.',
    promoted: 'promoted', retained: 'kept current best', evaluating: 'evaluating',
    initialBaseline: 'Initial milestone baseline; strength not validated', acceptedModel: 'Accepted model (provisional)',
    latestUnvalidated: 'Latest learner · not evaluated', currentBestSuffix: 'current best', thinking: 'AI is thinking…',
    phase_bootstrap: 'Bootstrap', phase_self_play: 'Self-play', phase_optimizing: 'Optimizing', phase_evaluating: 'Evaluating',
    phase_paused: 'Paused', phase_error: 'Error', phase_not_started: 'Not started',
    source_mcts: 'MCTS', source_heuristic: 'Heuristic', source_forced: 'Only legal candidate', source_proof_move: 'Short proof', source_tss_move: 'TSS proof',
    series_loss: 'total loss', series_policy_loss: 'policy', series_value_loss: 'value', series_score: 'score', series_win_rate: 'win rate',
    ruleTitle: ({size, connect}) => `${size}×${size} · CONNECT ${connect}`,
    ruleDetail: ({starter, stones}) => `Black opens with ${starter} stone${starter === 1 ? '' : 's'}; later turns use ${stones}. Training uses all 8 rotations/reflections.`,
    boardAria: ({size}) => `${size} by ${size} board; use arrow keys to move and Enter to play`,
    cellAria: ({x, y, who}) => `Column ${x}, row ${y}: ${who}`,
    winner: ({who}) => `${who} wins`, placementsLeft: ({count}) => `${count} placement${count === 1 ? '' : 's'} left this turn`,
    moveCount: ({count}) => `${formatNumber(count)} placement${count === 1 ? '' : 's'}`,
    winCondition: ({connect}) => `Connect ${connect} or more horizontally, vertically, or diagonally to win.`,
    milestoneStatus: ({next, minutes, share, parallel}) => `Next model: ${formatNumber(next)} games · ${minutes} min left · Full tournament after each milestone (${share}% evaluation) · ${parallel} concurrent games`,
    selfplayRange: ({first, last}) => `Games ${formatNumber(first)}–${formatNumber(last)} summarized; self-play move histories are not saved`,
    resultCount: ({label, count}) => `${label} · ${formatNumber(count)} game${count === 1 ? '' : 's'}`,
    lengthRange: ({min, max}) => `Shortest ${min} · longest ${max} placements; partial opening and final turns count toward length.`,
    sourceEntry: ({name, count, percent}) => `${name} ${formatNumber(count)} (${percent}%)`,
    modelRuntime: ({device, architecture, channels, blocks, workers}) => `C++ / LibTorch · ${device} · ${architecture} · ${channels} channels × ${blocks} residual blocks · ${workers} CPU workers`,
    nativeTimes: ({search, inference, learning, checkpoint}) => `Measured totals: native search ${search}s · inference ${inference}s · learning ${learning}s · saving ${checkpoint}s. These are not hardware-utilization rates.`,
    lossValue: ({loss, policy, value}) => `Total loss ${loss} · policy ${policy} · value ${value}`,
    gateResult: ({candidate, incumbent, games, wins, draws, losses, score, state}) => `${candidate} vs ${incumbent}: ${games}/100 games · ${wins}W ${draws}D ${losses}L · score ${score}${state ? ` · ${state}` : ''}`,
    milestoneChoice: ({games, suffix}) => `${formatNumber(games)}-game model${suffix ? ` · ${suffix}` : ''}`,
    acceptedAt: ({games}) => `Accepted model (provisional) · ${formatNumber(games)} self-play games`,
    frozenGate: ({done, total, game, stones}) => `Frozen gate: ${done}/${total} games · game ${game}, ${stones} stones.`,
    savedStatus: 'Saved. Run the same CLI command to resume; Ctrl+C requests a safe stop.'
  },
  'zh-Hant': {
    title: '連棋 AI 研究室', brand: '連棋 AI', dashboard: '/ 成果儀表板', language: '語言', languageLabel: 'Dashboard 語言',
    localOnly: '僅限本機', ruleTitleDefault: '可設定的連棋規則', introLead: '自我對弈，', introAccent: '持續改進。',
    ruleDetailDefault: 'AlphaZero 式策略／價值網路與威脅空間搜尋。訓練使用 CLI；此頁用於對戰和查看實測結果。',
    boardRegion: '連棋對戰', board: '棋盤', turnBlackFirst: '黑棋先下', cellsDefault: '81 格',
    boardAriaDefault: '棋盤；方向鍵移動，Enter 下棋', gameDetailDefault: '連成五子或以上即可獲勝。',
    color: '執子顏色', black: '黑棋', white: '白棋', model: '對戰模型', best: '目前最佳', heuristic: '啟發式基準',
    thinkingSimulations: 'MCTS／子', undoMove: '悔棋', newGame: '新對局', aiMove: 'AI 落子',
    pauseTraining: '請先暫停 CLI 訓練再對局。MCTS 上限按每次落子計算；戰術落子可能提早完成，載入模型也需時間。', trainingStatusAria: '訓練狀態與 CLI 指令',
    trainingStatus: '訓練狀態', notStarted: '尚未開始', selfplayGames: '自我對弈棋局', optimizationUpdates: '最佳化更新',
    replayPositions: 'Replay 位置', artifacts: '保存產物',
    trainCommandDefault: 'uv run python train.py · 再次執行以續訓，Ctrl+C 安全存檔。', nativeEngine: 'C++ / LibTorch 原生引擎',
    runtime: '執行時間', noTiming: '尚無量測資料', lossDiagnostic: 'loss 是訓練診斷指標；模型升級依對戰結果判定。',
    recentSelfplay: '最近 1,000 盤訓練自我對弈', noCompletedSummary: '尚無完整棋局摘要',
    meanPlacements: '平均落子數', meanTurns: '平均回合數', medianPlacements: '落子數中位數',
    lengthAndSources: '長度範圍與落子來源', noLength: '尚無長度資料', noSources: '尚無落子來源資料',
    selfplayDisclaimer: '訓練期間約每 3 秒更新；不含評估對局。勝率以全部棋局為分母（含和棋），不代表棋力。',
    trainingLoss: '訓練 Loss', lossFormula: 'L = 策略交叉熵 + 價值均方誤差', totalLoss: '總 loss', policy: '策略', value: '價值',
    chartSampling: '（曲線抽樣，原始數據完整保存）', lossChartAria: '訓練 loss 圖', noTrainingUpdates: '尚無訓練更新',
    gateTitle: '候選模型對目前最佳 · 每版 100 場', scoreLegend: '得分率：勝 1、和 0.5', winRate: '純勝率',
    thresholdLegend: '虛線為 55% 暫定升級門檻', gateChartAria: '候選版本配對對戰得分及勝率圖', gateResultsAria: '模型評估結果',
    noGateResults: '尚無里程碑評估。第一個里程碑是初始基準；之後以最新固定里程碑挑戰目前最佳。',
    currentBest: '目前最佳', noAcceptedModel: '尚無已通過評估的模型',
    incumbentDisclaimer: '候選總得分至少 55% 且通過計時檢查才可暫定升級，不代表已證實棋力。失敗版本和評估棋譜仍保留。',
    footerTraining: 'FREESTYLE CONNECTION GAMES · 自行訓練權重', footerRanking: '棋力依本機配對評估呈現，不宣稱外部排名',
    noData: '尚無資料', gameOver: '對局結束', yourTurn: '你的回合', aiTurn: 'AI 的回合', draw: '和棋',
    trainingRunning: 'CLI 訓練執行中；暫停後可請求 AI 落子。', noMilestonePreview: '尚無里程碑模型；啟發式預覽',
    defaultStatus: 'Loss 僅供診斷；100 場得分至少 55% 才可暫定升級。', noSelfplaySummary: '尚無完整棋局摘要',
    sourceSuffix: '不含預設開局子。', noModelRuntime: 'C++ / LibTorch · 尚無訓練模型；可使用原生啟發式對戰',
    noTimingDetail: '搜尋／推論／學習／保存時間隨 CLI 即時進度更新；未量測值不代表零成本。',
    promoted: '已升級', retained: '保留舊版', evaluating: '評估中', initialBaseline: '初始里程碑基準；棋力尚未驗證',
    acceptedModel: '已接受模型（暫定）', latestUnvalidated: '最新訓練模型 · 未驗證', currentBestSuffix: '目前最佳', thinking: 'AI 正在思考…',
    phase_bootstrap: '啟動資料', phase_self_play: '自我對弈', phase_optimizing: '最佳化', phase_evaluating: '評估中',
    phase_paused: '已暫停', phase_error: '錯誤', phase_not_started: '尚未開始',
    source_mcts: 'MCTS', source_heuristic: '啟發式', source_forced: '單一候選', source_proof_move: '短程證明', source_tss_move: 'TSS 證明',
    series_loss: '總 loss', series_policy_loss: '策略', series_value_loss: '價值', series_score: '得分率', series_win_rate: '純勝率',
    ruleTitle: ({size, connect}) => `${size}×${size} · 連${connect}棋`,
    ruleDetail: ({starter, stones}) => `黑棋開局 ${starter} 子，之後每回合 ${stones} 子。訓練採用 8 種旋轉／反射增強。`,
    boardAria: ({size}) => `${size}乘${size}棋盤；方向鍵移動，Enter 下棋`,
    cellAria: ({x, y, who}) => `${x},${y} ${who}`, winner: ({who}) => `${who}獲勝`,
    placementsLeft: ({count}) => `本回合剩 ${count} 子`, moveCount: ({count}) => `${formatNumber(count)} 手`,
    winCondition: ({connect}) => `橫、直或斜向連成 ${connect} 子以上即勝。`,
    milestoneStatus: ({next, minutes, share, parallel}) => `下一個模型：${formatNumber(next)} 場 · ${minutes} 分鐘剩餘 · 里程碑後完成整場評估（評估已用 ${share}%）· 同時 ${parallel} 盤`,
    selfplayRange: ({first, last}) => `第 ${formatNumber(first)}–${formatNumber(last)} 盤摘要；不保存自我對弈棋譜`,
    resultCount: ({label, count}) => `${label} · ${formatNumber(count)} 盤`,
    lengthRange: ({min, max}) => `最短 ${min} 子 · 最長 ${max} 子；開局與終局的未滿回合皆計入長度。`,
    sourceEntry: ({name, count, percent}) => `${name} ${formatNumber(count)}（${percent}%）`,
    modelRuntime: ({device, architecture, channels, blocks, workers}) => `C++ / LibTorch · ${device} · ${architecture} · ${channels} 通道 × ${blocks} 殘差區塊 · ${workers} CPU 工作執行緒`,
    nativeTimes: ({search, inference, learning, checkpoint}) => `累計實測：原生搜尋 ${search}s · 推論 ${inference}s · 學習 ${learning}s · 保存 ${checkpoint}s。不是硬體使用率。`,
    lossValue: ({loss, policy, value}) => `總 loss ${loss} · 策略 ${policy} · 價值 ${value}`,
    gateResult: ({candidate, incumbent, games, wins, draws, losses, score, state}) => `${candidate} vs ${incumbent}: ${games}/100 場 · ${wins}勝 ${draws}和 ${losses}敗 · 得分 ${score}${state ? ` · ${state}` : ''}`,
    milestoneChoice: ({games, suffix}) => `${formatNumber(games)} 場模型${suffix ? ` · ${suffix}` : ''}`,
    acceptedAt: ({games}) => `已接受模型（暫定）· ${formatNumber(games)} 盤自我對弈`,
    frozenGate: ({done, total, game, stones}) => `固定模型評估：${done}/${total} 盤 · 第 ${game} 盤，已下 ${stones} 子。`,
    savedStatus: '已保存。執行相同 CLI 指令可續訓；Ctrl+C 會要求安全停止。'
  }
};
let state = null, history = {metrics: [], gates: [], models: []}, pending = false, polling = false;
let size = 0, language = initialLanguage(), gameRevision = 0;
const cells = [];
let currentNotice = {kind: 'text', value: '', error: false};

function initialLanguage() {
  try {
    const saved = localStorage.getItem('bestconnectx-language');
    if (TEXT[saved]) return saved;
  } catch (_) {}
  const preferred = (navigator.languages || [navigator.language]).find(Boolean) || '';
  return /^zh(?:-|$)/i.test(preferred) ? 'zh-Hant' : 'en';
}
function t(key, values = {}) {
  const value = TEXT[language][key] ?? TEXT.en[key] ?? key;
  return typeof value === 'function' ? value(values) : value;
}
function formatNumber(value) { return Number(value || 0).toLocaleString(language === 'en' ? 'en-US' : 'zh-TW'); }
function applyLanguage(persist = true) {
  document.documentElement.lang = language;
  document.title = t('title');
  document.querySelectorAll('[data-i18n]').forEach(node => { node.textContent = t(node.dataset.i18n); });
  document.querySelectorAll('[data-i18n-aria]').forEach(node => { node.setAttribute('aria-label', t(node.dataset.i18nAria)); });
  $('language').value = language;
  if (persist) try { localStorage.setItem('bestconnectx-language', language); } catch (_) {}
  renderNotice();
  render();
}
function setLanguage(value, persist = true) {
  language = value === 'zh-Hant' ? 'zh-Hant' : 'en';
  applyLanguage(persist);
}
function notice(text, error = false) { currentNotice = {kind: 'text', value: text, error}; renderNotice(); }
function noticeKey(key, error = false) { currentNotice = {kind: 'key', value: key, error}; renderNotice(); }
function serverNotice(text, error = true) { currentNotice = {kind: 'server', value: text, error}; renderNotice(); }
function renderNotice() {
  const node = $('notice');
  node.textContent = currentNotice.kind === 'key' ? t(currentNotice.value)
    : currentNotice.kind === 'server' ? localizeServer(currentNotice.value) : currentNotice.value;
  node.classList.toggle('error', currentNotice.error);
}
function localizeServer(text) {
  if (language === 'en') return text;
  const exact = {
    'Dashboard connection failed.': 'Dashboard 連線失敗。', 'Request failed.': '請求失敗。',
    'Invalid local request token or origin.': '本機請求 token 或來源無效。', 'Expected a JSON object.': '需要 JSON 物件。',
    'Choose black or white.': '請選黑棋或白棋。', 'It is not your turn.': '現在不是你的回合。',
    'Pause the CLI training process before GPU-assisted bot play.': '請先暫停 CLI 訓練，再讓 GPU bot 落子。',
    'MCTS simulations must be an integer between 1 and 4096.': 'MCTS 模擬次數必須是 1 到 4096 的整數。',
    'No human placement to undo.': '尚無可悔棋的落子。',
    'Invalid model selection.': '模型選擇無效。', 'Bot reply crosses a turn boundary.': 'Bot 回覆跨過回合邊界。',
    'Bot did not finish its turn.': 'Bot 沒有完成整個回合。', 'Not found.': '找不到資源。'
  };
  if (exact[text]) return exact[text];
  return text.startsWith('Bot failed: ') ? `Bot 執行失敗：${text.slice(12)}` : text;
}
function localizeStatus(message) {
  if (!message) return t('defaultStatus');
  if (language === 'en') return message;
  if (message === TEXT.en.savedStatus) return t('savedStatus');
  const gate = /^Frozen gate: (\d+)\/(\d+) games \| game (\d+), (\d+) stones\.$/.exec(message);
  return gate ? t('frozenGate', {done: gate[1], total: gate[2], game: gate[3], stones: gate[4]}) : message;
}
function phaseLabel(phase) { return TEXT[language][`phase_${phase}`] || String(phase).replaceAll('_', ' '); }
function colorName(value) { return t(value === 1 ? 'black' : 'white'); }
function incumbentLabel(entry) {
  if (!entry?.file) return t('noMilestonePreview');
  if (entry.initial_baseline) return t('initialBaseline');
  if (entry.gate || entry.label?.startsWith('Accepted model')) return t('acceptedAt', {games: entry.games});
  return entry.label || entry.id;
}
function svg(name, attrs = {}, text = '') {
  const node = document.createElementNS(NS, name);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  node.textContent = text;
  return node;
}
function plot(id, rows, keys, percent = false) {
  const root = $(id), w = 600, h = 210, pad = 42;
  root.replaceChildren();
  const values = rows.flatMap(row => keys.map(key => row[key])).filter(value => Number.isFinite(value));
  if (!values.length) { root.append(svg('text', {x: pad, y: h / 2, fill: '#9aa9a7'}, t('noData'))); return; }
  const low = percent ? 0 : Math.min(0, ...values), high = percent ? 1 : Math.max(.01, ...values);
  const y = value => h - pad - (value - low) / (high - low) * (h - 2 * pad);
  const x = i => pad + i * (w - 2 * pad) / Math.max(1, rows.length - 1);
  for (let i = 0; i < 5; i++) {
    const value = low + i * (high - low) / 4;
    root.append(svg('line', {x1: pad, x2: w - pad, y1: y(value), y2: y(value), stroke: '#344246'}));
    root.append(svg('text', {x: 2, y: y(value) + 4, fill: '#9aa9a7', 'font-size': 11}, percent ? `${Math.round(value * 100)}%` : value.toFixed(2)));
  }
  if (percent) root.append(svg('line', {x1: pad, x2: w - pad, y1: y(.55), y2: y(.55), stroke: '#d3b477', 'stroke-dasharray': '6 5'}));
  const colors = ['#cff394', '#78c9f0', '#ef9565'];
  keys.forEach((key, series) => {
    const points = rows.flatMap((row, i) => Number.isFinite(row[key]) ? [`${x(i)},${y(row[key])}`] : []);
    root.append(svg('polyline', {points: points.join(' '), fill: 'none', stroke: colors[series], 'stroke-width': 2}));
    if (rows.length < 80) rows.forEach((row, i) => {
      if (!Number.isFinite(row[key])) return;
      const dot = svg('circle', {cx: x(i), cy: y(row[key]), r: 3, fill: colors[series]});
      dot.append(svg('title', {}, `${row.step ?? row.candidate}: ${t(`series_${key}`)} = ${row[key].toFixed(4)}`));
      root.append(dot);
    });
  });
  root.append(svg('text', {x: pad, y: h - 8, fill: '#9aa9a7', 'font-size': 11}, String(rows[0].step ?? rows[0].candidate)));
  root.append(svg('text', {x: w - pad, y: h - 8, fill: '#9aa9a7', 'font-size': 11, 'text-anchor': 'end'}, String(rows.at(-1).step ?? rows.at(-1).candidate)));
}
function buildBoard(game) {
  if (size === game.size) return;
  size = game.size;
  cells.length = 0;
  $('board').replaceChildren();
  $('board').style.gridTemplateColumns = `repeat(${size}, minmax(0,1fr))`;
  for (let i = 0; i < size * size; i++) {
    const button = document.createElement('button'), r = Math.floor(i / size), c = i % size;
    button.className = 'cell';
    button.dataset.row = r; button.dataset.col = c;
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
      event.preventDefault();
      const next = Math.max(0, Math.min(size * size - 1, i + delta));
      cells.forEach(cell => { cell.tabIndex = -1; });
      cells[next].tabIndex = 0; cells[next].focus();
    });
    cells.push(button); $('board').append(button);
  }
}
function render() {
  if (!state) return;
  const game = state.game, rules = state.rule_config, last = game.moves.at(-1);
  buildBoard(game);
  $('board').setAttribute('aria-label', t('boardAria', {size}));
  $('rule-title').textContent = t('ruleTitle', {size, connect: rules.connect});
  $('rule-detail').textContent = t('ruleDetail', {starter: rules.starter_stones, stones: rules.stones_per_turn});
  $('train-command').textContent = `uv run python train.py --connect ${rules.connect} --board_size "${size}*${size}" --stones_per_turn ${rules.stones_per_turn} --starter-stones ${rules.starter_stones} --data "${state.data_directory}"`;
  cells.forEach((button, i) => {
    button.classList.toggle('black', game.board[i] === 1); button.classList.toggle('white', game.board[i] === -1); button.classList.toggle('last', i === last);
    const who = game.board[i] === 1 ? t('black') : game.board[i] === -1 ? t('white') : language === 'en' ? 'Empty' : '空格';
    button.setAttribute('aria-label', t('cellAria', {x: i % size + 1, y: Math.floor(i / size) + 1, who}));
    button.setAttribute('aria-disabled', String(pending || game.done || game.player !== game.human || !!game.board[i]));
  });
  $('turn').textContent = game.done ? (game.winner ? t('winner', {who: colorName(game.winner)}) : t('draw'))
    : `${colorName(game.player)} · ${t(game.player === game.human ? 'yourTurn' : 'aiTurn')}`;
  $('placements').textContent = game.done ? t('gameOver') : t('placementsLeft', {count: game.left});
  $('move-count').textContent = t('moveCount', {count: game.moves.length});
  $('game-detail').textContent = state.running ? t('trainingRunning') : t('winCondition', {connect: rules.connect});
  $('bot-move').disabled = pending || state.running || game.done || game.player === game.human;
  $('new-game').disabled = pending; $('model').disabled = pending;
  $('undo-move').disabled = pending || !game.can_undo;
  $('phase').textContent = state.running ? phaseLabel(state.phase) : phaseLabel(state.phase === 'error' ? 'error' : state.phase === 'not_started' ? 'not_started' : 'paused');
  for (const [id, key] of [['games', 'games'], ['updates', 'updates'], ['replay', 'replay_positions']]) $(id).textContent = formatNumber(state[key]);
  $('storage').textContent = `${(Number(state.artifact_bytes || 0) / 2**30).toFixed(2)} GiB`;
  $('incumbent').textContent = incumbentLabel(state.incumbent);
  $('message').textContent = localizeStatus(state.message);
  $('milestone').textContent = t('milestoneStatus', {next: state.next_milestone || 10000,
    minutes: Math.ceil((state.remaining_seconds || 0) / 60), share: (100 * Number(state.evaluation_share || 0)).toFixed(1), parallel: state.parallel || 64});
  const selfplay = history.selfplay || {window: 1000, games: 0, source_counts: {}};
  $('selfplay-window').textContent = `${formatNumber(selfplay.games)} / ${formatNumber(selfplay.window)}`;
  $('selfplay-range').textContent = selfplay.games ? t('selfplayRange', {first: selfplay.first_game, last: selfplay.last_game}) : t('noSelfplaySummary');
  for (const [id, key] of [['selfplay-length', 'mean_placements'], ['selfplay-turns', 'mean_turns'], ['selfplay-median', 'median_placements']]) {
    $(id).textContent = Number.isFinite(selfplay[key]) ? selfplay[key].toFixed(1) : '—';
  }
  for (const [id, key] of [['black-rate', 'black_win_rate'], ['white-rate', 'white_win_rate'], ['draw-rate', 'draw_rate']]) {
    $(id).textContent = Number.isFinite(selfplay[key]) ? `${(100 * selfplay[key]).toFixed(1)}%` : '—';
  }
  for (const [id, key, label] of [['black-wins', 'black_wins', t('black') + (language === 'en' ? ' wins' : '勝')],
    ['white-wins', 'white_wins', t('white') + (language === 'en' ? ' wins' : '勝')], ['selfplay-draws', 'draws', t('draw')]]) {
    $(id).textContent = t('resultCount', {label, count: Number(selfplay[key] || 0)});
  }
  $('selfplay-length-range').textContent = selfplay.games ? t('lengthRange', {min: selfplay.min_placements, max: selfplay.max_placements}) : t('noLength');
  const sources = Object.entries(selfplay.source_counts || {}), decisions = sources.reduce((total, [, count]) => total + count, 0);
  $('selfplay-sources').textContent = decisions ? `${sources.map(([source, count]) => t('sourceEntry', {
    name: TEXT[language][`source_${source}`] || source, count, percent: (100 * count / decisions).toFixed(1)})).join(' · ')}. ${t('sourceSuffix')}` : t('noSources');
  const config = state.network, timing = state.timings || {};
  $('model-runtime').textContent = config ? t('modelRuntime', {device: state.device || (language === 'en' ? 'device not selected' : '尚未選擇裝置'),
    architecture: config.architecture || 'residual', channels: config.channels, blocks: config.blocks, workers: state.workers || 6}) : t('noModelRuntime');
  $('native-times').textContent = Object.keys(timing).length ? t('nativeTimes', {search: (timing.cpu_search_seconds || 0).toFixed(2),
    inference: (timing.inference_seconds || 0).toFixed(2), learning: (timing.learning_seconds || 0).toFixed(2), checkpoint: (timing.checkpoint_seconds || 0).toFixed(2)}) : t('noTimingDetail');
  const metric = state.loss_metrics || history.metrics.at(-1);
  plot('loss-chart', history.metrics, ['loss', 'policy_loss', 'value_loss']);
  $('loss-value').textContent = metric ? t('lossValue', {loss: metric.loss.toFixed(4), policy: metric.policy_loss.toFixed(4), value: metric.value_loss.toFixed(4)}) : t('noTrainingUpdates');
  plot('gate-chart', history.gates, ['score', 'win_rate'], true);
  $('gate-results').textContent = history.gates.length ? history.gates.map(row => t('gateResult', {candidate: row.candidate,
    incumbent: row.incumbent, games: row.games, wins: row.wins, draws: row.draws, losses: row.losses,
    score: row.score == null ? '—' : `${(100 * row.score).toFixed(1)}%`, state: row.promoted ? t('promoted') : row.complete ? t('retained') : t('evaluating')})).join(language === 'en' ? '; ' : '；') : t('noGateResults');
  const selector = $('model'), selected = selector.value;
  const choices = [{id: 'best', label: t('best')}, {id: 'heuristic', label: t('heuristic')},
    ...(state.candidate_available ? [{id: 'latest', label: t('latestUnvalidated')}] : []),
    ...history.models.map(model => ({id: model.file, label: t('milestoneChoice', {games: model.games,
      suffix: state.incumbent?.file === `models/${model.file}` ? t('currentBestSuffix') : ''})}))];
  selector.replaceChildren(...choices.map(choice => {
    const option = document.createElement('option'); option.value = choice.id; option.textContent = choice.label; return option;
  }));
  if (choices.some(choice => choice.id === selected)) selector.value = selected;
  if (state.phase === 'error') serverNotice(state.message, true);
}
async function refresh() {
  if (polling || pending) return;
  polling = true;
  const revision = gameRevision;
  try {
    await Promise.all([
      (async () => {
        const status = await fetch('/api/status');
        if (!status.ok) throw new Error('Dashboard connection failed.');
        const updated = await status.json();
        if (state && (pending || revision !== gameRevision)) updated.game = state.game;
        state = updated; render();
      })(),
      (async () => {
        const records = await fetch('/api/history');
        if (!records.ok) throw new Error('Dashboard connection failed.');
        history = await records.json(); render();
      })()
    ]);
  } catch (error) { serverNotice(error.message, true); }
  finally { polling = false; }
}
async function action(path, body) {
  if (pending) return false;
  pending = true; gameRevision++; render();
  let success = false;
  try {
    const response = await fetch(path, {method: 'POST', headers: {'Content-Type': 'application/json', 'X-Engine-Token': token}, body: JSON.stringify(body)});
    const reply = await response.json();
    if (!response.ok) throw new Error(reply.error || 'Request failed.');
    state.game = reply.game;
    notice(''); success = true;
  } catch (error) { serverNotice(error.message, true); }
  finally { pending = false; render(); }
  return success;
}
async function bot() { noticeKey('thinking'); await action('/api/bot', {model: $('model').value, simulations: Number($('simulations').value)}); }
$('language').addEventListener('change', event => setLanguage(event.target.value));
$('bot-move').addEventListener('click', bot);
$('undo-move').addEventListener('click', () => action('/api/undo', {}));
$('new-game').addEventListener('click', async () => {
  if (await action('/api/new', {human: Number($('color').value)}) && state.game.player !== state.game.human && !state.running) await bot();
});
applyLanguage(false);
refresh();
setInterval(refresh, 2000);
