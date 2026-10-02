'use strict';
const $ = id => document.getElementById(id);
const token = document.querySelector('meta[name="connect6-token"]').content;
let state = null, pending = false, polling = false;
const cells = [];
function notice(text, error = false) { $('notice').textContent = text; $('notice').classList.toggle('error', error); }
for (let i = 0; i < 361; i++) {
  const button = document.createElement('button');
  const row = Math.floor(i / 19), col = i % 19;
  button.className = 'cell'; button.dataset.row = row; button.dataset.col = col;
  if ([3, 9, 15].includes(row) && [3, 9, 15].includes(col)) button.classList.add('star');
  button.tabIndex = i === 180 ? 0 : -1;
  const stone = document.createElement('span'); stone.className = 'stone'; button.append(stone);
  button.addEventListener('click', async () => {
    if (!state || pending || state.game.done || state.game.player !== state.game.human || state.game.board[i]) return;
    await action('/api/move', {cell: i});
    if (state && !state.game.done && state.game.player !== state.game.human && !state.running) await bot();
  });
  button.addEventListener('keydown', event => {
    const shifts = {ArrowLeft: -1, ArrowRight: 1, ArrowUp: -19, ArrowDown: 19};
    if (!(event.key in shifts)) return;
    event.preventDefault(); const next = Math.max(0, Math.min(360, i + shifts[event.key]));
    cells.forEach(cell => cell.tabIndex = -1); cells[next].tabIndex = 0; cells[next].focus();
  });
  cells.push(button); $('board').append(button);
}
function render() {
  if (!state) return;
  const g = state.game, last = g.moves[g.moves.length - 1];
  cells.forEach((button, i) => {
    button.classList.toggle('black', g.board[i] === 1); button.classList.toggle('white', g.board[i] === -1);
    button.classList.toggle('last', i === last);
    const color = g.board[i] === 1 ? 'black' : g.board[i] === -1 ? 'white' : 'empty';
    const label = `${String.fromCharCode(65 + i % 19)}${19 - Math.floor(i / 19)}, ${color}`;
    button.setAttribute('aria-label', label); button.title = label;
    button.setAttribute('aria-disabled', String(pending || g.done || g.player !== g.human || !!g.board[i]));
  });
  const color = g.player === 1 ? 'Black' : 'White';
  $('turn').textContent = g.done ? (g.winner ? `${g.winner === 1 ? 'Black' : 'White'} wins` : 'Draw') : `${color} ${g.player === g.human ? '· your turn' : '· bot to play'}`;
  $('placements').textContent = g.done ? 'GAME OVER' : `${g.left} stone${g.left === 1 ? '' : 's'} left`;
  $('move-count').textContent = `${g.moves.length} placements`;
  $('game-detail').textContent = state.running ? 'Pause training before requesting a bot turn.' : $('opponent').value === 'candidate' ? 'Experimental candidate: has not passed the strength gate.' : 'Black opens with one stone. Then two per turn.';
  $('opponent').querySelector('[value="candidate"]').disabled = !state.candidate_available;
  $('opponent').disabled = pending;
  $('bot-move').disabled = pending || state.running || g.done || g.player === g.human;
  $('new-game').disabled = pending;
  $('train').disabled = pending || state.running;
  $('stop').disabled = pending || !state.running;
  $('phase').textContent = (state.running ? state.phase : state.phase === 'error' ? 'error' : state.phase === 'not_started' ? 'not started' : 'paused').replaceAll('_', ' ').toUpperCase();
  $('games').textContent = (state.games || 0).toLocaleString();
  $('updates').textContent = (state.updates || 0).toLocaleString();
  $('replay').textContent = (state.replay_positions || 0).toLocaleString();
  $('loss').textContent = state.loss == null ? '—' : Number(state.loss).toFixed(3);
  const counts = state.selfplay_counts || {};
  $('tactical-training').textContent = `${(counts.proof_win || 0) + (counts.tss_win || 0) + (counts.proof_loss || 0)} tactical/TSS-adjudicated games · ${counts.forced || 0} forced placements · ${counts.mcts || 0} neural decisions, counted since game ${state.counted_from_game || 0}. Completed games include proven outcomes.`;
  $('storage').textContent = `${(state.artifact_bytes / 2**30).toFixed(2)} / ${((state.cap_bytes || 20 * 2**30) / 2**30).toFixed(1)} GiB`;
  $('vram').textContent = state.cuda_allocated_bytes ? `${(state.cuda_allocated_bytes / 2**30).toFixed(2)} GiB` : '—';
  $('device').textContent = (state.device || 'not initialized').toUpperCase();
  const elapsed = state.session_seconds || 0, remaining = state.remaining_seconds || 0;
  $('progress').style.width = `${elapsed + remaining ? 100 * elapsed / (elapsed + remaining) : 0}%`;
  $('time').textContent = state.running ? `${Math.ceil(remaining / 60)} min remaining` : 'Saved sessions can be resumed';
  $('incumbent').textContent = state.incumbent?.kind === 'network' ? `Validated network · update ${state.incumbent.step}` : state.incumbent?.label || 'Untrained tactical baseline';
  const report = state.last_evaluation;
  $('gate').textContent = state.gate_games ? `Held-out evaluation: ${state.gate_games} / ${state.gate_required_games} games.` : report ? `Last candidate: ${(report.score * 100).toFixed(1)}% score over ${report.games} games. Lower 95% bound: ${(report.lower_95 * 100).toFixed(1)}%. ${report.promoted ? 'Promoted.' : 'Not promoted.'}` : state.incumbent?.kind === 'network' ? 'Passed paired evaluation against the previous incumbent. Not a global ranking.' : 'No learned model has passed evaluation yet.';
  if (state.worker_error) notice(state.worker_error, true);
  else if (state.phase === 'error') notice(state.message, true);
}
async function refresh() {
  if (polling) return;
  polling = true;
  try {
    const response = await fetch('/api/status');
    if (!response.ok) throw new Error('Dashboard connection failed.');
    state = await response.json(); render();
  } catch (error) { notice(error.message, true); }
  finally { polling = false; }
}
async function action(path, body) {
  if (pending) return false;
  pending = true; render();
  let success = false;
  try {
    const response = await fetch(path, {method: 'POST', headers: {'Content-Type': 'application/json', 'X-Connect6-Token': token}, body: JSON.stringify(body)});
    const reply = await response.json();
    if (!response.ok) throw new Error(reply.error || 'Request failed.');
    notice(reply.message); success = true;
  } catch (error) { notice(error.message, true); }
  finally { pending = false; await refresh(); render(); }
  return success;
}
async function bot() { notice('Bot is thinking. A fresh worker may take a few seconds to start.'); await action('/api/bot', {seconds: Number($('seconds').value), candidate: $('opponent').value === 'candidate'}); }
$('opponent').addEventListener('change', render);
$('bot-move').addEventListener('click', bot);
$('new-game').addEventListener('click', async () => { await action('/api/new', {human: Number($('color').value)}); if (state && state.game.player !== state.game.human && !state.running) await bot(); });
$('train').addEventListener('click', () => {
  const hours = Number($('hours').value), disk = Number($('disk').value);
  if (hours > 2 && !confirm(`Run training for up to ${hours} hours? This uses your GPU and electricity until stopped.`)) return;
  action('/api/train', {hours, disk_gib: disk});
});
$('stop').addEventListener('click', () => action('/api/stop', {}));
refresh(); setInterval(refresh, 2000);
