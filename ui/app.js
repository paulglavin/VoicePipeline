/* ---------------------------------------------------------------------------
   Theme
   --------------------------------------------------------------------------- */

const _THEME_KEY = 'vp-theme';
const _mq = window.matchMedia('(prefers-color-scheme: dark)');

function _applyTheme(pref) {
  const resolved = pref === 'system' ? (_mq.matches ? 'dark' : 'light') : pref;
  document.documentElement.setAttribute('data-theme', resolved);
  document.querySelectorAll('.theme-btn').forEach(btn => {
    btn.classList.toggle('is-active', btn.dataset.themeVal === pref);
  });
}

function initTheme() {
  const saved = localStorage.getItem(_THEME_KEY) || 'system';
  _applyTheme(saved);

  document.querySelectorAll('.theme-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const val = btn.dataset.themeVal;
      localStorage.setItem(_THEME_KEY, val);
      _applyTheme(val);
    });
  });

  // Track OS-level changes when in system mode
  _mq.addEventListener('change', () => {
    if ((localStorage.getItem(_THEME_KEY) || 'system') === 'system') {
      _applyTheme('system');
    }
  });
}

/* ---------------------------------------------------------------------------
   API
   --------------------------------------------------------------------------- */

const API = '/api/v1';

async function apiFetch(path, options = {}) {
  const res = await fetch(API + path, options);
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(body.detail || res.statusText);
  }
  return res.json();
}

/* ---------------------------------------------------------------------------
   Toast
   --------------------------------------------------------------------------- */

let _toastTimer = null;

function showToast(msg) {
  const el = document.getElementById('toast');
  el.textContent = msg;
  el.classList.add('is-visible');
  clearTimeout(_toastTimer);
  _toastTimer = setTimeout(() => el.classList.remove('is-visible'), 3500);
}

/* ---------------------------------------------------------------------------
   Audio
   --------------------------------------------------------------------------- */

let _activeAudio = null;
let _activePlayBtn = null;

function stopActiveAudio() {
  if (_activeAudio) {
    _activeAudio.pause();
    _activeAudio.currentTime = 0;
  }
  if (_activePlayBtn) {
    _activePlayBtn.classList.remove('is-playing');
    _activePlayBtn.querySelector('.play-icon').textContent = '▶';
  }
  _activeAudio = null;
  _activePlayBtn = null;
}

function togglePlay(url, btn) {
  // Clicking the currently-playing button stops it
  if (_activePlayBtn === btn) {
    stopActiveAudio();
    return;
  }
  stopActiveAudio();

  const audio = new Audio(url);
  _activeAudio = audio;
  _activePlayBtn = btn;
  btn.classList.add('is-playing');
  btn.querySelector('.play-icon').textContent = '■';

  audio.play().catch(() => {
    showToast('Could not play audio — file may be unavailable.');
    stopActiveAudio();
  });

  audio.addEventListener('ended', () => {
    if (_activeAudio === audio) stopActiveAudio();
  });
}

function makePlayBtn(label, url) {
  const btn = document.createElement('button');
  btn.className = 'btn-play';
  btn.innerHTML = `<span class="play-icon">▶</span><span>${label}</span>`;
  if (!url) {
    btn.disabled = true;
    btn.title = 'Audio unavailable';
  } else {
    btn.addEventListener('click', () => togglePlay(url, btn));
  }
  return btn;
}

/* ---------------------------------------------------------------------------
   Dropdown — one open at a time, close on outside click
   --------------------------------------------------------------------------- */

let _openDropdown = null;

document.addEventListener('click', () => {
  if (_openDropdown) {
    _openDropdown.querySelector('.dropdown-menu').classList.remove('is-open');
    _openDropdown = null;
  }
});

function toggleDropdown(wrapper, e) {
  e.stopPropagation();
  const menu = wrapper.querySelector('.dropdown-menu');
  const opening = !menu.classList.contains('is-open');

  // Close any other open dropdown first
  if (_openDropdown && _openDropdown !== wrapper) {
    _openDropdown.querySelector('.dropdown-menu').classList.remove('is-open');
  }

  menu.classList.toggle('is-open', opening);
  _openDropdown = opening ? wrapper : null;
}

/* ---------------------------------------------------------------------------
   Shared speaker cache (populated when pending tab loads)
   --------------------------------------------------------------------------- */

let _speakersCache = [];

async function refreshSpeakerCache() {
  try {
    _speakersCache = await apiFetch('/speakers');
  } catch {
    _speakersCache = [];
  }
}

/* ---------------------------------------------------------------------------
   Pending tab
   --------------------------------------------------------------------------- */

async function loadPending() {
  await refreshSpeakerCache();
  try {
    const items = await apiFetch('/interactions/pending');
    const countEl = document.getElementById('pending-count');
    countEl.textContent = items.length || '';

    const list = document.getElementById('pending-list');
    list.innerHTML = '';

    if (items.length === 0) {
      list.innerHTML = '<div class="empty-state">No pending interactions — all clear.</div>';
      return;
    }

    items.forEach(item => list.appendChild(buildPendingCard(item)));
  } catch (err) {
    showToast('Failed to load pending: ' + err.message);
  }
}

function buildPendingCard(item) {
  const unknown = !item.matched_speaker;

  const card = document.createElement('div');
  card.className = `card ${unknown ? 'card--unknown' : 'card--low-confidence'}`;
  card.dataset.id = item.id;

  const body = document.createElement('div');
  body.className = 'card-body';

  // --- Header ---
  const header = document.createElement('div');
  header.className = 'card-header';

  const time = document.createElement('span');
  time.className = 'card-time';
  time.textContent = formatDateTime(item.recorded_at);

  const dur = document.createElement('span');
  dur.className = 'card-duration';
  dur.textContent = formatDuration(item.duration_ms);

  const label = document.createElement('span');
  label.className = `card-label ${unknown ? 'card-label--unknown' : 'card-label--low'}`;
  if (unknown) {
    label.textContent = 'Unknown speaker';
  } else {
    const pct = Math.round((item.match_confidence ?? 0) * 100);
    label.textContent = `${item.matched_speaker} · ${pct}% match`;
  }

  header.append(time, dur, label);

  // --- Transcript ---
  const transcript = document.createElement('div');
  transcript.className = 'card-transcript';
  transcript.textContent = `"${item.transcript}"`;

  // --- Audio row ---
  const audioRow = document.createElement('div');
  audioRow.className = 'audio-row';

  if (unknown) {
    const block = makeAudioBlock('Voice clip', item.audio_url, 'Play clip');
    audioRow.appendChild(block);
  } else {
    audioRow.appendChild(makeAudioBlock('This clip',                          item.audio_url,            'Play'));
    audioRow.appendChild(makeAudioBlock(`Reference: ${item.matched_speaker}`, item.reference_clip_url,   'Play reference'));
  }

  // --- Actions ---
  const actions = buildActionRow(item, unknown);

  body.append(header, transcript, audioRow, actions);
  card.appendChild(body);
  return card;
}

function makeAudioBlock(labelText, url, btnLabel) {
  const block = document.createElement('div');
  block.className = 'audio-block';

  const lbl = document.createElement('div');
  lbl.className = 'audio-label';
  lbl.textContent = labelText;

  block.append(lbl, makePlayBtn(btnLabel, url));
  return block;
}

function buildActionRow(item, unknown) {
  const row = document.createElement('div');
  row.className = 'card-actions';

  function lockRow()   { row.querySelectorAll('button').forEach(b => b.disabled = true); }
  function unlockRow() { row.querySelectorAll('button').forEach(b => b.disabled = false); }

  async function doResolve(action, assignedTo) {
    lockRow();
    try {
      await apiFetch(`/interactions/${item.id}/resolve`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action, assigned_to: assignedTo ?? null }),
      });
      dismissCard(item.id);
    } catch (err) {
      showToast('Error: ' + err.message);
      unlockRow();
    }
  }

  // Confirm (low-confidence only)
  if (!unknown) {
    const confirm = document.createElement('button');
    confirm.className = 'btn btn--confirm';
    confirm.textContent = `Confirm ${item.matched_speaker}`;
    confirm.addEventListener('click', () => doResolve('confirm', item.matched_speaker));
    row.appendChild(confirm);
  }

  // Reject
  const reject = document.createElement('button');
  reject.className = 'btn btn--reject';
  reject.textContent = 'Reject';
  reject.addEventListener('click', () => doResolve('reject', null));
  row.appendChild(reject);

  // Other match dropdown
  row.appendChild(buildOtherMatchDropdown(item, doResolve, row));

  return row;
}

function buildOtherMatchDropdown(item, doResolve, actionsRow) {
  const wrapper = document.createElement('div');
  wrapper.className = 'dropdown';

  const triggerBtn = document.createElement('button');
  triggerBtn.className = 'btn btn--ghost';
  triggerBtn.innerHTML = 'Other match <span style="font-size:.7em;margin-left:2px">▾</span>';
  triggerBtn.addEventListener('click', e => toggleDropdown(wrapper, e));

  const menu = document.createElement('div');
  menu.className = 'dropdown-menu';

  // Enrolled speakers, excluding the already-matched one
  const others = _speakersCache.filter(s => s.name !== item.matched_speaker);

  others.forEach(speaker => {
    const speakerBtn = document.createElement('button');
    speakerBtn.className = 'dropdown-item';
    speakerBtn.textContent = speaker.name;
    speakerBtn.addEventListener('click', e => {
      e.stopPropagation();
      menu.classList.remove('is-open');
      _openDropdown = null;
      doResolve('assign', speaker.name);
    });
    menu.appendChild(speakerBtn);
  });

  if (others.length > 0) {
    const divider = document.createElement('div');
    divider.className = 'dropdown-divider';
    menu.appendChild(divider);
  }

  const enrolBtn = document.createElement('button');
  enrolBtn.className = 'dropdown-item dropdown-item--enrol';
  enrolBtn.textContent = 'Enrol as new speaker…';
  enrolBtn.addEventListener('click', e => {
    e.stopPropagation();
    menu.classList.remove('is-open');
    _openDropdown = null;
    showEnrolInput(actionsRow, doResolve);
  });
  menu.appendChild(enrolBtn);

  wrapper.append(triggerBtn, menu);
  return wrapper;
}

function showEnrolInput(actionsRow, doResolve) {
  // Remove existing enrol row if any
  actionsRow.querySelector('.enrol-row')?.remove();

  const row = document.createElement('div');
  row.className = 'enrol-row';

  const input = document.createElement('input');
  input.className = 'enrol-input';
  input.type = 'text';
  input.placeholder = 'Speaker name';

  const submitBtn = document.createElement('button');
  submitBtn.className = 'btn btn--primary';
  submitBtn.textContent = 'Enrol';

  const cancelBtn = document.createElement('button');
  cancelBtn.className = 'btn btn--ghost';
  cancelBtn.textContent = 'Cancel';

  function submit() {
    const name = input.value.trim();
    if (!name) { input.focus(); return; }
    doResolve('enrol', name);
  }

  submitBtn.addEventListener('click', submit);
  input.addEventListener('keydown', e => { if (e.key === 'Enter') submit(); });
  cancelBtn.addEventListener('click', () => row.remove());

  row.append(input, submitBtn, cancelBtn);
  actionsRow.appendChild(row);
  setTimeout(() => input.focus(), 30);
}

function dismissCard(interactionId) {
  const card = document.querySelector(`.card[data-id="${interactionId}"]`);
  if (!card) return;

  card.style.transition = 'opacity .25s, max-height .3s, margin .3s, padding .3s';
  card.style.opacity = '0';
  card.style.maxHeight = card.offsetHeight + 'px';

  // Trigger reflow, then collapse
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      card.style.maxHeight = '0';
      card.style.marginBottom = '0';
      card.style.overflow = 'hidden';
    });
  });

  setTimeout(() => {
    card.remove();
    const list = document.getElementById('pending-list');
    const remaining = list.querySelectorAll('.card').length;
    document.getElementById('pending-count').textContent = remaining || '';
    if (remaining === 0) {
      list.innerHTML = '<div class="empty-state">No pending interactions — all clear.</div>';
    }
  }, 320);
}

/* ---------------------------------------------------------------------------
   Enrolled speakers tab
   --------------------------------------------------------------------------- */

async function loadSpeakers() {
  try {
    const speakers = await apiFetch('/speakers');
    const list = document.getElementById('speakers-list');
    list.innerHTML = '';

    if (speakers.length === 0) {
      list.innerHTML = '<div class="empty-state">No enrolled speakers yet.</div>';
      return;
    }

    speakers.forEach(s => list.appendChild(buildSpeakerCard(s)));
  } catch (err) {
    showToast('Failed to load speakers: ' + err.message);
  }
}

function buildSpeakerCard(s) {
  const card = document.createElement('div');
  card.className = 'speaker-card';

  const name = document.createElement('div');
  name.className = 'speaker-name';
  name.textContent = s.name;

  const stats = [
    ['Interactions',    s.interaction_count],
    ['Avg confidence',  s.avg_confidence != null ? Math.round(s.avg_confidence * 100) + '%' : '—'],
    ['Reference clips', s.reference_clip_count],
    ['Last active',     s.last_active_at ? formatDate(s.last_active_at) : 'Never'],
  ];

  card.appendChild(name);
  stats.forEach(([label, value]) => {
    const row = document.createElement('div');
    row.className = 'speaker-stat';
    row.innerHTML = `<span>${label}</span><strong>${value}</strong>`;
    card.appendChild(row);
  });

  return card;
}

/* ---------------------------------------------------------------------------
   History tab
   --------------------------------------------------------------------------- */

async function loadHistory() {
  const speaker  = document.getElementById('filter-speaker').value.trim();
  const fromDate = document.getElementById('filter-from').value;
  const toDate   = document.getElementById('filter-to').value;

  const params = new URLSearchParams();
  if (speaker)  params.set('speaker',   speaker);
  if (fromDate) params.set('date_from', fromDate + 'T00:00:00Z');
  if (toDate)   params.set('date_to',   toDate   + 'T23:59:59Z');

  const qs = params.toString() ? '?' + params.toString() : '';

  try {
    const items = await apiFetch('/interactions/history' + qs);
    const list = document.getElementById('history-list');
    list.innerHTML = '';

    if (items.length === 0) {
      list.innerHTML = '<div class="empty-state">No interactions found.</div>';
      return;
    }

    items.forEach(item => list.appendChild(buildHistoryRow(item)));
  } catch (err) {
    showToast('Failed to load history: ' + err.message);
  }
}

function buildHistoryRow(item) {
  const row = document.createElement('div');
  row.className = 'history-row';

  // ── Summary (always visible, click to expand) ──────────────────────
  const summary = document.createElement('div');
  summary.className = 'history-summary';

  const left = document.createElement('div');
  left.className = 'history-left';

  const transcript = document.createElement('div');
  transcript.className = 'history-transcript';
  transcript.textContent = `"${item.transcript}"`;

  const meta = document.createElement('div');
  meta.className = 'history-meta';

  const displaySpeaker = item.current_assigned_to || item.matched_speaker || 'Unknown';
  const conf = item.match_confidence != null
    ? ` (${Math.round(item.match_confidence * 100)}%)`
    : '';

  const speakerSpan = document.createElement('span');
  speakerSpan.className = 'history-speaker';
  speakerSpan.textContent = displaySpeaker + conf;

  meta.append(
    document.createTextNode(`${formatDateTime(item.recorded_at)} · ${formatDuration(item.duration_ms)} · `),
    speakerSpan,
  );
  left.append(transcript, meta);

  const right = document.createElement('div');
  right.className = 'history-right';

  const badge = document.createElement('span');
  badge.className = `status-badge status-badge--${item.status}`;
  badge.textContent = item.status;
  right.appendChild(badge);

  if (item.audio_url) {
    const playBtn = makePlayBtn('Play', item.audio_url);
    playBtn.classList.add('history-play');
    right.appendChild(playBtn);
  }

  if (item.status === 'resolved' && item.current_assigned_to) {
    const reconfirmBtn = document.createElement('button');
    reconfirmBtn.className = 'btn btn--ghost btn--reconfirm';
    reconfirmBtn.title = `Re-confirm as ${item.current_assigned_to} (adds clip to reference set)`;
    reconfirmBtn.textContent = '↻';
    reconfirmBtn.addEventListener('click', async e => {
      e.stopPropagation();
      reconfirmBtn.disabled = true;
      try {
        await apiFetch(`/interactions/${item.id}/resolve`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ action: 'confirm', assigned_to: item.current_assigned_to }),
        });
        showToast(`Re-confirmed as ${item.current_assigned_to} — clip added to reference set.`);
      } catch (err) {
        showToast('Re-confirm failed: ' + err.message);
      } finally {
        reconfirmBtn.disabled = false;
      }
    });
    right.appendChild(reconfirmBtn);
  }

  const chevron = document.createElement('span');
  chevron.className = 'history-chevron';
  chevron.textContent = '▼';
  right.appendChild(chevron);

  summary.append(left, right);

  // ── Trace panel (expandable) ────────────────────────────────────────
  const trace = document.createElement('div');
  trace.className = 'history-trace';

  // VAD confidence
  const vadPct = Math.round(item.vad_confidence * 100);
  trace.appendChild(traceItem('VAD confidence',
    `${vadPct}%`,
    vadPct >= 70 ? 'good' : 'warn',
  ));

  // Pipeline speaker match
  if (item.match_confidence != null) {
    const pct = Math.round(item.match_confidence * 100);
    const label = item.matched_speaker
      ? `${item.matched_speaker} — ${pct}%`
      : `No match (best: ${pct}%)`;
    trace.appendChild(traceItem('Pipeline match', label, item.matched_speaker ? 'good' : 'warn'));
  } else {
    trace.appendChild(traceItem('Pipeline match', 'No enrolled speakers', 'none'));
  }

  // Resolution
  if (item.current_action) {
    const by = item.current_assigned_to ? ` → ${item.current_assigned_to}` : '';
    trace.appendChild(traceItem('Resolution', `${item.current_action}${by}`));
  } else {
    trace.appendChild(traceItem('Resolution', 'Unresolved', 'none'));
  }

  // Audio duration
  trace.appendChild(traceItem('Audio duration', formatDuration(item.duration_ms)));

  // Pipeline timings
  if (item.timings) {
    const t = item.timings;
    trace.appendChild(traceItem('Total processing', `${t.total}ms`));
    trace.appendChild(traceItem('Noise suppressor', `${t.ns}ms`));
    trace.appendChild(traceItem('VAD', `${t.vad}ms`));
    trace.appendChild(traceItem('Speaker ID', `${t.sid}ms`));
    trace.appendChild(traceItem('STT', `${t.stt}ms`));
  }

  // Resolved at
  if (item.last_resolved_at) {
    trace.appendChild(traceItem('Resolved at', formatDateTime(item.last_resolved_at)));
  }

  // Toggle expand on summary click, but don't swallow play-button clicks
  summary.addEventListener('click', e => {
    if (e.target.closest('.btn-play')) return;
    row.classList.toggle('is-expanded');
  });

  row.append(summary, trace);
  return row;
}

function traceItem(label, value, modifier) {
  const el = document.createElement('div');
  el.className = 'trace-item';
  const l = document.createElement('div');
  l.className = 'trace-label';
  l.textContent = label;
  const v = document.createElement('div');
  v.className = 'trace-value' + (modifier ? ` trace-value--${modifier}` : '');
  v.textContent = value;
  el.append(l, v);
  return el;
}

/* ---------------------------------------------------------------------------
   Settings tab
   --------------------------------------------------------------------------- */

async function loadSettings() {
  try {
    const s = await apiFetch('/settings');
    document.getElementById('s-pending').value  = s.pending_retention_days;
    document.getElementById('s-resolved').value = s.resolved_retention_days;
    document.getElementById('s-clips').value    = s.reference_clips_per_speaker;
    updateSettingsWarnings(s.resolved_retention_days);
  } catch (err) {
    showToast('Failed to load settings: ' + err.message);
  }
}

function updateSettingsWarnings(resolvedDays) {
  document.getElementById('warn-resolved').hidden = Number(resolvedDays) !== 0;
}

document.getElementById('s-resolved').addEventListener('input', e => {
  updateSettingsWarnings(e.target.value);
});

document.getElementById('settings-form').addEventListener('submit', async e => {
  e.preventDefault();
  const btn = e.target.querySelector('[type=submit]');
  btn.disabled = true;
  try {
    const body = {
      pending_retention_days:    parseInt(document.getElementById('s-pending').value,  10),
      resolved_retention_days:   parseInt(document.getElementById('s-resolved').value, 10),
      reference_clips_per_speaker: parseInt(document.getElementById('s-clips').value,  10),
    };
    await apiFetch('/settings', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    updateSettingsWarnings(body.resolved_retention_days);
    const saved = document.getElementById('settings-saved');
    saved.hidden = false;
    setTimeout(() => { saved.hidden = true; }, 2000);
  } catch (err) {
    showToast('Failed to save settings: ' + err.message);
  } finally {
    btn.disabled = false;
  }
});

/* ---------------------------------------------------------------------------
   Formatting helpers
   --------------------------------------------------------------------------- */

function formatDateTime(iso) {
  return new Date(iso).toLocaleString(undefined, {
    month: 'short', day: 'numeric',
    hour: '2-digit', minute: '2-digit',
  });
}

function formatDate(iso) {
  return new Date(iso).toLocaleDateString(undefined, {
    month: 'short', day: 'numeric', year: 'numeric',
  });
}

function formatDuration(ms) {
  return ms < 1000 ? `${ms}ms` : `${(ms / 1000).toFixed(1)}s`;
}

/* ---------------------------------------------------------------------------
   Tab switching
   --------------------------------------------------------------------------- */

const tabLoaders = {
  pending:  loadPending,
  speakers: loadSpeakers,
  history:  loadHistory,
  settings: loadSettings,
};

document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const tab = btn.dataset.tab;

    document.querySelectorAll('.tab-btn')
      .forEach(b => b.classList.toggle('is-active', b === btn));
    document.querySelectorAll('.tab-panel')
      .forEach(p => p.classList.toggle('is-active', p.id === `tab-${tab}`));

    tabLoaders[tab]?.();
  });
});

/* ---------------------------------------------------------------------------
   Search / clear buttons (history)
   --------------------------------------------------------------------------- */

document.getElementById('btn-search').addEventListener('click', loadHistory);

document.getElementById('btn-clear').addEventListener('click', () => {
  document.getElementById('filter-speaker').value = '';
  document.getElementById('filter-from').value    = '';
  document.getElementById('filter-to').value      = '';
  loadHistory();
});

/* ---------------------------------------------------------------------------
   Boot
   --------------------------------------------------------------------------- */

initTheme();
loadPending();
