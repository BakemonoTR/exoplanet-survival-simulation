(() => {
    'use strict';

    let csrfToken = null;
    let latestStatus = null;
    let refreshTimer = null;
    const $ = (id) => document.getElementById(id);
    const esc = (value) => String(value ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#039;');
    const pretty = (value) => String(value || '—').replace(/_/g, ' ').replace(/-/g, ' ').replace(/\b\w/g, (letter) => letter.toUpperCase());
    const number = (value, digits = 0) => Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : '—';

    async function request(path, options = {}) {
        const method = options.method || 'GET';
        const headers = { Accept: 'application/json', ...(options.headers || {}) };
        if (options.body !== undefined) headers['Content-Type'] = 'application/json';
        if (!['GET', 'HEAD'].includes(method) && csrfToken) headers['X-CSRF-Token'] = csrfToken;
        const response = await fetch(path, {
            ...options, method, headers, credentials: 'same-origin',
            body: options.body === undefined ? undefined : JSON.stringify(options.body),
        });
        let payload = {};
        try { payload = await response.json(); } catch (_) {}
        if (!response.ok) {
            if (response.status === 401 && path !== '/api/admin/login') showLogin('Your session ended. Sign in again.');
            throw new Error(payload.detail || `Request failed (${response.status})`);
        }
        return payload;
    }

    function message(id, text, success = false) {
        const node = $(id);
        if (!node) return;
        node.textContent = text;
        node.classList.toggle('is-success', success);
    }

function showLogin(text = '') {
  clearInterval(refreshTimer);
  refreshTimer = null;
  csrfToken = null;
  $('adminApp').hidden = true;
  $('loginView').hidden = false;
  const passwordInput = $('adminPassword');
  const loginButton = $('loginForm').querySelector('button');
  passwordInput.value = '';
  passwordInput.disabled = false;
  loginButton.disabled = false;
  message('loginMessage', text);
}

    function showApp() {
        $('loginView').hidden = true;
        $('adminApp').hidden = false;
        if (!refreshTimer) refreshTimer = setInterval(refresh, 1500);
        refresh();
    }

    async function checkSession() {
        try {
  const session = await request('/api/admin/session');
  if (!session.configured) {
    showLogin('Admin password is not configured yet. Add ICARUS_ADMIN_PASSWORD to .env and restart the server.');
    $('adminPassword').focus();
    return;
  }
            if (session.authenticated) {
                csrfToken = session.csrf_token;
                showApp();
            } else showLogin();
        } catch (error) { showLogin(error.message); }
    }

    async function login(event) {
        event.preventDefault();
        const button = $('loginForm').querySelector('button');
        button.disabled = true;
        message('loginMessage', '');
        try {
            const payload = await request('/api/admin/login', { method: 'POST', body: { password: $('adminPassword').value } });
            csrfToken = payload.csrf_token;
            showApp();
        } catch (error) { message('loginMessage', error.message); }
        finally { button.disabled = false; }
    }

    async function logout() {
        try { await request('/api/admin/logout', { method: 'POST', body: {} }); } catch (_) {}
        showLogin();
    }

    function renderStatus(status) {
        latestStatus = status;
        const session = status.debug_session || {};
        const running = !!status.running;
        const paused = !!status.paused;
        const phase = session.phase || status.status || 'idle';
        $('liveDot').classList.toggle('running', running);
        $('topStatus').textContent = `${pretty(phase).toUpperCase()}${running ? ' · LIVE' : ''}`;
        $('phaseValue').textContent = pretty(phase).toUpperCase();
        $('runtimePill').textContent = paused ? 'PAUSED' : running ? 'RUNNING' : pretty(phase).toUpperCase();
        $('runtimePill').classList.toggle('ready', running && !paused);
        $('runValue').textContent = session.run_id ? `Run #${session.run_id}` : 'No active run';
        $('planetValue').textContent = pretty(session.planet || status.planet);
        $('attemptValue').textContent = `Attempt ${session.attempt || '—'} / ${session.max_attempts || '—'}`;
        $('tickValue').textContent = `${status.tick || 0} / ${status.max_ticks || session.max_ticks || 0}`;
        $('dayValue').textContent = `Day ${number((status.tick || 0) / 144, 1)}`;
        $('scoreValue').textContent = `${number(status.colony_score || 0, 1)}%`;
        $('crewValue').textContent = `${status.agents_alive || 0} crew alive`;
        document.querySelectorAll('[data-control]').forEach((button) => {
            const action = button.dataset.control;
            button.disabled = action === 'pause' ? (!running || paused) : action === 'resume' ? (!running || !paused) : (!running && !session.restart_pending);
        });
        $('restartButton').disabled = !session.run_id || !!session.restart_pending;
        renderPolicies(status.planets || [], status.planet_rl || {}, session.planet);
        renderChallenge(session.challenge);
        renderHistory(session.history || []);
        renderNarrative(status.narrative_status || status.llm || {});
    }

    function renderPolicies(planets, summaries, activePlanet) {
        const select = $('planetSelect');
        const prior = select.value;
        select.innerHTML = planets.map((planet) => `<option value="${esc(planet)}">${esc(pretty(planet))}</option>`).join('');
        if (planets.includes(prior)) select.value = prior;
        $('planetPolicies').innerHTML = planets.map((planet) => {
            const info = summaries[planet] || {};
            const count = Number(info.policy_snapshots || 0);
            return `<div class="policy-row"><div><strong>${esc(pretty(planet))}${planet === activePlanet ? ' · ACTIVE' : ''}</strong><small>${count} saved ${count === 1 ? 'policy' : 'policies'}${info.last_tick ? ` · last tick ${esc(info.last_tick)}` : ''}</small></div><button type="button" class="policy-reset" data-reset-planet="${esc(planet)}">Reset RL</button></div>`;
        }).join('') || '<p class="empty">No planet configuration found.</p>';
    }

    function renderChallenge(challenge) {
        const state = challenge?.state;
        if (!state) {
            $('challengeList').innerHTML = '<p class="empty">Campaign has not started.</p>';
            return;
        }
        const complete = new Set(state.completed_planets || []);
        $('challengeList').innerHTML = (state.planet_ids || []).map((planet) => `<div class="challenge-row ${complete.has(planet) ? 'is-complete' : ''}"><i></i><div><strong>${esc(pretty(planet))}</strong><small>${state.attempts_by_planet?.[planet] || 0} attempts · best ${number(state.best_score_by_planet?.[planet] || 0, 1)}%</small></div><span>${complete.has(planet) ? 'COMPLETED' : 'NOT COMPLETED'}</span></div>`).join('');
    }

    function renderHistory(history) {
        $('attemptHistory').innerHTML = history.length ? [...history].reverse().slice(0, 12).map((entry) => `<div class="history-row"><span>#${esc(entry.attempt)}</span><div><strong>${esc(pretty(entry.planet))}</strong><small>${esc(pretty(entry.end_reason))} · ${esc(entry.total_ticks || 0)} ticks${entry.run_id ? ` · run ${esc(entry.run_id)}` : ''}</small></div></div>`).join('') : '<p class="empty">No completed attempts.</p>';
    }

    function renderNarrative(narrative) {
        const status = narrative.status || 'unconfigured';
        $('narrativePill').textContent = status.toUpperCase();
        $('narrativePill').classList.toggle('ready', status === 'ready');
        $('providerValue').textContent = narrative.provider || 'local_gpt2';
        $('modelValue').textContent = narrative.model || 'gpt2';
        $('generationValue').textContent = narrative.total_calls || 0;
        $('pendingValue').textContent = narrative.pending || 0;
        $('narrativeReason').textContent = narrative.reason || 'Fine-tuned local GPT-2 provider is not connected.';
    }

    function renderState(state) {
        $('telemetryTick').textContent = `TICK ${state.tick ?? '—'}`;
        const agents = state.agents || [];
        $('agentRows').innerHTML = agents.length ? agents.map((agent) => {
            const policy = agent.rl_policy || {};
            const rewards = policy.reward_history || [];
            const last = rewards[rewards.length - 1];
            const reward = Number(last?.reward || 0);
            const action = agent.action?.action_type || agent.last_decision?.action || 'idle';
            const position = agent.position || {};
            return `<tr><td><strong>${esc(agent.name || agent.id)}</strong></td><td>${esc(pretty(agent.status))}</td><td>${esc(position.x ?? '—')}, ${esc(position.y ?? '—')}</td><td>${esc(pretty(action))}</td><td>${number(policy.total_reward || 0, 1)}</td><td class="${reward >= 0 ? 'reward-positive' : 'reward-negative'}">${last ? `${reward >= 0 ? '+' : ''}${number(reward, 2)} · ${esc(last.reason || 'update')}` : '—'}</td></tr>`;
        }).join('') : '<tr><td colspan="6" class="empty">No telemetry yet.</td></tr>';
        const details = document.querySelector('.telemetry-card details');
        if (details?.open) $('rawState').textContent = JSON.stringify(state, null, 2);
    }

    async function refresh() {
        try {
            const status = await request('/api/admin/status');
            renderStatus(status);
            if (status.debug_session?.run_id) {
                try { renderState(await request('/api/simulation/state')); } catch (_) {}
            }
        } catch (error) {
            if (!String(error.message).includes('session ended')) message('runtimeMessage', error.message);
        }
    }

    async function control(action) {
        message('runtimeMessage', 'Applying command…');
        try {
            const result = await request('/api/admin/control', { method: 'POST', body: { action } });
            message('runtimeMessage', `Command accepted: ${pretty(result.status)}.`, true);
            await refresh();
        } catch (error) { message('runtimeMessage', error.message); }
    }

    async function applySpeed() {
        const multiplier = Number($('liveSpeed').value);
        try {
            await request('/api/admin/control', { method: 'POST', body: { action: 'speed', value: multiplier } });
            message('runtimeMessage', `Simulation speed set to ${multiplier}×.`, true);
            await refresh();
        } catch (error) { message('runtimeMessage', error.message); }
    }

    function launchPayload(challengeMode) {
        const multiplier = Number($('launchSpeed').value || 1);
        return {
            planet: $('planetSelect').value,
            seed: Number($('seedInput').value || 42),
            max_ticks: Number($('maxTicksInput').value || 51840),
            tick_speed: 4 / Math.max(.1, multiplier),
            challenge_mode: challengeMode,
            dialogue_generation_enabled: $('dialogueEnabled').checked,
        };
    }

    async function launch(challengeMode) {
        message('launchMessage', challengeMode ? 'Starting campaign dispatch…' : 'Starting selected planet…');
        try {
            const result = await request('/api/admin/start', { method: 'POST', body: launchPayload(challengeMode) });
            message('launchMessage', challengeMode ? `Wheel dispatch queued for ${pretty(result.planet)}.` : `Simulation started on ${pretty(result.planet)}.`, true);
            await refresh();
        } catch (error) { message('launchMessage', error.message); }
    }

    async function restart() {
        message('runtimeMessage', 'Stopping attempt and rolling back this run’s learned policy…');
        try {
            const result = await request('/api/admin/restart', { method: 'POST', body: { discard_current_rl: true } });
            message('runtimeMessage', `Restarted ${pretty(result.planet)} from its previous RL history.`, true);
            await refresh();
        } catch (error) { message('runtimeMessage', error.message); }
    }

    async function resetPlanet(planet) {
        if (!window.confirm(`Reset all saved RL policies for ${pretty(planet)}? Run history will remain.`)) return;
        try {
            const result = await request(`/api/admin/planets/${encodeURIComponent(planet)}/reset-rl`, { method: 'POST', body: {} });
            message('runtimeMessage', `${pretty(planet)} RL reset: ${result.policies_deleted || 0} policies removed.`, true);
            await refresh();
        } catch (error) { message('runtimeMessage', error.message); }
    }

    function initialize() {
        $('loginForm').addEventListener('submit', login);
        $('logoutButton').addEventListener('click', logout);
        document.querySelectorAll('[data-control]').forEach((button) => button.addEventListener('click', () => control(button.dataset.control)));
        $('liveSpeed').addEventListener('input', () => { $('liveSpeedLabel').textContent = `${number($('liveSpeed').value, 1).replace('.0', '')}×`; });
        $('applySpeed').addEventListener('click', applySpeed);
        $('campaignButton').addEventListener('click', () => launch(true));
        $('manualButton').addEventListener('click', () => launch(false));
        $('restartButton').addEventListener('click', restart);
        $('planetPolicies').addEventListener('click', (event) => {
            const button = event.target.closest('[data-reset-planet]');
            if (button) resetPlanet(button.dataset.resetPlanet);
        });
        document.querySelector('.telemetry-card details').addEventListener('toggle', refresh);
        setInterval(() => { $('systemClock').textContent = new Date().toLocaleString(); }, 1000);
        $('systemClock').textContent = new Date().toLocaleString();
        checkSession();
    }

    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', initialize);
    else initialize();
})();
