(() => {
    'use strict';

    const state = {
        ticks: [],
        selectedTick: null,
        selectedAgent: null,
        followLive: true,
        events: new Map(),
        rewards: new Map(),
        rewardKeys: new Set(),
        activeRunId: null,
        mounted: false,
    };

    const esc = (value) => String(value ?? '')
        .replace(/&/g, '&amp;').replace(/</g, '&lt;')
        .replace(/>/g, '&gt;').replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
    const title = (value) => String(value || 'unknown')
        .replace(/_/g, ' ').replace(/\b\w/g, (letter) => letter.toUpperCase());
    const num = (value, digits = 1) => Number.isFinite(Number(value))
        ? Number(value).toFixed(digits) : '—';
    const signed = (value) => `${Number(value) >= 0 ? '+' : ''}${num(value, 2)}`;

    function mount() {
        if (state.mounted) return;
        const root = document.getElementById('agentInspectorMount');
        if (!root) return;
        root.innerHTML = `
            <section class="agent-workspace" aria-label="Tick-level agent inspector">
                <header class="agent-workspace__head">
                    <div><h2>Agent Inspector</h2><p>Actions, movement, reinforcement learning and social telemetry at every captured tick.</p></div>
                    <nav class="tick-nav" aria-label="Telemetry tick">
                        <button type="button" data-tick-step="-1" aria-label="Previous tick">←</button>
                        <output id="agentTickOutput">NO DATA</output>
                        <button type="button" data-tick-step="1" aria-label="Next tick">→</button>
                    </nav>
                </header>
                <div class="agent-workspace__body">
                    <aside class="agent-directory" id="agentDirectory"></aside>
                    <div class="agent-detail" id="agentDetail"><div class="workspace-empty">Waiting for simulation telemetry.</div></div>
                </div>
            </section>`;
        root.addEventListener('click', (event) => {
            const agentButton = event.target.closest('[data-agent-id]');
            if (agentButton) {
                state.selectedAgent = agentButton.dataset.agentId;
                render();
                return;
            }
            const stepButton = event.target.closest('[data-tick-step]');
            if (stepButton) stepTick(Number(stepButton.dataset.tickStep));
        });
        state.mounted = true;
        render();
    }

    function snapshotForTick(tick) {
        return state.ticks.find((item) => Number(item.tick) === Number(tick));
    }

    function ingest(data, shouldRender = true) {
        if (!data || data.tick === undefined || data.tick === null) return;
        const tick = Number(data.tick);
        const runId = data.run_id ?? data.debug_session?.run_id ?? null;
        if (runId !== null && state.activeRunId !== null && String(runId) !== String(state.activeRunId)) {
            state.ticks.length = 0;
            state.events.clear();
            state.rewards.clear();
            state.rewardKeys.clear();
            state.selectedTick = null;
            state.followLive = true;
        }
        if (runId !== null) state.activeRunId = runId;
        const agents = (Array.isArray(data.agents) ? data.agents : []).map((agent) => {
            const policy = agent.rl_policy || {};
            const history = Array.isArray(policy.reward_history) ? policy.reward_history : [];
            for (const reward of history) {
                const key = `${agent.id}|${reward.tick}|${reward.state_key}|${reward.action}|${reward.reward}|${reward.reason}|${reward.q_after}`;
                if (state.rewardKeys.has(key)) continue;
                state.rewardKeys.add(key);
                const ledger = state.rewards.get(String(agent.id)) || [];
                ledger.push(reward);
                ledger.sort((a, b) => Number(a.tick || 0) - Number(b.tick || 0));
                if (ledger.length > 500) ledger.splice(0, ledger.length - 500);
                state.rewards.set(String(agent.id), ledger);
            }
            return {
                ...agent,
                rl_policy: {
                    ...policy,
                    reward_history: history.filter((reward) => Number(reward.tick) === tick),
                },
            };
        });
        const next = {
            tick,
            run_id: runId,
            agents,
            colony: data.colony || data.colony_score || {},
            narrative_status: data.narrative_status || {},
        };
        const existing = state.ticks.findIndex((item) => item.tick === tick);
        if (existing >= 0) state.ticks[existing] = next;
        else state.ticks.push(next);
        state.ticks.sort((a, b) => a.tick - b.tick);
        if (state.ticks.length > 1000) state.ticks.splice(0, state.ticks.length - 1000);
        if (state.followLive || state.selectedTick === null) state.selectedTick = tick;
        if (!state.selectedAgent && next.agents[0]) state.selectedAgent = String(next.agents[0].id);
        if (shouldRender) render();
    }

    function recordEvent(event, shouldRender = true) {
        if (!event) return;
        const tick = Number(event.tick || 0);
        const rows = state.events.get(tick) || [];
        rows.push(event);
        if (rows.length > 100) rows.shift();
        state.events.set(tick, rows);
        if (state.events.size > 1000) state.events.delete(Math.min(...state.events.keys()));
        if (shouldRender && Number(state.selectedTick) === tick) render();
    }

    function stepTick(direction) {
        if (!state.ticks.length) return;
        let index = state.ticks.findIndex((item) => item.tick === state.selectedTick);
        if (index < 0) index = state.ticks.length - 1;
        index = Math.max(0, Math.min(state.ticks.length - 1, index + direction));
        state.selectedTick = state.ticks[index].tick;
        state.followLive = index === state.ticks.length - 1;
        render();
    }

    function actionLabel(agent) {
        const action = agent.action || {};
        return action.action_type || action.type || agent.last_decision?.action || 'idle';
    }

    function movement(agent, snapshot) {
        const index = state.ticks.findIndex((item) => item.tick === snapshot.tick);
        const previous = index > 0 ? state.ticks[index - 1] : null;
        const old = previous?.agents?.find((item) => String(item.id) === String(agent.id));
        const position = agent.position || { x: agent.pos?.[0], y: agent.pos?.[1] };
        const prior = old?.position || { x: old?.pos?.[0], y: old?.pos?.[1] };
        const moved = old && (Number(position.x) !== Number(prior.x) || Number(position.y) !== Number(prior.y));
        return moved
            ? `Moved ${prior.x ?? '—'}, ${prior.y ?? '—'} → ${position.x ?? '—'}, ${position.y ?? '—'}`
            : `Held at ${position.x ?? '—'}, ${position.y ?? '—'}`;
    }

    function renderDirectory(snapshot) {
        const directory = document.getElementById('agentDirectory');
        if (!directory) return;
        const agents = snapshot?.agents || [];
        directory.innerHTML = `<div class="agent-directory__label">CREW · ${agents.length}</div>${agents.map((agent) => {
            const id = String(agent.id);
            const dead = String(agent.status).toLowerCase() === 'dead';
            return `<button type="button" data-agent-id="${esc(id)}" class="${id === state.selectedAgent ? 'is-active' : ''}">
                <i class="agent-dot ${dead ? 'is-dead' : ''}"></i>
                <span class="agent-name">${esc(agent.name || id)}</span>
                <span class="agent-state">${esc(agent.status || 'unknown')}</span>
            </button>`;
        }).join('')}`;
    }

    function renderVitals(needs) {
        const labels = { hunger: 'Hunger', thirst: 'Thirst', energy: 'Energy', hygiene: 'Hygiene', o2_supply: 'O₂', temperature_stress: 'Thermal' };
        const rows = Object.entries(needs || {}).filter(([key]) => labels[key]).map(([key, value]) =>
            `<div class="vital"><span>${labels[key]}</span><strong>${num(value, 0)}%</strong></div>`
        );
        return rows.length ? rows.join('') : '<div class="empty-data">No vitals at this tick.</div>';
    }

    function renderRewards(agentId, snapshot) {
        const history = (state.rewards.get(String(agentId)) || [])
            .filter((reward) => Number(reward.tick || 0) <= Number(snapshot.tick))
            .slice(-14).reverse();
        if (!history.length) return '<div class="empty-data">No reward transition recorded yet.</div>';
        return history.map((entry) => {
            const reward = Number(entry.reward || 0);
            const components = entry.components && typeof entry.components === 'object'
                ? Object.entries(entry.components).map(([key, value]) => `${title(key)} ${signed(value)}`).join(' · ')
                : '';
            return `<div class="reward-row" title="${esc(components)}">
                <span class="row-mono">T${esc(entry.tick ?? '—')}</span>
                <span class="row-copy">${esc(entry.reason || entry.source || 'Policy update')}</span>
                <strong class="${reward >= 0 ? 'is-gain' : 'is-loss'}">${signed(reward)}</strong>
            </div>`;
        }).join('');
    }

    function renderQValues(policy) {
        const values = Object.entries(policy.current_q_values || {}).sort((a, b) => Number(b[1]) - Number(a[1])).slice(0, 10);
        if (!values.length) return '<div class="empty-data">This state has no learned action values yet.</div>';
        return values.map(([action, value]) => `<div class="q-row"><span class="row-mono">Q</span><span class="row-copy">${esc(title(action))}</span><strong>${signed(value)}</strong></div>`).join('');
    }

    function renderRelations(agent) {
        const rows = Array.isArray(agent.relationships) ? agent.relationships : [];
        if (!rows.length) return '<div class="empty-data">No social relationship telemetry yet.</div>';
        return rows.map((row) => `<div class="relation-row">
            <span class="row-mono">${signed(row.trust || 0)}</span>
            <span class="row-copy">${esc(row.name || row.agent_id)}</span>
            <strong>${esc(row.opinion || 'No narrative')}</strong>
        </div>`).join('');
    }

    function renderThoughts(agent, snapshot) {
        const direct = Array.isArray(agent.thoughts) ? agent.thoughts : [];
        const related = (state.events.get(snapshot.tick) || []).filter((event) =>
            String(event.agent || '') === String(agent.name || '') &&
            ['llm_thought', 'llm_dialogue', 'llm_reflection', 'communicate'].includes(event.type)
        );
        const rows = [...direct, ...related];
        if (!rows.length) return '<div class="empty-data">Local GPT-2 is not connected. Generated thoughts and opinions will appear here when the provider is ready.</div>';
        return rows.slice(-12).reverse().map((row) => `<div class="thought-row">
            <span class="row-mono">${esc(title(row.type || 'thought'))}</span>
            <span class="row-copy">${esc(row.dialogue || row.reasoning || row.insight || row.text || '')}</span>
            <strong>${esc(row.target_agent || '')}</strong>
        </div>`).join('');
    }

    function renderTickEvents(snapshot) {
        const rows = state.events.get(snapshot.tick) || [];
        if (!rows.length) return '<div class="empty-data">No event emitted at this tick.</div>';
        return rows.slice(-16).reverse().map((event) => `<div class="tick-event-row">
            <span class="row-mono">${esc(title(event.type))}</span>
            <span class="row-copy">${esc(event.dialogue || event.reasoning || event.cause || event.description || event.goal || event.type)}</span>
            <strong>${esc(event.agent || '')}</strong>
        </div>`).join('');
    }

    function renderDetail(snapshot) {
        const detail = document.getElementById('agentDetail');
        if (!detail) return;
        if (!snapshot?.agents?.length) {
            detail.innerHTML = '<div class="workspace-empty">Waiting for simulation telemetry.</div>';
            return;
        }
        let agent = snapshot.agents.find((item) => String(item.id) === state.selectedAgent);
        if (!agent) {
            agent = snapshot.agents[0];
            state.selectedAgent = String(agent.id);
        }
        const policy = agent.rl_policy || {};
        const decision = agent.last_decision || {};
        const position = agent.position || { x: agent.pos?.[0], y: agent.pos?.[1] };
        detail.innerHTML = `
            <div class="agent-detail__top">
                <div><h3>${esc(agent.name || agent.id)}</h3><div class="agent-detail__meta">${esc(title(agent.status))} · ${esc(movement(agent, snapshot))} · ID ${esc(agent.id)}</div></div>
                <div class="action-chip">${esc(title(actionLabel(agent)))}</div>
            </div>
            <div class="agent-grid">
                <section class="data-card"><h4>Vitals at tick ${esc(snapshot.tick)}</h4><div class="vital-list">${renderVitals(agent.needs)}</div></section>
                <section class="data-card"><h4>Current decision</h4>
                    <div class="rl-summary"><div>Position<strong>${esc(position.x ?? '—')}, ${esc(position.y ?? '—')}</strong></div><div>State<strong>${esc(policy.state_key || 'nominal')}</strong></div><div>Source<strong>${esc(decision.source || 'simulation')}</strong></div></div>
                    <div class="empty-data" style="text-align:left;padding:8px 0 0">${esc(decision.reasoning || `Executing ${title(actionLabel(agent))}.`)}</div>
                </section>
                <section class="data-card"><h4>RL policy</h4>
                    <div class="rl-summary"><div>Total reward<strong>${signed(policy.total_reward || agent.rl_reward || 0)}</strong></div><div>Learned states<strong>${esc(policy.learned_states ?? agent.q_states ?? 0)}</strong></div><div>Explore ε<strong>${num(policy.epsilon, 3)}</strong></div></div>
                    <div class="q-list">${renderQValues(policy)}</div>
                </section>
                <section class="data-card"><h4>Reward gains & losses</h4><div class="reward-list">${renderRewards(agent.id, snapshot)}</div></section>
                <section class="data-card"><h4>Relationships & opinions</h4><div class="relation-list">${renderRelations(agent)}</div></section>
                <section class="data-card"><h4>Thoughts & conversations</h4><div class="thought-list">${renderThoughts(agent, snapshot)}</div></section>
                <section class="data-card data-card--wide"><h4>Tick ${esc(snapshot.tick)} event journal</h4><div class="tick-event-list">${renderTickEvents(snapshot)}</div></section>
            </div>`;
    }

    function render() {
        mount();
        const snapshot = snapshotForTick(state.selectedTick) || state.ticks[state.ticks.length - 1];
        const output = document.getElementById('agentTickOutput');
        if (output) output.textContent = snapshot ? `TICK ${snapshot.tick}${state.followLive ? ' · LIVE' : ''}` : 'NO DATA';
        renderDirectory(snapshot);
        renderDetail(snapshot);
    }

    async function hydrate() {
        try {
            const response = await fetch('/api/simulation/telemetry?limit=200&latest=true', { headers: { Accept: 'application/json' } });
            if (!response.ok) return;
            const payload = await response.json();
            if (state.activeRunId !== null && payload.run_id !== null && String(payload.run_id) !== String(state.activeRunId)) return;
            for (const row of payload.ticks || []) {
                ingest(row, false);
                for (const event of row.events || []) recordEvent(event, false);
            }
            render();
        } catch (_) {
            // Live WebSocket updates remain the primary source.
        }
    }

    window.AnalysisWorkspace = { update: ingest, event: recordEvent, render };
    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', () => { mount(); hydrate(); });
    else { mount(); hydrate(); }
})();
