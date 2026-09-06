(() => {
    'use strict';

    const fallbackPlanets = [
        'kepler-442b', 'proxima-centauri-b', 'ross-128b',
        'teegardens-star-b', 'trappist-1e',
    ];
    let lastSpin = null;
    let spinTimer = null;
    let hideTimer = null;
    let latestChallenge = null;

    const pretty = (value) => String(value || 'Unknown world')
        .split('-').map((part) => /\d/.test(part) ? part.toUpperCase() : part.charAt(0).toUpperCase() + part.slice(1))
        .join(' ')
        .replace(/Trappist/i, 'TRAPPIST')
        .replace(/Kepler/i, 'Kepler');

    function challengeState(session) {
        const challenge = session?.challenge || latestChallenge;
        return challenge?.state || {};
    }

    function renderRoster(session) {
        const roster = document.getElementById('planetDispatchRoster');
        if (!roster) return;
        const challenge = challengeState(session);
        const planets = challenge.planet_ids?.length ? challenge.planet_ids : fallbackPlanets;
        const completed = new Set(challenge.completed_planets || []);
        roster.replaceChildren(...planets.map((planet) => {
            const item = document.createElement('span');
            item.className = `planet-dispatch__roster-item${completed.has(planet) ? ' is-complete' : ''}`;
            item.textContent = `${pretty(planet)} · ${completed.has(planet) ? 'COMPLETED' : 'NOT COMPLETED'}`;
            return item;
        }));
    }

    function setName(planet, completed) {
        const name = document.getElementById('planetDispatchName');
        const status = document.getElementById('planetDispatchStatus');
        if (!name || !status) return;
        name.classList.remove('is-changing');
        void name.offsetWidth;
        name.textContent = pretty(planet);
        name.classList.add('is-changing');
        status.textContent = completed ? 'COMPLETED' : 'NOT COMPLETED';
        status.classList.toggle('is-complete', completed);
    }

    function beginSpin(session) {
        const overlay = document.getElementById('planetDispatch');
        if (!overlay) return;
        clearInterval(spinTimer);
        clearTimeout(hideTimer);
        const challenge = challengeState(session);
        const planets = challenge.planet_ids?.length ? challenge.planet_ids : fallbackPlanets;
        const completed = new Set(challenge.completed_planets || []);
        const selected = session.wheel?.selected_planet || session.planet || planets[0];
        const duration = Math.max(.6, Number(session.wheel?.remaining_seconds || session.wheel?.duration_seconds || 4));
        overlay.style.setProperty('--dispatch-duration', `${duration}s`);
        const progress = document.getElementById('planetDispatchProgress');
        if (progress) {
            progress.style.animation = 'none';
            void progress.offsetWidth;
            progress.style.animation = '';
        }
        overlay.classList.add('is-visible');
        overlay.setAttribute('aria-hidden', 'false');
        document.body.classList.add('is-dispatching');
        renderRoster(session);

        let index = 0;
        setName(planets[index], completed.has(planets[index]));
        spinTimer = setInterval(() => {
            index = (index + 1) % planets.length;
            setName(planets[index], completed.has(planets[index]));
        }, 140);
        window.setTimeout(() => {
            clearInterval(spinTimer);
            setName(selected, completed.has(selected));
        }, Math.max(120, duration * 1000 - 520));
    }

    function endSpin() {
        const overlay = document.getElementById('planetDispatch');
        if (!overlay || !overlay.classList.contains('is-visible')) return;
        clearInterval(spinTimer);
        hideTimer = window.setTimeout(() => {
            overlay.classList.remove('is-visible');
            overlay.setAttribute('aria-hidden', 'true');
            document.body.classList.remove('is-dispatching');
        }, 180);
    }

    function update(session) {
        if (!session || typeof session !== 'object') return;
        if (session.challenge) latestChallenge = session.challenge;
        renderRoster(session);
        const active = session.phase === 'wheel' || session.wheel?.active;
        const spin = String(session.wheel?.spin_id ?? session.spin_id ?? `${session.planet}:${session.restart_at}`);
        if (active && spin !== lastSpin) {
            lastSpin = spin;
            beginSpin(session);
        } else if (!active) {
            endSpin();
        }
    }

    async function initialize() {
        try {
            const challengeResponse = await fetch('/api/challenge/status', { headers: { Accept: 'application/json' } });
            if (challengeResponse.ok) latestChallenge = await challengeResponse.json();
        } catch (_) {}
        try {
            const statusResponse = await fetch('/api/simulation/status', { headers: { Accept: 'application/json' } });
            if (statusResponse.ok) {
                const status = await statusResponse.json();
                update(status.debug_session || {});
            } else renderRoster({});
        } catch (_) { renderRoster({}); }
    }

    window.PlanetDispatch = { update };
    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', initialize);
    else initialize();
})();
