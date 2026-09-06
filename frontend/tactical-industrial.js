(function () {
    'use strict';

    const palette = {
        text: '#f1f5f9',
        muted: '#94a3b8',
        steel: '#64748b',
        steelLight: '#cbd5e1',
        regolith: '#8b5cf6',
        regolithDark: '#312e81',
        safety: '#818cf8',
        water: '#06b6d4',
        oxygen: '#38bdf8',
        agriculture: '#10b981',
        danger: '#f43f5e',
    };

    // Keep index.html's original, brighter map palette. The theme now owns
    // silhouettes only and deliberately publishes no map-colour overrides.
    const mapPalette = {};

    const structureScales = {
        potable_water_tank: 0.42,
        oxygen_buffer_tank: 0.42,
        life_support_distribution_grid: 0.40,
        power_distribution_grid: 0.46,
        water_collector: 0.56,
        water_purifier: 0.54,
        isru_o2_unit: 0.58,
        isru_unit: 0.58,
        stone_furnace: 0.58,
        forge: 0.58,
        cnc_fabricator: 0.60,
        storage_crate: 0.48,
        communications_array: 0.58,
        medical_station: 0.62,
        radiation_shelter: 0.70,
        habitat_module: 0.72,
        basic_shelter: 0.68,
        greenhouse: 0.72,
        hydroponics: 0.72,
        solar_panel: 0.82,
        solar_array: 0.82,
        landing_zone: 0.78,
    };

    const labels = {
        eclss_lander_hub: 'LANDER HUB',
        habitat_module: 'HABITAT',
        basic_shelter: 'SHELTER',
        radiation_shelter: 'STORM SHELTER',
        medical_station: 'MEDICAL',
        greenhouse: 'GREENHOUSE',
        hydroponics: 'HYDROPONICS',
        solar_panel: 'PV ARRAY',
        solar_array: 'PV ARRAY',
        life_support_distribution_grid: 'UTILITY VAULT',
        power_distribution_grid: 'PMAD',
        water_collector: 'WATER EXT',
        water_purifier: 'WATER PROC',
        potable_water_tank: 'H₂O TANK',
        oxygen_buffer_tank: 'O₂ BUFFER',
        isru_o2_unit: 'O₂ ISRU',
        isru_unit: 'ISRU',
        communications_array: 'COMMS',
        stone_furnace: 'FURNACE',
        forge: 'FORGE',
        cnc_fabricator: 'CNC CELL',
        storage_crate: 'STORES',
        landing_zone: 'LANDING PAD',
    };

    function roundedRect(ctx, x, y, width, height, radius) {
        ctx.beginPath();
        ctx.roundRect(x, y, width, height, Math.max(1, radius));
    }

    function structureScale(type) {
        return structureScales[type] || 0.58;
    }

    function compactLabel(ctx, text, x, y, mapZoom, color = palette.text) {
        const fontSize = Math.max(6, Math.min(8, 7 * mapZoom));
        ctx.font = `700 ${fontSize}px JetBrains Mono, monospace`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        const width = ctx.measureText(text).width + 8;
        const height = fontSize + 5;
        ctx.fillStyle = 'rgba(20, 20, 17, 0.88)';
        roundedRect(ctx, x - width / 2, y - height / 2, width, height, 2);
        ctx.fill();
        ctx.strokeStyle = 'rgba(183, 170, 144, 0.32)';
        ctx.lineWidth = 1;
        ctx.stroke();
        ctx.fillStyle = color;
        ctx.fillText(text, x, y + 0.5);
    }

    function drawGroundPad(ctx, cx, cy, width, height) {
        ctx.fillStyle = 'rgba(18, 18, 15, 0.48)';
        ctx.beginPath();
        ctx.ellipse(cx, cy + height * 0.35, width * 0.62, height * 0.25, 0, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = 'rgba(157, 143, 116, 0.30)';
        ctx.lineWidth = 1;
        ctx.stroke();
    }

    function drawPressureModule(ctx, cx, cy, bw, bh, accent, medical = false) {
        const bodyW = bw * 0.82;
        const bodyH = bh * 0.48;
        ctx.fillStyle = '#5f6565';
        ctx.strokeStyle = '#bdc0b9';
        ctx.lineWidth = 1.2;
        roundedRect(ctx, cx - bodyW / 2, cy - bodyH / 2, bodyW, bodyH, bodyH / 2);
        ctx.fill();
        ctx.stroke();
        ctx.strokeStyle = 'rgba(31, 32, 29, 0.85)';
        ctx.lineWidth = 1;
        [-0.22, 0.22].forEach(offset => {
            ctx.beginPath();
            ctx.moveTo(cx + bodyW * offset, cy - bodyH / 2);
            ctx.lineTo(cx + bodyW * offset, cy + bodyH / 2);
            ctx.stroke();
        });
        ctx.fillStyle = accent;
        ctx.fillRect(cx - bodyW * 0.36, cy - bodyH * 0.39, bodyW * 0.72, Math.max(2, bodyH * 0.10));
        ctx.fillStyle = '#292b29';
        ctx.strokeStyle = accent;
        ctx.fillRect(cx + bodyW * 0.30, cy - bodyH * 0.18, bodyW * 0.18, bodyH * 0.36);
        ctx.strokeRect(cx + bodyW * 0.30, cy - bodyH * 0.18, bodyW * 0.18, bodyH * 0.36);
        if (medical) {
            const cross = Math.max(2, Math.min(bodyW, bodyH) * 0.13);
            ctx.fillStyle = '#d8d3c6';
            ctx.fillRect(cx - cross * 1.5, cy - cross / 2, cross * 3, cross);
            ctx.fillRect(cx - cross / 2, cy - cross * 1.5, cross, cross * 3);
        }
    }

    function drawTank(ctx, cx, cy, bw, bh, accent) {
        const tankW = bw * 0.54;
        const tankH = bh * 0.78;
        const x = cx - tankW / 2;
        const y = cy - tankH / 2;
        const gradient = ctx.createLinearGradient(x, y, x + tankW, y);
        gradient.addColorStop(0, '#3a3e3d');
        gradient.addColorStop(0.46, '#9da29e');
        gradient.addColorStop(1, '#4c5150');
        ctx.fillStyle = gradient;
        ctx.strokeStyle = '#c0beb5';
        ctx.lineWidth = 1;
        roundedRect(ctx, x, y, tankW, tankH, tankW / 2);
        ctx.fill();
        ctx.stroke();
        ctx.fillStyle = accent;
        ctx.fillRect(x, cy - Math.max(1.5, tankH * 0.07), tankW, Math.max(3, tankH * 0.14));
        ctx.strokeStyle = '#2b2c2a';
        ctx.beginPath();
        ctx.moveTo(cx - tankW * 0.28, y + tankH);
        ctx.lineTo(cx - tankW * 0.36, y + tankH * 1.10);
        ctx.moveTo(cx + tankW * 0.28, y + tankH);
        ctx.lineTo(cx + tankW * 0.36, y + tankH * 1.10);
        ctx.stroke();
    }

    function drawIndustrialSkid(ctx, cx, cy, bw, bh, accent, type) {
        const x = cx - bw * 0.43;
        const y = cy - bh * 0.28;
        const width = bw * 0.86;
        const height = bh * 0.55;
        ctx.fillStyle = '#343735';
        ctx.strokeStyle = '#8e928c';
        ctx.lineWidth = 1;
        roundedRect(ctx, x, y, width, height, 2);
        ctx.fill();
        ctx.stroke();
        ctx.fillStyle = '#686d69';
        ctx.fillRect(x + width * 0.08, y + height * 0.18, width * 0.31, height * 0.64);
        ctx.fillStyle = accent;
        ctx.fillRect(x + width * 0.48, y + height * 0.20, width * 0.40, height * 0.13);
        ctx.strokeStyle = accent;
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        ctx.arc(x + width * 0.67, y + height * 0.62, height * 0.17, 0, Math.PI * 2);
        ctx.stroke();
        if (type === 'stone_furnace' || type === 'forge') {
            ctx.fillStyle = '#5b2f1f';
            ctx.fillRect(x + width * 0.49, y + height * 0.43, width * 0.34, height * 0.33);
            ctx.fillStyle = '#c96f37';
            ctx.fillRect(x + width * 0.57, y + height * 0.52, width * 0.18, height * 0.14);
        }
    }

    function drawStructure({ ctx, struct, type, cx, cy, bw, bh, mapZoom, health }) {
        ctx.save();
        ctx.lineJoin = 'round';
        ctx.lineCap = 'round';
        drawGroundPad(ctx, cx, cy, bw, bh);

        if (type === 'habitat_module' || type === 'basic_shelter' || type === 'medical_station') {
            drawPressureModule(ctx, cx, cy, bw, bh, type === 'medical_station' ? palette.danger : palette.safety, type === 'medical_station');
        } else if (type === 'radiation_shelter') {
            ctx.fillStyle = palette.regolithDark;
            ctx.strokeStyle = palette.regolith;
            ctx.lineWidth = 1.2;
            ctx.beginPath();
            ctx.ellipse(cx, cy + bh * 0.10, bw * 0.44, bh * 0.35, 0, Math.PI, Math.PI * 2);
            ctx.lineTo(cx + bw * 0.44, cy + bh * 0.22);
            ctx.lineTo(cx - bw * 0.44, cy + bh * 0.22);
            ctx.closePath();
            ctx.fill();
            ctx.stroke();
            ctx.fillStyle = '#2d302e';
            ctx.strokeStyle = palette.safety;
            ctx.fillRect(cx - bw * 0.11, cy - bh * 0.05, bw * 0.22, bh * 0.27);
            ctx.strokeRect(cx - bw * 0.11, cy - bh * 0.05, bw * 0.22, bh * 0.27);
        } else if (type === 'greenhouse' || type === 'hydroponics') {
            const x = cx - bw * 0.45;
            const y = cy - bh * 0.25;
            const width = bw * 0.90;
            const height = bh * 0.50;
            ctx.fillStyle = 'rgba(90, 111, 91, 0.66)';
            ctx.strokeStyle = '#9cab91';
            ctx.lineWidth = 1.2;
            roundedRect(ctx, x, y, width, height, height / 2);
            ctx.fill();
            ctx.stroke();
            ctx.strokeStyle = 'rgba(210, 217, 196, 0.55)';
            for (let offset = 0.18; offset < 0.9; offset += 0.18) {
                ctx.beginPath();
                ctx.moveTo(x + width * offset, y + 1);
                ctx.lineTo(x + width * offset, y + height - 1);
                ctx.stroke();
            }
            ctx.fillStyle = '#4f674b';
            ctx.fillRect(x + width * 0.08, cy + height * 0.06, width * 0.84, Math.max(2, height * 0.12));
        } else if (type === 'solar_panel' || type === 'solar_array') {
            const panelW = bw * 0.41;
            const panelH = bh * 0.57;
            [-0.24, 0.24].forEach(offset => {
                const x = cx + bw * offset - panelW / 2;
                const y = cy - panelH / 2;
                ctx.fillStyle = '#26363a';
                ctx.strokeStyle = '#80969a';
                ctx.lineWidth = 1;
                ctx.fillRect(x, y, panelW, panelH);
                ctx.strokeRect(x, y, panelW, panelH);
                ctx.strokeStyle = 'rgba(145, 171, 176, 0.42)';
                ctx.beginPath();
                ctx.moveTo(x + panelW / 2, y); ctx.lineTo(x + panelW / 2, y + panelH);
                ctx.moveTo(x, y + panelH / 2); ctx.lineTo(x + panelW, y + panelH / 2);
                ctx.stroke();
            });
            ctx.strokeStyle = '#a49b89';
            ctx.beginPath();
            ctx.moveTo(cx, cy - bh * 0.34);
            ctx.lineTo(cx, cy + bh * 0.42);
            ctx.stroke();
        } else if (type === 'potable_water_tank') {
            drawTank(ctx, cx, cy, bw, bh, palette.water);
        } else if (type === 'oxygen_buffer_tank') {
            drawTank(ctx, cx, cy, bw, bh, palette.oxygen);
        } else if (type === 'life_support_distribution_grid') {
            const size = Math.min(bw, bh) * 0.72;
            ctx.fillStyle = '#303432';
            ctx.strokeStyle = '#aaa797';
            ctx.lineWidth = 1.2;
            roundedRect(ctx, cx - size / 2, cy - size / 2, size, size, 3);
            ctx.fill();
            ctx.stroke();
            ctx.strokeStyle = '#5d91a0';
            [-0.22, 0, 0.22].forEach(offset => {
                ctx.beginPath();
                ctx.moveTo(cx - size * 0.32, cy + size * offset);
                ctx.lineTo(cx + size * 0.32, cy + size * offset);
                ctx.stroke();
            });
            ctx.fillStyle = palette.safety;
            ctx.fillRect(cx - size * 0.08, cy - size * 0.08, size * 0.16, size * 0.16);
        } else if (type === 'communications_array') {
            ctx.strokeStyle = '#b9b4a7';
            ctx.lineWidth = 1.4;
            ctx.beginPath();
            ctx.arc(cx, cy - bh * 0.08, Math.min(bw, bh) * 0.28, 0.15 * Math.PI, 0.85 * Math.PI);
            ctx.stroke();
            ctx.beginPath();
            ctx.moveTo(cx, cy); ctx.lineTo(cx, cy + bh * 0.33);
            ctx.moveTo(cx - bw * 0.20, cy + bh * 0.33); ctx.lineTo(cx + bw * 0.20, cy + bh * 0.33);
            ctx.stroke();
            ctx.fillStyle = palette.safety;
            ctx.beginPath(); ctx.arc(cx, cy - bh * 0.08, Math.max(2, 2.5 * mapZoom), 0, Math.PI * 2); ctx.fill();
        } else if (type === 'landing_zone') {
            const radius = Math.min(bw, bh) * 0.39;
            ctx.fillStyle = '#2a2a25';
            ctx.strokeStyle = '#b78a50';
            ctx.lineWidth = Math.max(1.2, 1.6 * mapZoom);
            ctx.beginPath(); ctx.arc(cx, cy, radius, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
            ctx.fillStyle = '#cbb58f';
            ctx.font = `800 ${Math.max(8, 12 * mapZoom)}px JetBrains Mono, monospace`;
            ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText('H', cx, cy + 1);
        } else if (type === 'storage_crate') {
            ctx.fillStyle = '#665d4d';
            ctx.strokeStyle = '#aaa087';
            ctx.lineWidth = 1;
            ctx.fillRect(cx - bw * 0.33, cy - bh * 0.28, bw * 0.66, bh * 0.56);
            ctx.strokeRect(cx - bw * 0.33, cy - bh * 0.28, bw * 0.66, bh * 0.56);
            ctx.beginPath();
            ctx.moveTo(cx - bw * 0.33, cy - bh * 0.28); ctx.lineTo(cx + bw * 0.33, cy + bh * 0.28);
            ctx.moveTo(cx + bw * 0.33, cy - bh * 0.28); ctx.lineTo(cx - bw * 0.33, cy + bh * 0.28);
            ctx.stroke();
        } else {
            const accent = type === 'water_collector' || type === 'water_purifier'
                ? palette.water
                : (type === 'isru_o2_unit' ? palette.oxygen : palette.safety);
            drawIndustrialSkid(ctx, cx, cy, bw, bh, accent, type);
        }

        if (Number(health) < 0.99) {
            const width = bw * 0.72;
            const y = cy + bh * 0.43;
            ctx.fillStyle = '#242420';
            ctx.fillRect(cx - width / 2, y, width, 2.5);
            ctx.fillStyle = Number(health) > 0.5 ? palette.safety : palette.danger;
            ctx.fillRect(cx - width / 2, y, width * Math.max(0, Number(health)), 2.5);
        }
        compactLabel(ctx, labels[type] || String(type).replace(/_/g, ' ').toUpperCase(), cx, cy - bh * 0.55, mapZoom);
        ctx.restore();
        return true;
    }

    function drawFleetVehicle({
        ctx, vehicle, vehicleId, type, state, color, vx, vy,
        mapZoom, batteryPct, job, mission,
    }) {
        const vehicleStyles = {
            crew_rover: {
                lines: ['CREW', 'ROVER'], short: 'CR', border: '#497184',
                text: '#a9c6d3', fill: 'rgba(8, 20, 32, 0.97)',
            },
            cargo_transporter: {
                lines: ['CARGO', 'ROVER'], short: 'CT', border: '#805b43',
                text: '#cfad94', fill: 'rgba(29, 17, 13, 0.97)',
            },
            excavator: {
                lines: ['EXCAVATOR'], short: 'EX', border: '#776a3e',
                text: '#c8bb81', fill: 'rgba(27, 23, 12, 0.97)',
            },
            assembly_robot: {
                lines: ['ASSEMBLY', 'ROBOT'], short: 'AR', border: '#416f55',
                text: '#9ac5a9', fill: 'rgba(9, 25, 17, 0.97)',
            },
        };
        const style = vehicleStyles[type] || {
            lines: ['SURFACE', 'VEHICLE'], short: 'SV', border: '#64748b',
            text: '#b8c2cf',
            fill: 'rgba(15, 23, 42, 0.96)',
        };
        const stateNames = {
            cargo_outbound: 'EN ROUTE', cargo_returning: 'RETURNING',
            cargo_unloading: 'UNLOADING', outbound: 'EN ROUTE',
            returning: 'RETURNING', waiting_supervision: 'AWAITING CREW',
            excavating: 'EXCAVATING', working: 'ASSEMBLING',
            charging: 'CHARGING', in_use: 'CREW MISSION',
            cooldown: 'SYSTEM CHECK', fault: 'FAULT', idle: 'READY',
            unloading: 'UNLOADING',
        };
        const stateColor = state === 'fault' ? '#9f4056' : style.border;
        const suffix = String(vehicleId).match(/\d+$/)?.[0] || '1';
        const markerSize = Math.max(34, Math.min(48, 40 * mapZoom));
        const markerX = vx - markerSize / 2;
        const markerY = vy - markerSize / 2;

        let taskText = 'NO ACTIVE TASK';
        if (job) {
            taskText = String(job.resource || job.id || 'ACTIVE JOB')
                .replace(/_/g, ' ').toUpperCase();
        } else if (mission && type === 'crew_rover') {
            taskText = `${(mission.crew_ids || []).length} CREW ABOARD`;
        } else if (mission && mission.site_id) {
            taskText = `SITE ${String(mission.site_id).replace(/^struct_/, '').replace(/_/g, ' ').toUpperCase()}`;
        }
        if (type === 'cargo_transporter' && Number(vehicle.payload_mass_kg || 0) > 0) {
            taskText = `LOAD ${Math.round(Number(vehicle.payload_mass_kg))} KG`;
        }
        taskText = taskText.slice(0, 26);

        ctx.save();
        ctx.lineJoin = 'miter';

        // Deliberately symbolic: a readable grid token, not a fake rover drawing.
        const crispMarkerX = Math.round(markerX) + 0.5;
        const crispMarkerY = Math.round(markerY) + 0.5;
        const crispMarkerSize = Math.max(1, Math.round(markerSize) - 1);
        ctx.fillStyle = style.fill;
        ctx.fillRect(crispMarkerX, crispMarkerY, crispMarkerSize, crispMarkerSize);
        ctx.strokeStyle = stateColor;
        ctx.lineWidth = 1;
        ctx.strokeRect(crispMarkerX, crispMarkerY, crispMarkerSize, crispMarkerSize);

        const innerFont = Math.max(5.5, Math.min(8, 6.8 * mapZoom));
        ctx.fillStyle = style.text;
        ctx.font = `800 ${innerFont}px JetBrains Mono, monospace`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        if (style.lines.length === 1) {
            ctx.fillText(style.lines[0], vx, vy - 1);
        } else {
            ctx.fillText(style.lines[0], vx, vy - innerFont * 0.62);
            ctx.fillText(style.lines[1], vx, vy + innerFont * 0.62);
        }

        const batteryColor = batteryPct > 50
            ? '#50785d' : (batteryPct > 20 ? '#957039' : '#934858');
        const barInset = Math.max(3, 4 * mapZoom);
        const barHeight = Math.max(2.5, 3 * mapZoom);
        ctx.fillStyle = 'rgba(2, 6, 13, 0.96)';
        ctx.fillRect(markerX + barInset, markerY + markerSize - barInset - barHeight,
            markerSize - barInset * 2, barHeight);
        ctx.fillStyle = batteryColor;
        ctx.fillRect(markerX + barInset, markerY + markerSize - barInset - barHeight,
            (markerSize - barInset * 2) * batteryPct / 100, barHeight);

        // Two-line callout with a pointer makes status readable without turning
        // the vehicle token itself into a dense miniature illustration.
        const calloutFont = Math.max(6, Math.min(8.5, 7.3 * mapZoom));
        const title = `${style.short}-${suffix}  ${stateNames[state] || state.toUpperCase()}`;
        const detail = `${Math.round(batteryPct)}% BATTERY  ·  ${taskText}`;
        ctx.font = `800 ${calloutFont}px JetBrains Mono, monospace`;
        const titleWidth = ctx.measureText(title).width;
        ctx.font = `600 ${Math.max(5.5, calloutFont - 0.7)}px JetBrains Mono, monospace`;
        const detailWidth = ctx.measureText(detail).width;
        const calloutWidth = Math.max(78, titleWidth, detailWidth) + 14;
        const calloutHeight = calloutFont * 2 + 10;
        const calloutX = vx - calloutWidth / 2;
        const calloutY = markerY - calloutHeight - Math.max(8, 10 * mapZoom);

        const crispCalloutX = Math.round(calloutX) + 0.5;
        const crispCalloutY = Math.round(calloutY) + 0.5;
        const crispCalloutWidth = Math.max(1, Math.round(calloutWidth) - 1);
        const crispCalloutHeight = Math.max(1, Math.round(calloutHeight) - 1);
        ctx.fillStyle = 'rgba(7, 11, 20, 0.97)';
        ctx.strokeStyle = stateColor;
        ctx.lineWidth = 1;
        ctx.fillRect(crispCalloutX, crispCalloutY, crispCalloutWidth, crispCalloutHeight);
        ctx.strokeRect(crispCalloutX, crispCalloutY, crispCalloutWidth, crispCalloutHeight);
        ctx.beginPath();
        ctx.moveTo(vx - 5, calloutY + calloutHeight);
        ctx.lineTo(vx, calloutY + calloutHeight + 6);
        ctx.lineTo(vx + 5, calloutY + calloutHeight);
        ctx.closePath();
        ctx.fillStyle = 'rgba(7, 11, 20, 0.97)';
        ctx.fill();
        ctx.strokeStyle = stateColor;
        ctx.beginPath();
        ctx.moveTo(vx - 5, calloutY + calloutHeight);
        ctx.lineTo(vx, calloutY + calloutHeight + 6);
        ctx.lineTo(vx + 5, calloutY + calloutHeight);
        ctx.stroke();

        ctx.textBaseline = 'middle';
        ctx.textAlign = 'left';
        ctx.fillStyle = state === 'fault' ? '#c66b7e' : style.text;
        ctx.font = `800 ${calloutFont}px JetBrains Mono, monospace`;
        ctx.fillText(title, calloutX + 7, calloutY + calloutFont * 0.82 + 2);
        ctx.fillStyle = '#cbd5e1';
        ctx.font = `600 ${Math.max(5.5, calloutFont - 0.7)}px JetBrains Mono, monospace`;
        ctx.fillText(detail, calloutX + 7, calloutY + calloutFont * 1.78 + 3);

        ctx.fillStyle = stateColor;
        ctx.fillRect(
            Math.round(calloutX + calloutWidth - 10),
            Math.round(calloutY + 6),
            4,
            4,
        );
        ctx.restore();
        return true;
    }

    function fleetRectsOverlap(first, second, padding = 3) {
        return !(
            first.x2 + padding <= second.x1
            || first.x1 >= second.x2 + padding
            || first.y2 + padding <= second.y1
            || first.y1 >= second.y2 + padding
        );
    }

    function chooseFleetLabelRect({
        vx, vy, width, height, markerWidth, markerHeight,
        viewportWidth, viewportHeight,
        blockedRects,
    }) {
        const gap = Math.max(4, Math.min(13, markerWidth * 0.24));
        const halfMarkerX = markerWidth / 2;
        const halfMarkerY = markerHeight / 2;
        const candidates = [
            [vx - width / 2, vy - halfMarkerY - gap - height],
            [vx + halfMarkerX + gap, vy - height / 2],
            [vx - halfMarkerX - gap - width, vy - height / 2],
            [vx - width / 2, vy + halfMarkerY + gap],
            [vx + halfMarkerX + gap, vy - halfMarkerY - gap - height],
            [vx - halfMarkerX - gap - width, vy - halfMarkerY - gap - height],
            [vx + halfMarkerX + gap, vy + halfMarkerY + gap],
            [vx - halfMarkerX - gap - width, vy + halfMarkerY + gap],
        ];
        for (let lane = 1; lane <= 7; lane += 1) {
            const shift = Math.ceil(lane / 2) * (height + 5) * (lane % 2 ? 1 : -1);
            candidates.push(
                [vx + halfMarkerX + gap, vy - height / 2 + shift],
                [vx - halfMarkerX - gap - width, vy - height / 2 + shift],
            );
        }

        const margin = 5;
        const fits = (x, y) => {
            const rect = { x1: x, y1: y, x2: x + width, y2: y + height };
            if (
                rect.x1 < margin || rect.y1 < margin
                || rect.x2 > viewportWidth - margin
                || rect.y2 > viewportHeight - margin
            ) return null;
            return blockedRects.some(blocked => fleetRectsOverlap(rect, blocked))
                ? null : rect;
        };
        for (const [x, y] of candidates) {
            const result = fits(x, y);
            if (result) return result;
        }

        // Dense depot fallback: search the nearest free screen lane instead of
        // allowing labels to overlap. A thin leader preserves map association.
        const freeSlots = [];
        for (let y = margin; y + height < viewportHeight - margin; y += height + 5) {
            for (let x = margin; x + width < viewportWidth - margin; x += width + 5) {
                const rect = fits(x, y);
                if (rect) {
                    const dx = (x + width / 2) - vx;
                    const dy = (y + height / 2) - vy;
                    freeSlots.push({ rect, distance: dx * dx + dy * dy });
                }
            }
        }
        freeSlots.sort((first, second) => first.distance - second.distance);
        if (freeSlots.length) return freeSlots[0].rect;

        const x = Math.max(margin, Math.min(
            viewportWidth - width - margin,
            vx - width / 2,
        ));
        const y = Math.max(margin, Math.min(
            viewportHeight - height - margin,
            vy - halfMarkerY - gap - height,
        ));
        return { x1: x, y1: y, x2: x + width, y2: y + height };
    }

    function drawFleetVehicleIntegrated({
        ctx, vehicle, vehicleId, type, state, color, vx, vy,
        mapZoom, batteryPct, job, mission, vis,
        clusterVehicles = [], showDetails = true, drawMarker = true,
        fleetLabelRects = [], fleetMarkerRects = [],
        viewportWidth = 10000, viewportHeight = 10000,
    }) {
        const styles = {
            crew_rover: { name: 'CREW ROVER', lines: ['CREW', 'ROVER'], border: '#607b88', fill: 'rgba(9, 18, 31, 0.97)', text: '#a9bdc7', accent: '#3f879e' },
            cargo_transporter: { name: 'CARGO ROVER', lines: ['CARGO', 'ROVER'], border: '#806957', fill: 'rgba(27, 18, 15, 0.97)', text: '#c2aa98', accent: '#a56843' },
            excavator: { name: 'EXCAVATOR', lines: ['EXCAVATOR'], border: '#81734f', fill: 'rgba(28, 24, 15, 0.97)', text: '#c5b98d', accent: '#a67c31' },
            assembly_robot: { name: 'ASSEMBLY ROBOT', lines: ['ASSEMBLY', 'ROBOT'], border: '#587363', fill: 'rgba(11, 24, 18, 0.97)', text: '#a5bdaa', accent: '#4f8060' },
        };
        const style = styles[type] || {
            name: 'SURFACE VEHICLE', lines: ['SURFACE', 'VEHICLE'], border: '#64748b',
            fill: 'rgba(12, 20, 34, 0.97)', text: '#b6c0ce', accent: '#64748b',
        };
        const stateNames = {
            cargo_outbound: 'OUTBOUND', cargo_returning: 'RETURNING',
            cargo_unloading: 'UNLOADING', outbound: 'OUTBOUND',
            returning: 'RETURNING', waiting_supervision: 'CREW HOLD',
            excavating: 'EXCAVATING', working: 'WORKING', charging: 'CHARGING',
            in_use: 'CREW MISSION', cooldown: 'SYSTEM CHECK', fault: 'FAULT',
            idle: 'READY', unloading: 'UNLOADING',
        };
        const group = clusterVehicles.length ? clusterVehicles : [vehicle];
        const isCluster = group.length > 1;
        const suffix = String(vehicleId).match(/\d+$/)?.[0] || '1';
        // Deliberately shares the astronauts' simple marker grammar while the
        // square silhouette keeps vehicles unmistakable beside round crew.
        // Scale with the terrain exactly like crew markers do. Pixel caps made
        // vehicles appear to shrink relative to cells at close zoom levels.
        const coreRadius = 8 * mapZoom;
        const haloRadius = 16 * mapZoom;
        const markerWidth = haloRadius * 2;
        const markerHeight = markerWidth;
        const stateColor = state === 'fault' ? '#f43f5e' : (color || style.border);

        const vehicleBattery = item => {
            if (Number.isFinite(Number(item.battery_pct))) return Number(item.battery_pct);
            return Number(item.battery_kwh || 0)
                / Math.max(0.001, Number(item.battery_capacity_kwh || 1)) * 100;
        };
        const displayBatteryPct = isCluster
            ? group.reduce((sum, item) => sum + vehicleBattery(item), 0) / group.length
            : batteryPct;

        if (drawMarker) {
            ctx.save();
            ctx.shadowColor = style.accent;
            ctx.shadowBlur = Math.max(2, Math.min(8, 6 * mapZoom));

            const haloX = vx - haloRadius;
            const haloY = vy - haloRadius;
            const haloSize = haloRadius * 2;
            ctx.fillStyle = `${style.accent}20`;
            ctx.fillRect(haloX, haloY, haloSize, haloSize);
            ctx.strokeStyle = `${style.accent}66`;
            ctx.lineWidth = Math.max(1, 1.2 * mapZoom);
            ctx.strokeRect(haloX, haloY, haloSize, haloSize);

            const coreX = vx - coreRadius;
            const coreY = vy - coreRadius;
            const coreSize = coreRadius * 2;
            ctx.fillStyle = '#0f172a';
            ctx.fillRect(coreX, coreY, coreSize, coreSize);
            ctx.strokeStyle = style.accent;
            ctx.lineWidth = Math.max(1.5, Math.min(2.5, 2 * mapZoom));
            ctx.strokeRect(coreX, coreY, coreSize, coreSize);

            // A quiet in-core line carries charge state without turning the
            // marker into a miniature vehicle drawing.
            const batteryColor = displayBatteryPct > 50
                ? '#4f8060' : (displayBatteryPct > 20 ? '#a77a39' : '#a54a5d');
            ctx.shadowBlur = 0;
            const batteryInset = 2 * mapZoom;
            const batteryTrackWidth = Math.max(1, coreSize - batteryInset * 2);
            const batteryHeight = 1.35 * mapZoom;
            const batteryY = coreY + coreSize - batteryInset - batteryHeight;
            ctx.fillStyle = 'rgba(148, 163, 184, 0.18)';
            ctx.fillRect(coreX + batteryInset, batteryY, batteryTrackWidth, batteryHeight);
            ctx.fillStyle = batteryColor;
            ctx.fillRect(
                coreX + batteryInset,
                batteryY,
                batteryTrackWidth * Math.max(0, Math.min(100, displayBatteryPct)) / 100,
                batteryHeight,
            );

            if (isCluster) {
                const countFont = 6.5 * mapZoom;
                ctx.fillStyle = '#f1f5f9';
                ctx.font = `700 ${countFont}px Outfit, sans-serif`;
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText(String(group.length), vx, vy + 0.5);
            } else {
                const beaconSize = 5 * mapZoom;
                ctx.fillStyle = state === 'fault' ? '#f43f5e' : style.accent;
                ctx.fillRect(vx - beaconSize / 2, vy - beaconSize / 2, beaconSize, beaconSize);
            }
            ctx.restore();
        }

        // No label at overview scale; one line at normal scale, full telemetry
        // only when the visitor deliberately zooms in.
        if (!showDetails || mapZoom < 0.9) return true;
        const showFullDetail = mapZoom >= 1.35;

        let title;
        let detail;
        if (isCluster) {
            const counts = new Map();
            group.forEach(item => {
                const itemType = item.vehicle_type || 'surface_vehicle';
                const key = (styles[itemType] || { name: 'SURFACE VEHICLE' }).name;
                counts.set(key, (counts.get(key) || 0) + 1);
            });
            title = `FLEET ×${group.length}  [PARKED]  ·  AVG BATT ${Math.round(displayBatteryPct)}%`;
            detail = [...counts.entries()]
                .map(([key, count]) => `${key}×${count}`)
                .join('  ·  ');
        } else {
            const fullId = `${style.name} ${suffix}`;
            title = `${fullId}  [${stateNames[state] || state.toUpperCase()}]  ·  BATT ${Math.round(displayBatteryPct)}%`;
            const payloadKg = Number(vehicle.payload_mass_kg || 0);
            const load = payloadKg > 0
                ? `${payloadKg.toFixed(payloadKg >= 10 ? 0 : 1)} KG`
                : 'EMPTY';
            let assignment = 'IDLE';
            if (job) {
                assignment = String(job.resource || job.id || 'ACTIVE')
                    .replace(/_/g, ' ').toUpperCase();
            } else if (mission && type === 'crew_rover') {
                assignment = `CREW ${(mission.crew_ids || []).length}`;
            } else if (mission && mission.site_id) {
                assignment = `SITE ${String(mission.site_id).replace(/^struct_/, '').replace(/_/g, ' ').toUpperCase()}`;
            }
            const gx = Math.round(Number(vis?.gx ?? vehicle.x ?? 0));
            const gy = Math.round(Number(vis?.gy ?? vehicle.y ?? 0));
            detail = `[${gx},${gy}]  ·  JOB ${assignment.slice(0, 22)}  ·  LOAD ${load}`;
        }

        const titleFont = Math.max(5, Math.min(12, 5.5 * mapZoom));
        const detailFont = Math.max(4.5, Math.min(10.5, 4.6 * mapZoom));
        ctx.save();
        ctx.font = `600 ${titleFont}px Outfit, sans-serif`;
        const titleWidth = ctx.measureText(title).width;
        ctx.font = `600 ${detailFont}px JetBrains Mono, monospace`;
        const detailWidth = showFullDetail ? ctx.measureText(detail).width : 0;
        const horizontalPadding = Math.max(6, Math.min(18, 9 * mapZoom));
        const verticalPadding = Math.max(4, Math.min(14, 6.5 * mapZoom));
        const labelWidth = Math.max(46, titleWidth, detailWidth) + horizontalPadding;
        const labelHeight = titleFont
            + (showFullDetail ? detailFont : 0)
            + verticalPadding;
        const blockedRects = [...fleetMarkerRects, ...fleetLabelRects];
        const labelRect = chooseFleetLabelRect({
            vx, vy,
            width: labelWidth,
            height: labelHeight,
            markerWidth,
            markerHeight,
            viewportWidth,
            viewportHeight,
            blockedRects,
        });
        fleetLabelRects.push(labelRect);

        const anchorX = Math.max(labelRect.x1, Math.min(vx, labelRect.x2));
        const anchorY = Math.max(labelRect.y1, Math.min(vy, labelRect.y2));
        const leaderDx = anchorX - vx;
        const leaderDy = anchorY - vy;
        const leaderScale = 1 / Math.max(
            1,
            Math.abs(leaderDx) / Math.max(1, markerWidth / 2),
            Math.abs(leaderDy) / Math.max(1, markerHeight / 2),
        );
        const leaderStartX = vx + leaderDx * leaderScale;
        const leaderStartY = vy + leaderDy * leaderScale;
        ctx.strokeStyle = stateColor;
        ctx.globalAlpha = 0.45;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(leaderStartX, leaderStartY);
        ctx.lineTo(anchorX, anchorY);
        ctx.stroke();

        ctx.globalAlpha = 1;
        const labelX = Math.round(labelRect.x1) + 0.5;
        const labelY = Math.round(labelRect.y1) + 0.5;
        const labelW = Math.round(labelRect.x2 - labelRect.x1) - 1;
        const labelH = Math.round(labelRect.y2 - labelRect.y1) - 1;
        ctx.fillStyle = 'rgba(6, 9, 19, 0.9)';
        ctx.beginPath();
        ctx.roundRect(labelX, labelY, labelW, labelH, 4);
        ctx.fill();
        ctx.strokeStyle = stateColor;
        ctx.globalAlpha = 0.78;
        ctx.lineWidth = 1;
        ctx.stroke();
        ctx.globalAlpha = 1;

        ctx.textAlign = 'left';
        ctx.textBaseline = 'middle';
        ctx.fillStyle = '#f8fafc';
        ctx.font = `600 ${titleFont}px Outfit, sans-serif`;
        ctx.fillText(title, labelRect.x1 + 6, labelRect.y1 + titleFont * 0.82 + 1);
        if (showFullDetail) {
            ctx.fillStyle = '#cbd5e1';
            ctx.font = `600 ${detailFont}px JetBrains Mono, monospace`;
            ctx.fillText(detail, labelRect.x1 + 6, labelRect.y2 - detailFont * 0.72 - 1);
        }
        ctx.restore();
        return true;
    }

    const structureParcelScales = {
        storage_crate: 0.56,
        life_support_distribution_grid: 0.58,
        potable_water_tank: 0.58,
        oxygen_buffer_tank: 0.58,
        habitat_module: 0.72,
        radiation_shelter: 0.72,
        medical_station: 0.64,
        water_collector: 0.66,
        water_purifier: 0.60,
        isru_o2_unit: 0.64,
        greenhouse: 0.78,
        hydroponics: 0.78,
        power_distribution_grid: 0.58,
        solar_panel: 0.82,
        solar_array: 0.82,
        communications_array: 0.54,
        forge: 0.66,
        cnc_fabricator: 0.66,
        stone_furnace: 0.66,
    };

    function structureScale(type) {
        return structureParcelScales[type] || 0.66;
    }

    window.IndustrialTacticalTheme = {
        // Keep the original palette and structure drawings. The scale hook
        // makes their visible equipment envelope smaller than the 100 m civil
        // parcel, leaving a truthful maintenance and rover-access apron.
        palette: mapPalette,
        structureScale,
        drawFleetVehicle: drawFleetVehicleIntegrated,
    };
})();
