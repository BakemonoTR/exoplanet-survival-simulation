# 🪐 Microsoft AI Innovators — Exoplanet Survival Simulation

<div align="center">

**A scientifically grounded, reinforcement-learning multi-agent survival simulation on real exoplanets.**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Narrative](https://img.shields.io/badge/Narrative-Local_GPT--2-607D8B)](#local-narrative-layer)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## 📖 Overview

This project is a **research simulation** developed for the **Microsoft AI Innovators Program**. It models a realistic colony-building scenario where **6 autonomous AI agents** cooperate to prepare life-sustaining infrastructure for **106 incoming colonists** on scientifically accurate exoplanet environments.

Agent actions are controlled by deterministic simulation logic and **reinforcement learning**. An optional local GPT-2 adapter is reserved for non-authoritative dialogue and reflections:
- 🎯 **Reinforcement Learning** for action and effort-allocation optimization
- 💬 **Local narrative generation** for dialogue and reflections only
- 🔄 **Emergent cooperative behavior** under resource constraints
- 📊 **Realistic physics and resource systems** grounded in planetary science

## 🌍 Supported Exoplanets

| Planet | Distance | Type | Key Challenge |
|--------|----------|------|---------------|
| **TRAPPIST-1e** | 39.5 ly | Tidally locked | Extreme temperature gradients |
| **Proxima Centauri b** | 4.2 ly | Rocky, temperate | Stellar flare radiation |
| **Kepler-442b** | 1,206 ly | Super-Earth | High gravity (1.34g) |
| **Ross 128 b** | 11 ly | Temperate rocky | Low stellar energy |
| **Teegarden's Star b** | 12.5 ly | Earth-like | Limited sunlight |

## 🏗️ Architecture

```
src/
├── agents/          # AI agent system
│   ├── agent.py         # Core agent logic & behavior
│   ├── decision.py      # Decision-making pipeline
│   ├── prompts.py       # LLM prompt engineering
│   └── strategic_rl.py  # Reinforcement learning module
├── orchestration/   # Simulation engine
│   ├── engine.py        # Main simulation loop & tick processing
│   ├── llm_client.py    # Optional local GPT-2 narrative adapter
│   ├── challenge.py     # Dynamic challenge system
│   └── fallback.py      # Fallback decision logic
├── systems/         # Game systems
│   ├── colony_score.py      # Colony readiness scoring
│   ├── event_scheduler.py   # Dynamic event generation
│   ├── mission_profile.py   # Mission configuration
│   ├── surface_fleet.py     # Rover & vehicle management
│   ├── cargo_ledger.py      # Resource tracking
│   ├── airlock.py           # EVA management
│   └── timebase.py          # Time system
├── world/           # Procedural world generation
│   ├── generator.py     # Terrain & biome generation
│   └── sparse_state.py  # Efficient world state management
├── memory/          # Agent memory systems
│   ├── vector_store.py  # Semantic memory with embeddings
│   └── reflection.py    # Agent self-reflection
├── api/             # Web interface
│   ├── server.py        # FastAPI WebSocket server
│   └── database.py      # SQLite persistence
config/
├── planets/         # Exoplanet configurations (JSON)
├── mission_profile.json
├── agent_presets.json
├── colony_targets.json
└── recipes.json     # Manufacturing recipes
frontend/
├── index.html       # Public observation dashboard
└── icarus.html      # Private mission-control panel
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/exoplanet-survival-simulation.git
cd exoplanet-survival-simulation

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Configure local settings
cp .env.example .env
# Set ICARUS_ADMIN_PASSWORD before exposing the service
```

### Running the Simulation

```bash
# Start with web UI (default)
python run.py

# Run headless (no UI)
python run.py --headless

# Select a specific planet
python run.py --planet trappist-1e --seed 42

# Custom port
python run.py --port 8080
```

Then open `http://localhost:8000` in your browser to view the real-time simulation dashboard.
The protected mission-control panel is available at `http://localhost:8000/icarus`.

## 🎮 Key Features

### Multi-Agent AI System
- **6 specialized agents** with distinct roles (engineering, agriculture, medical, etc.)
- Agent actions remain deterministic and RL-authoritative
- **Reinforcement learning** optimizes effort allocation across tasks
- **Semantic memory** with sentence-transformer embeddings for experience recall
- **Self-reflection** system for learning from past decisions

### Realistic Simulation
- **Procedural world generation** with Perlin noise terrain
- **Physics-based resource systems** (atmosphere, water, food, power)
- **Dynamic event system** (dust storms, equipment failures, solar flares)
- **Manufacturing pipeline** with realistic recipes and BOM tracking
- **EVA (spacewalk) management** with airlock protocols
- **Rover fleet operations** for surface exploration

### Real-Time Dashboard
- **WebSocket-powered** live updates
- **Tick-by-tick** simulation visualization
- **Resource graphs** and agent status monitoring
- **Colony readiness score** tracking

### Local Narrative Layer
- No hosted language-model API is required
- A fine-tuned local GPT-2 provider can be connected through `LocalNarrativeProvider`
- Generated text is limited to conversations and reflections and cannot alter physics, rewards, trust, inventory, or actions

## 🧪 Testing

```bash
# Run all tests
python -m pytest

# Run specific test suites
python -m pytest test_mission_architecture.py
python -m pytest test_emergency_realism.py
python -m pytest test_food_mass_balance.py
```

## 🔧 Configuration

### Planet Configuration
Each planet is defined in `config/planets/<planet-name>.json` with scientifically accurate parameters including:
- Orbital mechanics (semi-major axis, eccentricity, period)
- Atmospheric composition and pressure
- Surface temperature ranges
- Gravity and radiation levels
- Available resources

### Mission Profile
`config/mission_profile.json` defines:
- Mission duration and tick limits
- Crew size and composition
- Supply manifest
- Success criteria and colony targets

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Microsoft AI Innovators Program** — for the opportunity and support
- **NASA Exoplanet Archive** — for planetary data

---

<div align="center">

**Built with 🚀 for the Microsoft AI Innovators Program**

</div>
