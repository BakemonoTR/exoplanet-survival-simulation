# 🪐 Microsoft AI Innovators — Exoplanet Survival Simulation

<div align="center">

**A scientifically grounded, LLM-powered multi-agent survival simulation on real exoplanets.**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Groq](https://img.shields.io/badge/Groq-LLaMA_3.3_70B-F55036)](https://groq.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## 📖 Overview

This project is a **research simulation** developed for the **Microsoft AI Innovators Program**. It models a realistic colony-building scenario where **6 autonomous AI agents** cooperate to prepare life-sustaining infrastructure for **106 incoming colonists** on scientifically accurate exoplanet environments.

Each agent is powered by **LLaMA 3.3 70B** (via Groq) and makes decisions through a sophisticated pipeline combining:
- 🧠 **LLM-based strategic reasoning** with contextual prompts
- 🎯 **Reinforcement Learning** for effort allocation optimization
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
│   ├── llm_client.py    # Groq API client with key rotation
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
└── index.html       # Real-time simulation dashboard
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- [Groq API key](https://console.groq.com/keys) (free tier available)

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

# Configure API keys
cp .env.example .env
# Edit .env and add your Groq API key(s)
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

## 🎮 Key Features

### Multi-Agent AI System
- **6 specialized agents** with distinct roles (engineering, agriculture, medical, etc.)
- Each agent uses **LLaMA 3.3 70B** for strategic decision-making
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
- **Groq** — for ultra-fast LLM inference
- **Meta AI** — for the LLaMA 3.3 70B model
- **NASA Exoplanet Archive** — for planetary data

---

<div align="center">

**Built with 🚀 for the Microsoft AI Innovators Program**

</div>
