# Intelligent Hedging with Reinforcement Learning

A research-oriented framework for developing, evaluating, and comparing reinforcement learning methods for dynamic derivative hedging under transaction costs.

---

## Overview

This project investigates whether reinforcement learning can outperform traditional delta hedging by learning optimal hedge adjustments under realistic market conditions.

The framework is designed to be:

- Modular
- Reproducible
- Extensible
- Research-friendly

The implementation is inspired primarily by:

- Deep Hedging of Derivatives Using Reinforcement Learning (Cao et al.)
- Hedging using Reinforcement Learning: Contextual Bandits versus Q-learning

---

## Features

- Black-Scholes pricing engine
- Option Greeks
- GBM and stochastic volatility simulators
- Portfolio accounting
- Delta hedging baseline
- Threshold hedging baseline
- Deep RL agents (DDPG, TD3, SAC)
- Contextual bandit adaptation
- Prioritized replay
- Particle Swarm Optimization (PSO)
- Comprehensive backtesting
- Risk metrics and stress testing

---

## Project Structure

```text
src/
    pricing/
    simulation/
    environment/
    baseline/
    agents/
    replay/
    tuning/
    evaluation/
    utils/
```

---

## Installation

Clone the repository.

```bash
git clone <repository-url>
```

Create a virtual environment.

```bash
python -m venv .venv
```

Activate it.

Windows

```bash
.venv\Scripts\activate
```

Linux/macOS

```bash
source .venv/bin/activate
```

Install dependencies.

```bash
pip install -r requirements.txt
```

---

## Development Workflow

Every feature follows the same process.

1. Implement
2. Test
3. Refactor
4. Document
5. Commit

No feature is considered complete until unit tests pass.

---

## Project Roadmap

### Phase 1 — Financial Foundations

- Option contract model
- Black-Scholes pricing
- Greeks
- GBM simulator
- Portfolio accounting
- Delta hedging

### Phase 2 — Environment

- Hedging environment
- Reward functions
- Transaction costs
- Accounting P&L

### Phase 3 — Baselines

- Delta hedging
- Threshold hedging
- No hedge

### Phase 4 — Reinforcement Learning

- Replay buffer
- DDPG
- Prioritized replay
- Twin critics
- Risk-aware objective

### Phase 5 — Online Adaptation

- Contextual bandits
- Hybrid RL + Bandit

### Phase 6 — Optimization

- Particle Swarm Optimization
- Hyperparameter tuning

---

## Current Status

| Module | Status |
|---------|--------|
| Project Skeleton | ✅ |
| Pricing Engine | ⏳ |
| Greeks | ⏳ |
| Simulator | ⏳ |
| Environment | ⏳ |
| Baselines | ⏳ |
| RL Agents | ⏳ |

---

## License

MIT License