# Derivative Hedging using Deep Reinforcement Learning

A modular Python framework for **option hedging** using **Deep Deterministic Policy Gradient (DDPG)**. The project compares a reinforcement learning based hedging strategy against classical Black-Scholes Delta Hedging and a Random baseline under simulated market conditions.

---

## Overview

This project implements an end-to-end reinforcement learning framework for dynamic option hedging.

The framework includes:

- Black-Scholes option pricing
- Greeks computation
- Geometric Brownian Motion (GBM) market simulator
- Portfolio and transaction cost modeling
- Custom Gymnasium environment
- DDPG reinforcement learning agent
- Classical Delta Hedging baseline
- Evaluation framework with multiple performance metrics
- Automated visualization and result generation

---

## Features

### Financial Models

- European Call and Put Options
- Black-Scholes Pricing
- Greeks
  - Delta
  - Gamma
  - Vega
  - Theta
  - Rho

### Market Simulation

- Geometric Brownian Motion (GBM)
- Configurable volatility
- Configurable interest rate
- Randomized market scenarios
- Multiple simulation paths

### Portfolio Management

- Cash account
- Stock inventory
- Short option position
- Portfolio valuation
- Transaction costs

### Reinforcement Learning

- Gymnasium Environment
- Continuous Action Space
- Reward Shaping
- Observation Normalization
- Stable-Baselines3 DDPG

### Hedging Strategies

- Random Agent
- Black-Scholes Delta Hedging
- Deep Reinforcement Learning (DDPG)

### Evaluation Metrics

- Average Reward
- Portfolio Value
- Transaction Cost
- Mean Hedging Error
- RMSE
- Standard Deviation
- Maximum Hedging Error

---

## Project Structure

```text
derivative-hedging-rl/
│
├── configs/
├── data/
├── docs/
├── notebooks/
├── outputs/
│   ├── figures/
│   ├── models/
│   └── results/
│
├── scripts/
│   ├── train_ddpg.py
│   ├── compare_agents.py
│   └── plot_results.py
│
├── src/
│   └── hedging/
│       ├── baseline/
│       ├── environment/
│       ├── evaluation/
│       ├── pricing/
│       └── simulation/
│
├── tests/
│
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/<your-username>/derivative-hedging-rl.git

cd derivative-hedging-rl
```

Create a virtual environment

```bash
python -m venv .venv
```

Activate the environment

Windows

```bash
.venv\Scripts\activate
```

Linux/macOS

```bash
source .venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Training

Train the DDPG agent

```bash
python -m scripts.train_ddpg
```

The trained model is saved to

```
outputs/models/
```

---

## Evaluation

Compare all hedging strategies

```bash
python -m scripts.compare_agents
```

Results are stored in

```
outputs/results/comparison.csv
```

---

## Generate Plots

```bash
python -m scripts.plot_results
```

Plots are saved in

```
outputs/figures/
```

---

## Testing

Run all unit tests

```bash
pytest
```

Current Status

```
18 / 18 Tests Passed
```

---

## Experimental Results

| Agent | Reward | Portfolio | Transaction Cost | Mean Error | RMSE |
|-------|--------:|----------:|-----------------:|-----------:|------:|
| Random | -1,185,554 | -89.64 | 89.96 | 50.92 | 68.59 |
| Delta Hedging | -764,844 | 28.37 | 2.75 | 35.05 | 55.09 |
| **DDPG (Proposed)** | **-702,970** | 7.48 | 8.44 | 38.38 | **52.82** |

### Key Findings

- DDPG achieved the **highest cumulative reward**.
- DDPG achieved the **lowest RMSE** among all evaluated strategies.
- Delta Hedging achieved the lowest transaction cost.
- Reward shaping and observation normalization significantly improved RL performance over the initial DDPG implementation.

---

## Technologies Used

- Python
- NumPy
- SciPy
- Gymnasium
- Stable-Baselines3
- PyTorch
- Matplotlib
- Pytest

---

## License

This project is licensed under the MIT License.

---
