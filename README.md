#  Derivative Hedging using Deep Reinforcement Learning

A Reinforcement Learning framework for dynamic option hedging using **Deep Deterministic Policy Gradient (DDPG)**. The project compares a learned hedging strategy against **Random Hedging** and the classical **Black–Scholes Delta Hedging** strategy using a realistic market simulation.

---

##  Overview

Dynamic option hedging aims to minimize the risk of holding derivative contracts by continuously adjusting the hedge position as market conditions evolve.

Traditional approaches rely on the **Black–Scholes Delta Hedging** strategy, which assumes perfect market conditions and continuous rebalancing. In practice, transaction costs, market uncertainty, and discrete trading reduce its effectiveness.

This project formulates option hedging as a **Reinforcement Learning** problem, where a DDPG agent learns an optimal hedging policy by interacting with a simulated financial market.

---

##  Features

-  Geometric Brownian Motion (GBM) stock price simulation
-  Black–Scholes option pricing
-  Analytical Greeks (Delta)
-  Portfolio simulation engine
-  Transaction cost modeling
-  Gymnasium-based Reinforcement Learning environment
-  Deep Deterministic Policy Gradient (DDPG)
-  Random Hedging baseline
-  Classical Black–Scholes Delta Hedging baseline
-  Comprehensive evaluation framework
-  Automatic CSV result generation
-  Publication-quality evaluation plots
-  Interactive Streamlit dashboard
-  Unit tested using PyTest

---

# Project Architecture

```
                +-----------------------+
                |  GBM Price Simulator  |
                +-----------+-----------+
                            |
                            |
                +-----------v-----------+
                |  Hedging Environment  |
                +-----------+-----------+
                            |
          +-----------------+-----------------+
          |                                   |
          |                                   |
+---------v---------+               +---------v---------+
|   Delta Hedging   |               |    DDPG Agent     |
+---------+---------+               +---------+---------+
          |                                   |
          +-----------------+-----------------+
                            |
                            |
                +-----------v-----------+
                | Evaluation Framework  |
                +-----------+-----------+
                            |
          +-----------------+-----------------+
          |                                   |
          |                                   |
+---------v---------+               +---------v---------+
| CSV Results       |               | Streamlit Dashboard|
+-------------------+               +-------------------+
```

---

# Repository Structure

```
derivative-hedging-rl/

├── dashboard/
│   ├── app.py
│   └── pages/
│
├── data/
│
├── outputs/
│   ├── figures/
│   ├── logs/
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
│       ├── environment/
│       ├── evaluation/
│       ├── pricing/
│       └── simulation/
│
├── tests/
│
├── README.md
└── pyproject.toml
```

---

# Installation

Clone the repository

```bash
git clone https://github.com/rahul-girish/derivative-hedging-rl.git
cd derivative-hedging-rl
```

Create a virtual environment

```bash
python -m venv .venv
```

Activate

Windows

```bash
.venv\Scripts\activate
```

Linux / macOS

```bash
source .venv/bin/activate
```

Install dependencies

```bash
pip install -e .
```

---

# Training the DDPG Agent

```bash
python -m scripts.train_ddpg
```

The trained model will be saved in

```
outputs/models/
```

---

# Evaluating Agents

Run

```bash
python -m scripts.compare_agents
```

This compares

- Random Hedging
- Black–Scholes Delta Hedging
- DDPG Hedging

Results are automatically stored in

```
outputs/results/comparison.csv
```

---

# Generating Evaluation Plots

```bash
python -m scripts.plot_results
```

Generated figures include

- Reward Comparison
- Portfolio Value
- Transaction Cost
- Mean Hedging Error
- RMSE
- Summary Figure

Saved in

```
outputs/figures/
```

---

# Streamlit Dashboard

Launch the interactive dashboard

```bash
streamlit run dashboard/app.py
```

The dashboard contains

###  Home

Project overview and KPI metrics.

###  Agent Comparison

Interactive Plotly comparison between

- Random
- Delta
- DDPG

###  Generated Figures

Automatically displays generated evaluation figures.

###  Environment Explorer

Interactive controls for

- Strike Price
- Volatility
- Interest Rate
- Maturity
- Initial Stock Price

###  Model Information

Displays

- DDPG architecture
- Hyperparameters
- Replay buffer
- Learning rate
- Training configuration

###  Performance Summary

Radar chart comparing all agents.

###  About

Financial theory including

- Black–Scholes
- Delta Hedging
- Geometric Brownian Motion
- Reinforcement Learning

---

# Evaluation Metrics

The framework evaluates

- Average Reward
- Portfolio Value
- Transaction Cost
- Mean Hedging Error
- RMSE
- Standard Deviation
- Maximum Hedging Error

---

# Example Results

| Agent | Reward | Portfolio | Cost | RMSE |
|-------|--------:|----------:|-----:|-----:|
| Random | -1,185,554 | -89.64 | 89.96 | 68.59 |
| Delta | -764,844 | 28.37 | 2.75 | 55.09 |
| DDPG | **-702,970** | 7.48 | 8.44 | **52.82** |

The trained DDPG agent achieved the best overall hedging performance by minimizing hedging error while maintaining relatively low transaction costs.

---

# Testing

Run all tests

```bash
pytest
```

---

# Technologies Used

- Python
- NumPy
- SciPy
- Pandas
- Gymnasium
- Stable-Baselines3
- PyTorch
- Streamlit
- Plotly
- Matplotlib
- PyTest

---

This project is intended for academic and educational purposes.
