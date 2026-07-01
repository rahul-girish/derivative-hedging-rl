# Intelligent Hedging System with Reinforcement Learning

## 1. Purpose

Build a dynamic hedging engine for derivative positions that learns from market state, transaction costs, and risk preferences instead of relying only on static delta hedging.

The design below combines:

- **Deep reinforcement learning for continuous hedging**
- **Contextual bandits for sample-efficient online adaptation**
- **A bio-inspired optimizer (Particle Swarm Optimization) for offline tuning**
- **A full simulator/backtester that can run on synthetic or historical data**
- **Risk controls, monitoring, and reproducible experiment management**

The target use case is a system that can decide **how much of the underlying to hold at each rebalance time** so the total hedging objective is minimized.

---

## 2. What the reference papers contribute

### Paper A: *Deep Hedging of Derivatives Using Reinforcement Learning* (Cao, Chen, Hull, Poulos)
Core ideas to keep in the implementation:

- Hedging is formulated as an RL problem under **transaction costs**.
- The paper compares **accounting P&L** versus **cash-flow** reward formulations.
- It uses **deep deterministic policy gradients (DPG / DDPG-style continuous control)** so hedge actions can be continuous.
- It introduces **two Q-functions** to track both the expected cost and the expected squared cost, which supports a mean-plus-standard-deviation style objective.
- It uses **prioritized experience replay** to improve learning speed and data efficiency.
- It finds that a **hybrid accounting P&L approach** with a simpler valuation model can work better than pure cash-flow hedging.

### Paper B: *Hedging using reinforcement learning: Contextual k-armed bandit versus Q-learning* (Cannelli, Nuti, Sala, Szehr)
Core ideas to keep in the implementation:

- Hedging is treated as a **risk-averse contextual bandit** problem.
- The motivation is **simplicity and sample efficiency**, especially when real market data is limited.
- The paper argues that simulated training can overfit the simulator, so **online adaptation** matters.
- The bandit view naturally fits a **Profit and Loss** hedging formulation.
- In low-data or high-adaptation settings, the bandit approach can be preferable to full Q-learning.

These two papers suggest a practical architecture: **use deep RL for richer offline policy learning and use a bandit layer for fast online adaptation**.

---

## 3. System architecture overview

### High-level flow

1. Ingest market and contract data.
2. Build a market state vector.
3. Step through a simulator or historical replay environment.
4. A hedging policy outputs a target hedge.
5. The environment computes execution, costs, and reward.
6. The learning module updates actor/critic or bandit statistics.
7. A risk engine checks exposure and hard limits.
8. A tuner searches for better hyperparameters and objective weights.
9. Results go to evaluation, reporting, and deployment.

### Core components

- `data/`
- `features/`
- `simulation/`
- `envs/`
- `agents/`
- `risk/`
- `tuning/`
- `evaluation/`
- `execution/`
- `orchestration/`
- `configs/`
- `tests/`

---

## 4. Recommended architecture in detail

## 4.1 Data ingestion layer

### Responsibilities
- Load historical spot prices, returns, volumes, option chains, and interest rates.
- Load contract specifications: strike, maturity, call/put, multiplier, dividend assumptions.
- Load transaction cost model parameters.
- Load optional implied volatility surface data.
- Normalize timestamps and align all sources.

### Implementation notes
Use a clean adapter interface so the same pipeline works for:
- CSV / Parquet files
- SQL tables
- market data APIs
- synthetic simulator output

### Example module boundaries

```python
class MarketDataSource(Protocol):
    def get_prices(self, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame: ...
    def get_option_chain(self, symbol: str, asof: pd.Timestamp) -> pd.DataFrame: ...
    def get_rates(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series: ...
```

### Best practice
Keep raw data immutable. All cleaning should create versioned processed datasets.

---

## 4.2 Feature engineering layer

The policy should not see only the spot price. It should see a compact but informative state.

### Suggested state variables
At each decision time `t`, include:

- current underlying price `S_t`
- log return(s)
- time to maturity
- realized volatility over multiple windows
- implied volatility if available
- option Greeks if available from a pricing model
- current hedge inventory
- cash balance
- funding rate
- transaction cost regime
- recent executed trade size
- bid-ask spread proxy
- volatility regime indicators
- regime probability from a hidden-state model if used
- risk limits remaining

### Feature design principles
- Avoid leakage: only use information available at decision time.
- Scale features consistently.
- Use both raw and derived state variables.
- Make the state Markov-like if possible.

### Suggested transforms
- `log(S_t / S_{t-1})`
- rolling mean/variance of returns
- exponential moving volatility
- moneyness `S_t / K`
- normalized time to maturity `tau / T`
- delta, gamma, vega from a simple pricing engine
- inventory normalized by maximum allowed inventory

---

## 4.3 Market simulator / environment

This is the most important module.

### Responsibilities
- Simulate or replay the market path.
- Apply trade execution and transaction costs.
- Update cash, inventory, and option liability.
- Emit reward.
- Terminate at maturity.
- Support training, validation, and stress testing.

### Required properties
The environment should support at least:

1. **Geometric Brownian motion (GBM)**  
   Good baseline and matches the first paper’s simplest case.

2. **Stochastic volatility**  
   Needed because the first paper explicitly compares GBM and stochastic volatility.

3. **Historical replay**  
   Useful for offline validation.

4. **Hybrid mode**  
   Use a simpler pricing model inside the reward calculation than the true market simulator when needed, mirroring the hybrid idea in the first paper.

5. **Market frictions**
   - proportional transaction costs
   - fixed fees
   - slippage
   - optional market impact
   - discrete trading grid
   - latency delay

### Environment state transition
At each step:

- observe state `s_t`
- choose action `a_t` = target hedge or hedge adjustment
- compute executed trade `Δh_t`
- update inventory and cash
- evolve underlying to `S_{t+1}`
- compute option mark-to-market or terminal payoff
- calculate reward `r_{t+1}`
- check terminal conditions

### Environment interface

```python
class HedgingEnv:
    def reset(self, episode_seed: int | None = None) -> State: ...
    def step(self, action: np.ndarray) -> tuple[State, float, bool, dict]: ...
```

### Important design choice: action semantics

Use one of these:

- **Target hedge ratio**: action is desired holding in the underlying.
- **Trade size**: action is how much to buy/sell now.
- **Delta adjustment**: action is increment to current hedge.

For practical implementation, **target hedge ratio** is easiest to reason about and safest to constrain.

---

## 4.4 Reward design

The reward is the heart of the project.

### Base idea
Reward should be the negative hedging objective. Typical objectives:

- minimize replication error
- minimize transaction costs
- minimize variance of hedging cost
- penalize tail loss
- penalize inventory changes or turnover

### A strong default reward
A useful starting point is:

\[
r_{t+1} = - \Big(
\text{hedging\_pnl\_error}
+ \lambda_c \cdot \text{transaction\_cost}
+ \lambda_v \cdot \text{risk\_penalty}
+ \lambda_t \cdot \text{turnover\_penalty}
\Big)
\]

### Practical reward variants
1. **Accounting P&L reward**  
   Matches the first paper’s better-performing formulation.  
   Use period-by-period P&L, including derivative revaluation.

2. **Cash-flow reward**  
   Easier if you do not want to assume a pricing model.  
   More delayed credit assignment, harder to learn.

3. **Risk-averse mean-variance reward**  
   Reward is based on mean cost plus a multiple of standard deviation.

4. **Tail-risk reward**  
   Use CVaR or quantile loss if you care about extreme hedging losses.

### Recommendation
Implement all four, but make **accounting P&L + risk penalty** the default training objective.

---

## 4.5 Baseline hedgers

Do not build RL first. Build baselines first.

### Baseline 1: static delta hedging
- Compute delta from a pricing model.
- Rebalance at fixed intervals.
- Use as the main benchmark.

### Baseline 2: threshold delta hedging
- Rebalance only when delta drift exceeds a threshold.
- Useful when transaction costs are high.

### Baseline 3: no hedge
- Sanity check only.

### Baseline 4: rule-based volatility-adjusted hedge
- Delta hedge plus a volatility regime correction.
- Useful as a simple non-RL reference.

### Why this matters
Without baselines, it is impossible to know whether RL improves anything or merely learns a noisy version of delta hedging.

---

## 5. Reinforcement learning design

## 5.1 RL problem formulation

### State `s_t`
A tensor containing market, contract, portfolio, and risk information.

### Action `a_t`
Continuous hedge size or hedge ratio.

### Reward `r_{t+1}`
Negative objective from hedging performance.

### Episode
One option lifecycle from inception to maturity, or a rolling horizon window.

### Discount factor
For hedging, the discount can often be set near 1, but keep it configurable.

---

## 5.2 Deep RL agent

### Recommended default: Actor-Critic with continuous actions
Use a DDPG-style implementation or a modern alternative such as TD3 or SAC.

### Why
- hedge ratios are naturally continuous
- discrete action grids introduce unnecessary approximation error
- transaction costs and risk penalties are better handled with continuous control

### Components
- **Actor network**: outputs hedge ratio
- **Critic network**: estimates action value
- **Target actor / target critic**: stabilize training
- **Replay buffer**: store transitions
- **Prioritized replay**: sample important experiences more often

### Paper-aligned implementation detail
The first paper uses **two Q-functions** to track both the expected cost and the expected squared cost. That means you can implement:

- `Q1(s, a)`: expected hedging cost
- `Q2(s, a)`: expected squared hedging cost

Then form a risk-aware objective from mean and variance:

\[
\text{Var}(C) = E[C^2] - (E[C])^2
\]

This is a clean way to encode the paper’s mean-plus-standard-deviation objective.

### Actor objective
The actor should maximize:

\[
J = - \big( E[C] + \lambda \cdot \sqrt{E[C^2] - E[C]^2} \big)
\]

### Critic loss
For each sampled transition:
- update `Q1` toward observed cost target
- update `Q2` toward squared-cost target

### Practical note
If you use SAC or TD3 instead of DDPG, keep the same reward design. The core architecture stays the same.

---

## 5.3 Contextual bandit module

The second paper argues that a contextual bandit is a very strong fit for P&L hedging when data is scarce and online adaptation matters.

### When to use it
- live deployment
- low data availability
- fast retraining requirement
- frequent regime changes
- simplified action sets
- online calibration

### Bandit formulation
At each time step:
- observe context `x_t`
- choose an action `a_t`
- receive immediate reward `r_t`

### Risk-averse contextual bandit
Use a mean-variance or risk-adjusted reward:

\[
U(a) = \mu(a|x) - \beta \sigma(a|x)
\]

where:
- `mu` is estimated mean reward
- `sigma` is estimated reward volatility
- `beta` controls risk aversion

### Useful implementations
- linear contextual bandit
- generalized linear bandit
- neural contextual bandit
- ensemble bandit with uncertainty estimates

### Recommended design
Maintain a **bandit head** on top of the same feature encoder used by the RL agent.

That gives you:
- fast online adaptation
- cheap updates
- compatibility with the same state representation

---

## 5.4 Hybrid policy

A very practical production design is:

- **offline deep RL** learns a robust baseline policy
- **online contextual bandit** nudges the action to adapt to current regime

### Example hybrid flow
1. Actor proposes target hedge ratio.
2. Bandit layer adjusts it based on recent realized P&L.
3. Risk engine clamps the result inside allowed bounds.
4. Execution engine sends the final trade.

### Why this is useful
- deep RL captures long-horizon behavior
- bandit adapts quickly to live changes
- the combined system is more realistic than a single monolithic model

---

## 6. Bio-inspired optimization module

Use **Particle Swarm Optimization (PSO)** as the required bio-inspired algorithm.

PSO is a swarm-intelligence method where candidate solutions move through the search space by sharing information about good positions found so far. It is especially useful for hyperparameter tuning and calibration because it handles nonconvex search spaces well and requires relatively little problem-specific gradient information.

## 6.1 Where PSO fits

Use PSO to optimize:

- RL learning rate
- replay buffer size
- prioritized replay exponent
- actor/critic layer widths
- entropy coefficient if using SAC
- transaction-cost penalties
- reward weights
- rebalance frequency
- lookback window lengths
- bandit exploration parameters
- simulator parameters for calibration
- action clipping bounds
- risk-aversion coefficient

## 6.2 PSO particle design

Each particle is one candidate configuration.

Example vector:

```text
[lr_actor, lr_critic, tau, gamma, batch_size, lambda_cost, lambda_risk, 
 hidden_size_1, hidden_size_2, replay_alpha, replay_beta, rebalance_interval]
```

### Particle fitness
Define fitness on a validation set as:

\[
\text{fitness} = 
w_1 \cdot \text{mean hedging error}
+ w_2 \cdot \text{variance}
+ w_3 \cdot \text{transaction cost}
+ w_4 \cdot \text{max drawdown}
+ w_5 \cdot \text{turnover}
\]

Lower fitness is better.

## 6.3 PSO loop
- initialize swarm
- evaluate each particle by training a short model or using a proxy score
- update particle velocity and position
- retain personal best and global best
- stop after budget or convergence

## 6.4 Why PSO is a good fit here
- objective is expensive and nonconvex
- derivatives of the validation score are unavailable
- many hyperparameters interact
- a small population often finds useful settings quickly

## 6.5 Optional alternative: Spider Monkey Optimization
If you want a second bio-inspired option, Spider Monkey Optimization can be used for the same tuning jobs. It is a swarm-inspired metaheuristic that can be dropped into the same tuning interface as PSO. Keep the interface abstract so either algorithm can be swapped in.

---

## 7. Suggested codebase structure

```text
hedging_project/
├─ configs/
│  ├─ env.yaml
│  ├─ rl.yaml
│  ├─ bandit.yaml
│  ├─ pso.yaml
│  └─ risk.yaml
├─ data/
│  ├─ raw/
│  ├─ processed/
│  └─ features/
├─ src/
│  ├─ data/
│  │  ├─ loaders.py
│  │  ├─ cleaners.py
│  │  └─ calendar.py
│  ├─ features/
│  │  ├─ engineering.py
│  │  ├─ normalization.py
│  │  └─ regime.py
│  ├─ simulation/
│  │  ├─ price_processes.py
│  │  ├─ execution.py
│  │  ├─ costs.py
│  │  └─ pricing.py
│  ├─ envs/
│  │  └─ hedging_env.py
│  ├─ agents/
│  │  ├─ actor.py
│  │  ├─ critic.py
│  │  ├─ replay_buffer.py
│  │  ├─ dpg_agent.py
│  │  ├─ bandit_agent.py
│  │  └─ hybrid_agent.py
│  ├─ risk/
│  │  ├─ metrics.py
│  │  ├─ constraints.py
│  │  └─ stress.py
│  ├─ tuning/
│  │  ├─ pso.py
│  │  └─ objective.py
│  ├─ evaluation/
│  │  ├─ backtest.py
│  │  ├─ report.py
│  │  └─ benchmarks.py
│  ├─ deployment/
│  │  ├─ policy_service.py
│  │  ├─ execution_router.py
│  │  └─ monitoring.py
│  └─ utils/
│     ├─ logging.py
│     ├─ seeds.py
│     └─ serialization.py
├─ notebooks/
├─ tests/
├─ scripts/
└─ README.md
```

---

## 8. Core class design

## 8.1 Environment

```python
@dataclass
class HedgingState:
    spot: float
    time_to_maturity: float
    inventory: float
    cash: float
    realized_vol: float
    implied_vol: float | None
    delta: float
    gamma: float
    vega: float
    spread: float
    regime_id: int

class HedgingEnv:
    def reset(self, seed: int | None = None) -> HedgingState: ...
    def step(self, action: float) -> tuple[HedgingState, float, bool, dict]: ...
```

## 8.2 Actor network

```python
class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)
```

Scale the `tanh` output to the legal hedge range.

## 8.3 Critic network

```python
class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        return self.net(x)
```

## 8.4 Prioritized replay buffer

Store:
- state
- action
- reward
- next state
- done
- importance weight
- priority

Key requirements:
- fast sampling
- update priorities after critic loss computation
- support episode grouping if needed

## 8.5 Bandit agent

Keep the bandit simple and interpretable.

```python
class ContextualBanditAgent:
    def predict(self, context: np.ndarray) -> float: ...
    def update(self, context: np.ndarray, action: float, reward: float) -> None: ...
```

Recommended implementations:
- linear model with ridge regression
- Bayesian linear regression
- neural contextual bandit with uncertainty head

## 8.6 Hybrid agent

```python
class HybridHedgingAgent:
    def act(self, state):
        base_action = self.rl_agent.act(state)
        adjustment = self.bandit_agent.predict(state.features)
        action = base_action + adjustment
        return self.risk_engine.clamp(action)
```

---

## 9. Training pipeline

## Stage 1: data preparation
- clean and align market data
- create feature windows
- split train/validation/test by time
- build volatility regimes
- estimate cost model

## Stage 2: simulator calibration
- fit GBM or stochastic volatility parameters
- validate price path distributions
- calibrate transaction cost assumptions
- tune slippage and spread proxies

## Stage 3: baseline backtest
- run delta hedging
- run threshold hedging
- measure objective values
- confirm environment correctness

## Stage 4: RL training
- initialize actor and critic
- collect trajectories
- compute reward
- sample prioritized replay buffer
- update networks
- periodically evaluate on validation paths
- save best model by validation objective

## Stage 5: bandit training
- run online or semi-online updates
- feed real market contexts
- update quickly after each episode or each trading day

## Stage 6: PSO tuning
- choose tunable parameters
- define objective on validation backtest
- run swarm search
- retrain final model with best settings

## Stage 7: final evaluation
- compare against baselines
- test under stress
- assess generalization across regimes
- freeze final policy

---

## 10. Pseudocode for the RL loop

```python
state = env.reset()
for episode in range(num_episodes):
    state = env.reset(seed=episode)
    done = False

    while not done:
        action = actor(state_tensor)
        next_state, reward, done, info = env.step(action)

        replay.add(state, action, reward, next_state, done)

        if len(replay) > warmup:
            batch = replay.sample()
            critic_loss = compute_critic_loss(batch)
            actor_loss = compute_actor_loss(batch)
            optimize(critic, critic_loss)
            optimize(actor, actor_loss)
            update_target_networks()

        state = next_state
```

For the paper-aligned version:
- maintain `Q1` and `Q2`
- derive mean and variance estimates from them
- optimize actor against the risk-adjusted objective

---

## 11. Evaluation metrics

Measure more than reward.

### Core metrics
- mean hedging error
- standard deviation of hedging error
- RMSE of replication error
- transaction cost
- turnover
- max drawdown
- CVaR of terminal hedging loss
- average inventory exposure
- training stability
- sample efficiency
- out-of-sample performance

### Benchmark comparisons
- delta hedging
- threshold rebalancing
- no hedge
- RL-only
- bandit-only
- hybrid RL + bandit

### Stress tests
- volatility spike
- jump shock
- widening spreads
- delayed execution
- regime shift
- sudden liquidity drop
- miscalibrated simulator

### Acceptance criteria
A model is only accepted if it:
- beats baseline delta hedging on out-of-sample objective
- maintains acceptable tail risk
- remains stable under stress
- does not explode in turnover
- respects hard risk limits

---

## 12. Risk controls

### Hard constraints
- maximum hedge size
- minimum cash reserve
- maximum daily turnover
- stop-loss threshold
- max drawdown
- position limits

### Soft penalties
- inventory penalty
- turnover penalty
- slippage penalty
- tail-risk penalty

### Kill switches
A production policy should immediately fall back to a baseline if:
- action distribution shifts sharply
- live slippage exceeds threshold
- reward drops below control limit
- data feed becomes unreliable

---

## 13. Testing strategy

### Unit tests
- reward calculation
- transaction cost calculation
- state normalization
- action clipping
- option payoff logic
- replay sampling

### Integration tests
- environment step consistency
- terminal state correctness
- backtest loop
- model save/load
- live policy inference

### Statistical tests
- path distribution sanity checks
- reward distribution checks
- stability across random seeds
- calibration sanity checks

### Regression tests
- if a code change improves one metric but breaks another, lock that in CI.

---

## 14. Deployment design

### Offline research mode
- train from historical and synthetic data
- sweep hyperparameters
- validate experiments

### Paper-trading mode
- consume live market data
- produce actions but do not execute
- compare to benchmark hedge

### Live execution mode
- generate hedge target
- pass through risk engine
- send orders through execution adapter
- monitor fills and slippage
- update bandit online

### Monitoring
Track:
- realized P&L
- hedge error
- transaction cost
- fill quality
- model confidence
- regime shifts
- latency

---

## 15. Practical implementation decisions

### Recommended stack
- Python
- PyTorch
- NumPy / pandas
- scikit-learn for bandit baselines and calibration
- Hydra or Pydantic for configuration
- MLflow or Weights & Biases for experiments
- FastAPI for live policy serving if needed

### Determinism
Set:
- random seeds
- torch deterministic flags
- versioned datasets
- frozen configs

### Numerical stability
- normalize inputs
- clip actions
- use gradient clipping
- avoid overly aggressive risk weights
- log all losses and advantages

### Performance
- vectorize simulations
- batch environments where possible
- cache pricing outputs
- avoid recomputing Greeks unnecessarily

---

## 16. Common failure modes

1. **Overfitting to simulator**
   - fix with regime randomization and historical replay

2. **Exploding turnover**
   - fix with turnover penalty and action smoothing

3. **Unstable learning**
   - fix with prioritized replay, target networks, gradient clipping

4. **Reward leakage**
   - check that future information is never used in state features

5. **Bad calibration**
   - fix simulator first, not RL last

6. **Poor live adaptation**
   - add contextual bandit layer and online updates

7. **Tail risk hidden by mean reward**
   - use CVaR or variance penalty

---

## 17. Build order

A sane implementation order is:

1. Build the data loader.
2. Build the simulator.
3. Build the delta hedging baseline.
4. Build the backtester.
5. Build the DDPG-style agent.
6. Add prioritized replay.
7. Add the bandit module.
8. Add the PSO tuner.
9. Add stress testing.
10. Add monitoring and deployment hooks.

Do not start with neural networks before the environment is correct.

---

## 18. Minimal deliverable definition

The project is complete only when all of the following exist:

- reproducible data pipeline
- calibrated simulator
- delta hedge baseline
- RL agent with continuous actions
- contextual bandit agent
- PSO tuner
- validation and stress test reports
- documented config files
- model save/load
- human-readable evaluation summary

---

## 19. Suggested APIs

### `train_rl.py`
Trains the deep RL hedger.

### `train_bandit.py`
Trains the contextual bandit or online updater.

### `tune_pso.py`
Searches over hyperparameters and reward weights.

### `backtest.py`
Runs historical evaluation and benchmark comparisons.

### `simulate.py`
Generates synthetic scenarios for stress testing.

### `serve_policy.py`
Loads a frozen model and exposes inference.

---

## 20. Recommended repository conventions

- Keep one config per experiment.
- Log every run with a unique ID.
- Save both model weights and exact config.
- Save validation plots and metric tables.
- Never mix train/test time windows.
- Version the simulator itself.

---

## 21. Example experiment matrix

Try at least these combinations:

| Model | Reward | Action | Online adaptation |
|---|---|---|---|
| Delta baseline | N/A | rule-based | no |
| DDPG | accounting P&L | continuous | no |
| DDPG + PER | accounting P&L | continuous | no |
| Bandit | mean-variance P&L | discrete or continuous | yes |
| Hybrid | accounting P&L + risk penalty | continuous + bandit correction | yes |

---

## 22. Final recommendation

The best engineering choice is:

- **offline**: train a continuous-control RL hedger with prioritized replay and accounting P&L reward
- **online**: use a contextual bandit to adapt quickly to live regime shifts
- **tuning**: use Particle Swarm Optimization to search reward weights, hyperparameters, and simulator calibration settings
- **deployment**: keep a delta-hedging fallback and strict risk clamps

That combination is closest to the papers’ message while still being practical in code.

---

## 23. References

1. Cao, J., Chen, J., Hull, J., & Poulos, Z. *Deep Hedging of Derivatives Using Reinforcement Learning*.
2. Cannelli, L., Nuti, G., Sala, M., & Szehr, O. *Hedging using reinforcement learning: Contextual k-armed bandit versus Q-learning*.

