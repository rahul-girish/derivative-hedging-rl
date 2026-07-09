# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

A research framework for RL-based derivative hedging under transaction costs, benchmarked against classical delta hedging. Based on two papers: *Deep Hedging of Derivatives Using Reinforcement Learning* (Cao et al.) and *Hedging using RL: Contextual Bandits vs Q-learning* (Cannelli et al.). `PLAN.md` contains the full design document (architecture, reward design, RL formulation, build order) — consult it before implementing any new module, as it specifies the intended design in detail.

## Commands

```bash
pip install -r requirements.txt        # install deps (Python >= 3.11 required)

pytest                                 # run all tests (testpaths=tests, pythonpath=src via pyproject.toml)
pytest tests/pricing/test_greeks.py    # run one test file
pytest tests/pricing/test_greeks.py::test_call_delta   # run one test

black src tests scripts                # format (line length 88)
ruff check src tests scripts           # lint (rules: E, F, I, UP, B)

python scripts/smoke_test.py           # quick end-to-end sanity check
python scripts/delta_hedge_demo.py     # full delta-hedging backtest demo
```

Scripts and tests import `hedging` directly; pytest gets this via `pythonpath = ["src"]`, but running scripts requires `src` on `PYTHONPATH` or an editable install (`pip install -e .`).

## Architecture

All code lives under `src/hedging/`. **Many modules are empty placeholder files** (environment/, agents/, replay/, tuning/, data/, baseline/threshold.py, baseline/no_hedge.py, all configs/*.yaml, scripts under src/hedging/scripts/). They mark the planned structure from PLAN.md, not working code — check file contents before assuming a module exists.

Implemented so far (Phase 1 of the roadmap), forming a pipeline that the delta-hedge backtest exercises end to end:

- **pricing/** — `OptionContract` (frozen dataclass; `maturity` is time-to-expiry in years), Black-Scholes pricing, delta/gamma greeks. Time decay is modeled by constructing a *new* contract with reduced maturity each step, since contracts are immutable.
- **simulation/** — `MarketSimulator` ABC → `GBMSimulator` produces a `MarketPath` (prices shaped `(n_paths, n_steps+1)`, dt, time_grid). `Portfolio` tracks cash/shares/short-option position (defaults to short 1 option). `TransactionCostModel` (proportional) feeds `ExecutionEngine.rebalance()`, which trades a portfolio to a target share count and returns the cost incurred.
- **baseline/** — `DeltaHedger.target_position()` returns BS delta; any hedger exposing this method plugs into the backtester.
- **evaluation/** — `Backtester.run(path, contract, hedger, volatility, rate)` steps through a single price path: sells the option at t=0, re-prices it each step with remaining maturity, rebalances to the hedger's target, and records everything in `BacktestResult`.

Key conventions established by the existing code:

- Hedger strategy interface is duck-typed: `target_position(contract, stock_price, volatility, rate) -> float` (target shares held).
- Dataclasses use `slots=True`; validation happens in `__post_init__` with `ValueError`.
- All modules use `from __future__ import annotations` and absolute imports rooted at `hedging.`.
- Simulators take `seed` for reproducibility via `np.random.default_rng`.
- New simulators (Heston, SABR, etc.) must subclass `MarketSimulator` and return `MarketPath` so they are drop-in replacements for GBM.

## Workflow

Per README: every feature follows implement → test → refactor → document → commit; a feature is not complete until unit tests pass. Tests mirror the source layout (`tests/pricing/`, `tests/simulation/`). Build order per PLAN.md: environment/baselines must be correct before RL agents are added; RL agents will be evaluated *against* the baselines using this same backtester.
