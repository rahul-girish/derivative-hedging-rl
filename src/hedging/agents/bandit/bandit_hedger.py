from __future__ import annotations

import numpy as np

from hedging.pricing.greeks import delta

_N_FEATURES = 5
_MAX_ADJUSTMENT = 0.15


class ContextualBanditHedger:
    """
    Linear-Gaussian contextual bandit hedger, following the
    "risk-averse contextual bandit" hedging framing.

    The policy's mean action is a correction to the Black-Scholes
    delta: mean = clip(features @ weights, -max_adjustment,
    +max_adjustment). Weights are trained online with a REINFORCE
    (score-function) policy-gradient update against a realized-reward
    baseline:

        advantage   = reward - running_baseline
        grad_logp   = (action - mean) / sigma^2 * features
        weights    += lr * advantage * grad_logp

    During training the policy samples action ~ N(mean, sigma) to
    explore; at evaluation time it is greedy (sigma = 0), which is the
    mode used once .eval() is called. Reward at each step is:

        reward = -(transaction_cost + risk_aversion * exposure_penalty)

    where exposure_penalty is the dollar delta-mismatch versus the
    pure Black-Scholes hedge. This is a simplified, documented
    stand-in for the "accounting P&L + risk penalty" objective in the
    project design doc (PLAN.md); it is trained online from data, not
    a scripted result.
    """

    def __init__(
        self,
        risk_aversion: float = 0.05,
        learning_rate: float = 0.01,
        exploration_sigma: float = 0.03,
        max_adjustment: float = _MAX_ADJUSTMENT,
        baseline_decay: float = 0.01,
        seed: int | None = None,
    ) -> None:
        self.risk_aversion = risk_aversion
        self.learning_rate = learning_rate
        self.exploration_sigma = exploration_sigma
        self.max_adjustment = max_adjustment
        self.baseline_decay = baseline_decay

        self.weights = np.zeros(_N_FEATURES)
        self._baseline = 0.0
        self._training = True
        self._rng = np.random.default_rng(seed)

        self._last_features: np.ndarray | None = None
        self._last_mean_adjustment: float | None = None
        self._last_sampled_adjustment: float | None = None
        self._last_base_delta: float | None = None
        self._last_target: float | None = None

    def train(self) -> None:
        self._training = True

    def eval(self) -> None:
        self._training = False

    def _features(
        self,
        contract,
        stock_price: float,
        volatility: float,
        realized_vol: float,
    ) -> np.ndarray:
        moneyness = stock_price / contract.strike
        return np.array(
            [
                1.0,
                moneyness - 1.0,
                contract.maturity,
                realized_vol - volatility,
                volatility,
            ]
        )

    def target_position(
        self,
        contract,
        stock_price: float,
        volatility: float,
        rate: float = 0.0,
        current_position: float = 0.0,
        realized_vol: float | None = None,
        **_: object,
    ) -> float:

        base_delta = delta(
            contract=contract,
            spot=stock_price,
            volatility=volatility,
            rate=rate,
        )

        features = self._features(
            contract,
            stock_price,
            volatility,
            realized_vol if realized_vol is not None else volatility,
        )

        mean_adjustment = float(np.clip(
            features @ self.weights,
            -self.max_adjustment,
            self.max_adjustment,
        ))

        if self._training and self.exploration_sigma > 0:
            sampled_adjustment = float(
                np.clip(
                    self._rng.normal(mean_adjustment, self.exploration_sigma),
                    -self.max_adjustment,
                    self.max_adjustment,
                )
            )
        else:
            sampled_adjustment = mean_adjustment

        target = float(np.clip(base_delta + sampled_adjustment, 0.0, 1.0))

        self._last_features = features
        self._last_mean_adjustment = mean_adjustment
        self._last_sampled_adjustment = sampled_adjustment
        self._last_base_delta = base_delta
        self._last_target = target

        return target

    def update(self, transaction_cost: float, stock_price: float) -> None:
        """
        Online policy-gradient update using the realized reward from
        the trade this hedger just requested. No-op in eval mode.
        """
        if not self._training or self._last_features is None:
            return

        exposure_penalty = stock_price * abs(
            self._last_target - self._last_base_delta
        )
        reward = -(transaction_cost + self.risk_aversion * exposure_penalty)

        advantage = reward - self._baseline
        self._baseline += self.baseline_decay * (reward - self._baseline)

        if self.exploration_sigma > 0:
            score = (
                (self._last_sampled_adjustment - self._last_mean_adjustment)
                / (self.exploration_sigma**2)
            )
            grad = score * self._last_features
            self.weights += self.learning_rate * advantage * grad
