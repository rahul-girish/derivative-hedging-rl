from __future__ import annotations

from hedging.pricing.greeks import delta


class ThresholdHedger:
    """
    Delta hedging that only rebalances when the drift between the
    current position and the Black-Scholes delta exceeds a threshold.

    This trades a small amount of hedging error for fewer trades,
    which reduces transaction costs relative to hedging every step.
    """

    def __init__(self, threshold: float = 0.05) -> None:
        self.threshold = threshold

    def target_position(
        self,
        contract,
        stock_price: float,
        volatility: float,
        rate: float = 0.0,
        current_position: float = 0.0,
        **_: object,
    ) -> float:

        target_delta = delta(
            contract=contract,
            spot=stock_price,
            volatility=volatility,
            rate=rate,
        )

        if abs(target_delta - current_position) < self.threshold:
            return current_position

        return target_delta
