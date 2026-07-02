import numpy as np

from hedging.environment.state import EnvironmentState


def test_state_to_numpy():

    state = EnvironmentState(
        stock_price=100,
        time_to_maturity=1.0,
        delta=0.5,
        shares_held=0.5,
        cash_balance=10,
    )

    obs = state.to_numpy()

    assert isinstance(obs, np.ndarray)

    assert obs.shape == (5,)