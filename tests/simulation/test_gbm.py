import numpy as np

from hedging.simulation.gbm import GBMSimulator


def test_output_shape():

    simulator = GBMSimulator(
        mu=0.05,
        sigma=0.20,
    )

    paths = simulator.simulate(
        s0=100,
        n_steps=252,
        n_paths=500,
        seed=42,
    )

    assert paths.prices.shape == (500, 253)


def test_first_price_equals_initial():

    simulator = GBMSimulator(
        mu=0.05,
        sigma=0.20,
    )

    paths = simulator.simulate(
        s0=100,
        n_steps=50,
        seed=1,
    )

    assert paths.prices[0, 0] == 100


def test_reproducibility():

    simulator = GBMSimulator(
        mu=0.05,
        sigma=0.20,
    )

    a = simulator.simulate(
        100,
        252,
        100,
        seed=123,
    )

    b = simulator.simulate(
        100,
        252,
        100,
        seed=123,
    )

    assert np.allclose(a.prices, b.prices)


def test_different_seed_changes_paths():

    simulator = GBMSimulator(
        mu=0.05,
        sigma=0.20,
    )

    a = simulator.simulate(
        100,
        252,
        100,
        seed=1,
    )

    b = simulator.simulate(
        100,
        252,
        100,
        seed=2,
    )

    assert not np.allclose(a.prices, b.prices)