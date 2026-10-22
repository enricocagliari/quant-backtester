import numpy as np
import pandas as pd
import pytest

from qbt import synthetic


@pytest.fixture
def prices() -> pd.DataFrame:
    return synthetic.random_walk(n_days=600, n_assets=4, seed=1)


@pytest.fixture
def flat_prices() -> pd.DataFrame:
    index = pd.bdate_range('2020-01-01', periods=10)
    return pd.DataFrame({'A': np.full(10, 100.0), 'B': np.full(10, 50.0)}, index=index)
