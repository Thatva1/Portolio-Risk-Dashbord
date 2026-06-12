import numpy as np
import pandas as pd
from risk.var import historical_var, parametric_var_normal

def test_historical_var():
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0, 0.01, 1000))
    var95 = historical_var(returns, 0.95)
    assert var95 < 0
    assert var95 > -0.05

def test_parametric_var_normal():
    returns = pd.Series(np.random.normal(0, 0.01, 1000))
    var95 = parametric_var_normal(returns, 0.95)
    assert var95 < 0
