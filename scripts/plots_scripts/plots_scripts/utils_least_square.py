# %%
from scipy.odr import Model, Data, ODR
from scipy.stats import linregress
import numpy as np
from sklearn.linear_model import LinearRegression


def total_least_squares(x, y, fit_intercept=True):
    """Perform an Orthogonal Distance Regression on the given data,
    using the same interface as the standard scipy.stats.linregress function.
    Arguments:
    x: x data
    y: y data
    Returns:
    [m, c]
    m: slope
    c: intercept
    Uses standard ordinary least squares to estimate the starting parameters then uses the scipy.odr interface to the ODRPACK Fortran code to do the orthogonal distance calculations.
    """
    linreg = ordinary_least_squares(x, y, fit_intercept=fit_intercept)
    if fit_intercept:
        f = lambda p, x: (p[0] * x) + p[1]
    else:
        f = lambda p, x: p[0] * x

    mod = Model(f)
    dat = Data(x, y)
    od = ODR(dat, mod, beta0=linreg)
    out = od.run()
    slope = out.beta[0]
    intercept = out.beta[1]
    return np.array([slope, intercept])

def ordinary_least_squares(x, y, fit_intercept=True):
    """Perform an Ordinary Least Squares Regression on the given data,
    using the same interface as the standard scipy.stats.linregress function.
    Arguments:
    x: x data
    y: y data
    Returns:
    [m, c]
    m: slope
    c: intercept
    Uses standard ordinary least squares to estimate the starting parameters
    """
    lr = LinearRegression(fit_intercept=fit_intercept).fit(
        x.reshape(-1, 1), y.reshape(-1, 1)
    )
    if fit_intercept:
        intercept = lr.intercept_[0]
    else:
        intercept = 0
    slope = lr.coef_[0][0]
    return np.array([slope, intercept])
