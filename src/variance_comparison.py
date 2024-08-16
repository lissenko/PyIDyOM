import numpy as np
import statsmodels.api as sm

from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt

def model_comparison(reference_model_data, test_model_data):
    test_model_data = sm.add_constant(test_model_data)

    ols_model = sm.OLS(reference_model_data, test_model_data).fit()

    R_squared = ols_model.rsquared
    R_squared_adj = ols_model.rsquared_adj
    F_statistic = ols_model.fvalue
    p_value = ols_model.f_pvalue
    R = np.sqrt(R_squared)

    return R, R_squared, R_squared_adj, F_statistic, p_value

