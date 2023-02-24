import numpy as np
from scipy import stats

#Normal distribution
def MLE_normal(p, x, y):
    yhat = p[0] + p[1] * x
    nll = -1 * np.sum(stats.norm.logpdf(y-yhat, 0, p[2]))
    return nll


#T distribution
def MLE_T(p, x, y):
    yhat = p[0] + p[1] * x
    nll = -1 * np.sum(stats.t.logpdf(y-yhat, p[2], scale=p[3]))
    return nll


#Goodness of Fit
#R^2
def R_sq(x, y, beta0, beta1):
    y_bar = np.mean(y)
    ss_tot = sum((y - y_bar)**2)
    error = y - (beta0 + beta1 * x)
    ee_res = sum((error - np.mean(error)) ** 2)
    r_sq = 1-ee_res/ss_tot
    return r_sq