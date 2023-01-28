import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew, t, ttest_1samp, norm
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize 


#Problem 1
#Test the kurtosis function for bias
#Null Hypothesis: Kurtosis function is unbiased.

sample_size = [100, 1000, 10000]
for s in sample_size:
    samples = 1000
    kurts = np.empty(samples)
    print("Sample size is {}." .format(s))
    for i in range(samples):
        kurts[i] = kurtosis(np.random.normal(0, 1, s))
    print(ttest_1samp(kurts, 0))
    t_statistic, p_value = ttest_1samp(kurts, 0)

    threshold = 0.05
    if p_value < threshold:
        print("#Test the kurtosis function for bias: Reject the Null Hypothesis.\n")
    else:
        print("#Test the kurtosis function for bias: Fail to reject the Null Hypothesis.\n")


#Test the Skewness function for bias
#Null Hypothesis: Skewness function is unbiased.
sample_size = [100, 1000, 10000]
for s in sample_size:
    samples = 1000
    skewness = np.empty(samples)
    print("Sample size is {}." .format(s))
    for i in range(samples):
       skewness[i] = skew(np.random.normal(0, 1, s))
    print(ttest_1samp(skewness, 0))
    t_statistic, p_value = ttest_1samp(skewness, 0)

    threshold = 0.05
    if p_value < threshold:
        print("#Test the skewness function for bias: Reject the Null Hypothesis.\n")
    else:
        print("#Test the skewness function for bias: Fail to reject the Null Hypothesis.\n")


#Problem 2
#Deal with the raw data
data = pd.read_csv('problem2.csv')
x = data['x']
y = data['y']
X = sm.add_constant(x)


#Fit the data using OLS
OLS = sm.OLS(y, X)
OLS_result = OLS.fit()
OLS_result.summary()

#Calculate error vector
error_vec = OLS_result.resid
#How well fit normal distribution
print("Mean of error vector is {}\n" .format(np.mean(error_vec)))
print("Variance of error vector is {}\n" .format(np.var(error_vec)))
print("Skewness of error vector is {}\n" .format(skew(error_vec)))
print("Kurtosis of error vector is {}\n" .format(kurtosis(error_vec)))
plt.hist(error_vec, bins = 20, density = True)
x_axis = np.linspace(-4, 4, 100)
plt.plot(x_axis, norm.pdf(x_axis, 0,1))

#Fit the data using MLE
#Normal distribution
def MLE_normal(p):
    yhat = p[0] + p[1] * x
    nll = -1 * np.sum(stats.norm.logpdf(y-yhat, 0, p[2]))
    return nll
result_N = minimize(MLE_normal, x0 = (1,1,1))
result_N

#T distribution
def MLE_T(p):
    yhat = p[0] + p[1] * x
    nll = -1 * np.sum(stats.t.logpdf(y-yhat, p[2], scale=p[3]))
    return nll
result_T = minimize(MLE_T, x0=(1,1,1,1))
result_T

#Goodness of Fit
#R^2
def R_sq(beta0, beta1):
    y_bar = np.mean(y)
    ss_tot = sum((y - y_bar)**2)
    error = y - (beta0 + beta1 * x)
    ee_res = sum((error - np.mean(error)) ** 2)
    r_sq = 1-ee_res/ss_tot
    return r_sq

r_sq_N = R_sq(result_N.x[0], result_N.x[1])
print("MLE R-suqare in normal distribution is {}" .format(r_sq_N))
r_sq_T = R_sq(result_T.x[0], result_T.x[1])
print("MLE R-suqare in T distribution is {}" .format(r_sq_T))

#AIC
AIC_N = 2 * 2 + 2* result_N.fun
print("MLE AIC in normal distribution is {}" .format(AIC_N))
AIC_T = 2 * 2 + 2* result_T.fun
print("MLE AIC in T distribution is {}" .format(AIC_T))

#BIC
BIC_N = 2 * np.log(len(x)) + 2 * result_N.fun
print("MLE BIC in normal distribution is {}" .format(BIC_N))
BIC_T = 2 * np.log(len(x)) + 2 * result_T.fun
print("MLE BIC in T distribution is {}" .format(BIC_T))


#Problem 3

AR1 = sm.tsa.arma_generate_sample(ar=[1,0.7], ma=[1], nsample=3000)
plt.plot(AR1)
plt.title("AR1")
sm.graphics.tsa.plot_acf(AR1,lags = 10)
sm.graphics.tsa.plot_pacf(AR1, lags = 10)


AR2 = sm.tsa.arma_generate_sample(ar=[1,0.7,0.7], ma=[1], nsample=3000)
plt.plot(AR2)
plt.title("AR2")
sm.graphics.tsa.plot_acf(AR2,lags = 10)
sm.graphics.tsa.plot_pacf(AR2,lags = 10)

AR3 = sm.tsa.arma_generate_sample(ar=[1,0.7,0.7,0.7], ma=[1], nsample=3000)
plt.plot(AR3)
plt.title("AR3")
sm.graphics.tsa.plot_acf(AR3,lags = 10)
sm.graphics.tsa.plot_pacf(AR3,lags = 10)

MA1 = sm.tsa.arma_generate_sample(ar=[1], ma=[1,0.7], nsample=3000)
plt.plot(MA1)
plt.title("MA1")
sm.graphics.tsa.plot_acf(MA1,lags = 10)
sm.graphics.tsa.plot_pacf(MA1,lags = 10)


MA2 = sm.tsa.arma_generate_sample(ar=[1], ma=[1,0.7,0.7], nsample=3000)
plt.plot(MA2)
plt.title("MR2")
sm.graphics.tsa.plot_acf(MA2,lags = 10)
sm.graphics.tsa.plot_pacf(MA2,lags = 10)

MA3 = sm.tsa.arma_generate_sample(ar=[1], ma=[1,0.7,0.7,0.7], nsample=3000)
plt.plot(MA3)
plt.title("MA3")
sm.graphics.tsa.plot_acf(MA3,lags = 10)
sm.graphics.tsa.plot_pacf(MA3,lags = 10)
