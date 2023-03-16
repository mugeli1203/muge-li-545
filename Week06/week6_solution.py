import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.stats import norm
import matplotlib.pyplot as plt
from ml_quant_risk import VaR
#Problem 1

def time_to_maturity(current_date, exp_date):
    ttm = (exp_date - current_date).days / 365
    return ttm
current_date = datetime(2023,3,3)
exp_date = datetime(2023,3,17)
ttm = time_to_maturity(current_date, exp_date)
print("Time to Maturity is ",round(ttm, 4))

def BS_option(option_type, S, X, T, sigma, r, b):
    d1 = (np.log(S/X)+(b + (sigma**2)/2)*T)/(sigma* np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "Call":
        call_value = S * np.exp((b-r)*T)*norm.cdf(d1) - X * np.exp(-r*T)*norm.cdf(d2)
        return call_value
    else:
        put_value = X * np.exp(-r*T)*norm.cdf(-d2) - S * np.exp((b-r)*T)*norm.cdf(-d1)
        return put_value


S = 165
X_call = 150
X_put = 180
r = 0.0425
coupon = 0.0053
b = r - coupon

call_values = []
put_values = []
sigmas = np.linspace(0.1, 0.8, 70)
for sigma in sigmas:
    call_value = BS_option("Call", S, X_call, ttm, sigma, r, b)
    call_values.append(call_value)
    put_value = BS_option("Put", S, X_put, ttm, sigma, r, b)
    put_values.append(put_value)

# print(call_values)
# print(put_values)
plt.figure()
plt.plot(sigmas, call_values, label="Call")
plt.plot(sigmas, put_values, label="Put")
plt.xlabel("Implied Volatility")
plt.ylabel("Value")
plt.legend()
plt.show()

# Problem 2
from scipy.optimize import brentq
import scipy

def find_iv(option_type, S, X, T, r, b, price, guess):
    def f(iv):
        return BS_option(option_type, S, X, T, iv, r, b) - price
    return brentq(f, guess, 1, xtol=1e-15, rtol=1e-15)

AAPL_options = pd.read_csv("AAPL_Options.csv", parse_dates=["Expiration"])
S = 151.03
call_X = []
call_iv =[]
put_X = []
put_iv = []
for i in range(len(AAPL_options)):
    option_type = AAPL_options['Type'][i]
    X = AAPL_options['Strike'][i]
    T = time_to_maturity(current_date, AAPL_options['Expiration'][i])
    price = AAPL_options['Last Price'][i]
    implied_volatility = find_iv(option_type, S, X, T, r, b, price, 0.0001)
    if option_type == "Call":
        call_X.append(X)
        call_iv.append(implied_volatility)
    if option_type == "Put":
        put_X.append(X)
        put_iv.append(implied_volatility)

# implied_vol = find_iv("CALL", 151.03, 125, ttm, r, b, 27.3, 0.0001)
# print(f"Implied Vol: {implied_vol:.4f}")
# print(call_X)
# print(call_iv)
plt.figure()
plt.plot(call_X, call_iv, label="Call")
plt.plot(put_X, put_iv, label="Put")
plt.xlabel("Strike Price")
plt.ylabel("Implied Volatility")
plt.legend()
plt.show()

#Problem 3

portfolios = pd.read_csv("problem3.csv", parse_dates=["ExpirationDate"])

ivs = []
for i in range(len(portfolios.index)):
  option_type = portfolios["OptionType"][i]
  X = portfolios["Strike"][i]
  T = time_to_maturity(current_date, portfolios["ExpirationDate"][i])
  price = portfolios["CurrentPrice"][i]
  sigma = find_iv(option_type, S, X, T, r, b, price, 0.001)
  ivs.append(sigma)
portfolios['IV'] = ivs

def cal_portfolio_value(portfolios, underlying_price, current_date):
    portfolio_values = pd.DataFrame(index=portfolios.index)
    portfolio_values['Portfolio'] = portfolios['Portfolio']

    one_values = []
    for i in range(len(portfolios.index)):
        if portfolios['Type'][i] == "Stock":
            one_p = underlying_price
        else:
            opt_type = portfolios["OptionType"][i]
            S = underlying_price
            X = portfolios["Strike"][i]
            T = time_to_maturity(current_date, portfolios["ExpirationDate"][i])
            iv = portfolios['IV'][i]
            one_p = BS_option(opt_type, S, X, T, iv, r, b)
        one_values.append(one_p)
    portfolio_values['Value'] = portfolios["Holding"] * np.array(one_values)
    return portfolio_values.groupby('Portfolio').sum()

underlying_values = np.linspace(101, 200, 100)
portfolio_values_diff = cal_portfolio_value(portfolios, 100, current_date)

for i in underlying_values:
    temp_one_pv = cal_portfolio_value(portfolios, i, current_date)
    portfolio_values_diff[str(i)] = temp_one_pv['Value']
# print(portfolio_values_diff)
fig, axes = plt.subplots(3, 3, figsize=(18, 16))
idx = 0
for portfolio, dataframe in portfolio_values_diff.groupby('Portfolio'):
    i, j = idx // 3, idx % 3
    ax = axes[i][j]
    ax.plot(underlying_values, dataframe.iloc[0, 1:].values)
    ax.set_title(portfolio)
    ax.set_xlabel('Underlying Price', fontsize=8)
    ax.set_ylabel('Portfolio Value', fontsize=8)
    idx += 1



from statsmodels.tsa.arima.model import ARIMA

prices = pd.read_csv('DailyPrices.csv')
all_returns = VaR.return_calculate(prices, method="LOG")
AAPL_returns = all_returns['AAPL']
AAPL_ret = AAPL_returns - AAPL_returns.mean()
ar_model = ARIMA(AAPL_ret, order=(1,0,0)).fit()
Beta = ar_model.params[0]
A = ar_model.params[1]
resid_std = np.std(ar_model.resid)
sim_returns = scipy.stats.norm(0, resid_std).rvs((10, 10000))
# print(sim_returns[0])
sim_returns[0] = Beta + AAPL_ret[len(AAPL_ret)]*A + sim_returns[0]
# print(sim_returns[0])

# print(sim_returns)
# print(len(sim_returns))
for i in range(1, len(sim_returns)):
    sim_returns[i] = Beta + sim_returns[i-1]*A + sim_returns[i]
# print(sim_returns)
# print(sim_returns.sum(axis=0))
sim_prices = 151.03 * np.exp(sim_returns.sum(axis=0))
# print(sim_prices)

current_date = datetime(2023, 3, 3) + timedelta(days=10)
ivs = []
for i in range(len(portfolios.index)):
  option_type = portfolios["OptionType"][i]
  X = portfolios["Strike"][i]
  T = time_to_maturity(current_date, portfolios["ExpirationDate"][i])
  price = portfolios["CurrentPrice"][i]
  sigma = find_iv(option_type, 151.03, X, T, r, b, price, 0.001)
  ivs.append(sigma)
portfolios['IV'] = ivs

portfolio_values_sim = cal_portfolio_value(portfolios, 100, current_date)
for i in sim_prices:
    temp_one_pv = cal_portfolio_value(portfolios, i, current_date)
    portfolio_values_sim[str(i)] = temp_one_pv['Value']
portfolio_values_sim.drop('Value', axis=1, inplace=True)
# print(portfolio_values_sim)
portfolios["CurrentValue"] = portfolios["CurrentPrice"] * portfolios["Holding"]
portfolio_values_curr = portfolios.groupby('Portfolio')['CurrentValue'].sum()
# print(portfolio_values_curr)
sim_value_changes = (portfolio_values_sim.T - portfolio_values_curr).T
# print(sim_value_changes)

def cal_var(sim_data, alpha = 0.05):
    sim_data_sorted = np.sort(sim_data)
    var = sim_data_sorted[int(alpha * len(sim_data))] * (-1)
    return var

def cal_es(sim_data, alpha = 0.05):
    sim_data_sorted = np.sort(sim_data)
    var = sim_data_sorted[int(alpha * len(sim_data))] * (-1)
    return -np.mean(sim_data_sorted[sim_data_sorted <= -var])


# Calculate the Mean, VaR and ES, and print the results
result = pd.DataFrame(index=sim_value_changes.index)
result['Mean'] = sim_value_changes.mean(axis=1)
result['VaR'] = sim_value_changes.apply(lambda x:cal_var(x, 0), axis=1)
result['ES'] = sim_value_changes.apply(lambda x:cal_es(x), axis=1)
print(result)