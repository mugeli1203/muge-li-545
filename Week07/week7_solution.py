import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.stats import norm

# Problem 1
print('Problem 1\n')
S = 165
X = 165
r = 0.0425
coupon = 0.0053
b = r - coupon
sigma = 0.2

def time_to_maturity(current_date, exp_date):
    ttm = (exp_date - current_date).days / 365
    return ttm

current_date = datetime(2022,3,13)
exp_date = datetime(2022,4,15)
T = time_to_maturity(current_date, exp_date)

def cal_d1_d2(S, X, T, sigma, b):
    d1 = (np.log(S/X)+(b + (sigma**2)/2)*T)/(sigma* np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2

# Implement the closed form greeks for GBSM
def delta_gbsm(option_type, S, X, T, sigma, b, r):
    d1, d2 = cal_d1_d2(S, X, T, sigma, b)
    if option_type == "Call":
        delta = np.exp((b-r)*T) * norm.cdf(d1)
    else:
        delta = np.exp((b-r)*T) * (norm.cdf(d1)-1)
    return delta

print('For the closed form greeks for GBSM:')
print("Call's Delta is ", delta_gbsm('Call', S, X, T, sigma, b, r))
print("Put's Delta is ", delta_gbsm('Put', S, X, T, sigma, b, r))

def gamma_gbsm(S, X, T, sigma, b, r):
    d1, d2 = cal_d1_d2(S, X, T, sigma, b)
    gamma = (norm.pdf(d1)*np.exp((b-r)*T)) / (S * sigma *np.sqrt(T))
    return gamma

print("Call's Gamma is ", gamma_gbsm( S, X, T, sigma, b, r))
print("Put's Gamma is ", gamma_gbsm( S, X, T, sigma, b, r))

def vega_gbsm(S, X, T, sigma, b, r):
    d1, d2 = cal_d1_d2(S, X, T, sigma, b)
    vega = S * np.exp((b-r)*T) * norm.pdf(d1) * np.sqrt(T)
    return vega
print("Call's Vega is ", vega_gbsm( S, X, T, sigma, b, r))
print("Put's Vega is ", vega_gbsm( S, X, T, sigma, b, r))

def theta_gbsm(option_type, S, X, T, sigma, b, r):
    d1, d2 = cal_d1_d2(S, X, T, sigma, b)
    if option_type == "Call":
        theta = -(S*np.exp((b-r)*T)*norm.pdf(d1)*sigma)/(2*np.sqrt(T))\
                -(b-r)*S*np.exp((b-r)*T)*norm.cdf(d1)\
                -r*X*np.exp(-r*T)*norm.cdf(d2)
    else:
        theta = -(S*np.exp((b-r)*T)*norm.pdf(d1)*sigma)/(2*np.sqrt(T))\
                +(b-r)*S*np.exp((b-r)*T)*norm.cdf(-d1)\
                +r*X*np.exp(-r*T)*norm.cdf(-d2)
    return theta
print("Call's Theta is ", theta_gbsm( "Call", S, X, T, sigma, b, r))
print("Put's Theta is ", theta_gbsm( "Put", S, X, T, sigma, b, r))       

def rho_gbsm(option_type, S, X, T, sigma, b, r):
    d1, d2 = cal_d1_d2(S, X, T, sigma, b)
    if option_type == "Call":
        rho = T*X*np.exp(-r*T)*norm.cdf(d2)
    else:
        rho = -T*X*np.exp(-r*T)*norm.cdf(-d2)
    return rho
print("Call's Rho is ", rho_gbsm( "Call", S, X, T, sigma, b, r))
print("Put's Rho is ", rho_gbsm( "Put", S, X, T, sigma, b, r)) 

def carry_rho_gbsm(option_type, S, X, T, sigma, b, r):
    d1, d2 = cal_d1_d2(S, X, T, sigma, b)
    if option_type == "Call":
        carry_rho = T*S*np.exp((b-r)*T)*norm.cdf(d1)
    else:
        carry_rho = -T*S*np.exp((b-r)*T)*norm.cdf(-d1)
    return carry_rho
print("Call's Carry Rho is ", carry_rho_gbsm( "Call", S, X, T, sigma, b, r))
print("Put's Carry Rho is ", carry_rho_gbsm( "Put", S, X, T, sigma, b, r)) 


# Implement a finite diference derivative calculation
def first_order_derivative(function, x, delta):
    result = (function(x+delta) - function(x-delta)) / (2*delta)
    return result

def second_order_devirvative(function, x, delta):
    result = (function(x+delta) + function(x-delta) - 2*function(x)) / (delta**2)
    return result
print('\n')
import inspect

def cal_derivative_wrt_one(function, order, object_arg, delta = 1e-3):
    all_args = list(inspect.signature(function).parameters.keys())
    orders_dic = {1:first_order_derivative, 2:second_order_devirvative}

    def cal_derivative(*args, **kwargs):
        args_dic = dict(list(zip(all_args, args)) + list(kwargs.items()))
        value_arg = args_dic.pop(object_arg)

        def trans_into_one_arg(x):
            all_args = {object_arg:x, **args_dic}
            return function(**all_args)
        return orders_dic[order](trans_into_one_arg, value_arg, delta)
    return cal_derivative

def gbsm(option_type, S, X, T, sigma, r, b):
    d1 = (np.log(S/X)+(b + (sigma**2)/2)*T)/(sigma* np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "Call":
        call_value = S * np.exp((b-r)*T)*norm.cdf(d1) - X * np.exp(-r*T)*norm.cdf(d2)
        return call_value
    else:
        put_value = X * np.exp(-r*T)*norm.cdf(-d2) - S * np.exp((b-r)*T)*norm.cdf(-d1)
        return put_value
    

print("For finite difference:")
gbsm_delta = cal_derivative_wrt_one(gbsm, 1, 'S')
print("Call's Delta is ", gbsm_delta( "Call", S, X, T, sigma, r, b))
print("Put's Delta is ", gbsm_delta( "Put", S, X, T, sigma, r, b)) 
gbsm_gamma = cal_derivative_wrt_one(gbsm, 2 ,'S')
print("Call's Gamma is ", gbsm_gamma( "Call", S, X, T, sigma, r, b))
print("Put's Gamma is ", gbsm_gamma( "Put", S, X, T, sigma, r, b)) 
gbsm_vega = cal_derivative_wrt_one(gbsm, 1 ,'sigma')
print("Call's Vega is ", gbsm_vega( "Call", S, X, T, sigma, r, b))
print("Put's Vega is ", gbsm_vega( "Put", S, X, T, sigma, r, b)) 
gbsm_theta = cal_derivative_wrt_one(gbsm, 1 ,'T')
print("Call's Theta is ", -gbsm_theta( "Call", S, X, T, sigma, r, b))
print("Put's Theta is ", -gbsm_theta( "Put", S, X, T, sigma, r, b))
gbsm_rho = cal_derivative_wrt_one(gbsm, 1 ,'r')
print("Call's Rho is ", gbsm_rho( "Call", S, X, T, sigma, r, b))
print("Put's Rho is ", gbsm_rho( "Put", S, X, T, sigma, r, b))

gbsm_carry_rho = cal_derivative_wrt_one(gbsm, 1 ,'b')
print("Call's Carry Rho is ", gbsm_carry_rho( "Call", S, X, T, sigma, r, b))
print("Put's Carry Rho is ", gbsm_carry_rho( "Put", S, X, T, sigma, r, b))

# No dividend binomial tree
def bt_no_div(call, underlying, strike, ttm, rf, b, ivol, N):
    dt = ttm/N
    u = np.exp(ivol*np.sqrt(dt))
    d = 1/u
    pu = (np.exp(b*dt)-d)/(u-d)
    pd = 1.0-pu
    df = np.exp(-rf*dt)
    z = 1 if call else -1

    def nNodeFunc(n):
        return (n+1)*(n+2) // 2
    def idxFunc(i,j):
        return nNodeFunc(j-1)+i
    nNodes = nNodeFunc(N)

    optionValues = [0.0] * nNodes

    for j in range(N,-1,-1):
        for i in range(j,-1,-1):
            idx = idxFunc(i,j)
            price = underlying*u**i*d**(j-i)
            optionValues[idx] = max(0,z*(price-strike))
            
            if j < N:
                optionValues[idx] = max(optionValues[idx], df*(pu*optionValues[idxFunc(i+1,j+1)] + pd*optionValues[idxFunc(i,j+1)]))

    return optionValues[0]

from typing import List

def bt_with_div(call: bool, underlying: float, strike: float, ttm: float, rf: float, b:float, divAmts: List[float], divTimes: List[int], ivol: float, N: int):
    # Actually b = rf in discrete dividend condition
    # if there are no dividends or the first dividend is outside out grid, return the standard bt_american value
    if not divAmts or not divTimes or divTimes[0] > N:
        return bt_no_div(call, underlying, strike, ttm, rf, b, ivol, N)
    
    dt = ttm / N
    u = np.exp(ivol * np.sqrt(dt))
    d = 1 / u
    pu = (np.exp(b * dt) - d) / (u - d)
    pd = 1 - pu
    df = np.exp(-rf * dt)
    z = 1 if call else -1

    def nNodeFunc(n: int) -> int:
        return int((n + 1) * (n + 2) / 2)
    
    def idxFunc(i: int, j: int) -> int:
        return nNodeFunc(j - 1) + i
    
    nDiv = len(divTimes)
    nNodes = nNodeFunc(divTimes[0])

    optionValues = [0] * nNodes

    for j in range(divTimes[0], -1, -1):
        for i in range(j, -1, -1):
            idx = idxFunc(i, j)
            price = underlying * (u ** i) * (d ** (j - i))
            
            if j < divTimes[0]:
                #times before the dividend working backward induction
                optionValues[idx] = max(0, z * (price - strike))
                optionValues[idx] = max(optionValues[idx], df * (pu * optionValues[idxFunc(i + 1, j + 1)] + pd * optionValues[idxFunc(i, j + 1)]))
            else:
                #time of the dividend
                valNoExercise = bt_with_div(call, price - divAmts[0], strike, ttm - divTimes[0] * dt, rf, b, divAmts[1:], [t - divTimes[0] for t in divTimes[1:]], ivol, N - divTimes[0])
                valExercise = max(0, z * (price - strike))
                optionValues[idx] = max(valNoExercise, valExercise)

    return optionValues[0]

print('\n')
N = 200
div = [0.88]
div_date = datetime(2022, 4, 11)
div_time = [round((div_date - current_date).days / (exp_date - current_date).days * N)]
#[round((dt.datetime(2023,4,11)-dt.datetime(2023,3,13)).days/(dt.datetime(2023,4,15)-dt.datetime(2023,3,13)).days*N)]
print("Using binomial tree")
print("For the condition without dividend:")
print("The value of call option is ", bt_no_div(True, S, X, T, r, b, sigma, N))
print("The value of put option is ", bt_no_div(False, S, X, T, r, b, sigma, N))

# When using discrete dividend, b = rf 
b = 0.0425
# print(b)
print("For the condition with dividend:")
print("The value of call option is ", bt_with_div(True, S, X, T, r, b, div, div_time, sigma, N))
print("The value of put option is ", bt_with_div(False, S, X, T, r, b, div, div_time, sigma, N))

print('\n')
print("Using binomial tree, calculate the greeks:")
bt_delta = cal_derivative_wrt_one(bt_with_div, 1, 'underlying')
print("Call's Delta is ", bt_delta(True, S, X, T, r, b, div, div_time, sigma, N))
print("Put's Delta is ", bt_delta(False, S, X, T, r, b, div, div_time, sigma, N)) 

bt_gamma = cal_derivative_wrt_one(bt_with_div, 2 ,'underlying')
print("Call's Gamma is ", bt_gamma( True, S, X, T, r, b, div, div_time, sigma, N))
print("Put's Gamma is ", bt_gamma( False, S, X, T, r, b, div, div_time, sigma, N)) 

bt_vega = cal_derivative_wrt_one(bt_with_div, 1 ,'ivol')
print("Call's Vega is ", bt_vega( True, S, X, T, r, b, div, div_time, sigma, N))
print("Put's Vega is ", bt_vega( False, S, X, T, r, b, div, div_time, sigma, N)) 

bt_theta = cal_derivative_wrt_one(bt_with_div, 1 ,'ttm')
print("Call's Theta is ", -bt_theta( True, S, X, T, r, b, div, div_time, sigma, N))
print("Put's Theta is ", -bt_theta( False, S, X, T, r, b, div, div_time, sigma, N))

bt_rho = cal_derivative_wrt_one(bt_with_div, 1 ,'rf')
print("Call's Rho is ", bt_rho( True, S, X, T, r, b, div, div_time, sigma, N))
print("Put's Rho is ", bt_rho( False, S, X, T, r, b, div, div_time, sigma, N))

bt_carry_rho = cal_derivative_wrt_one(bt_with_div, 1 ,'b')
print("Call's Carry Rho is ", bt_carry_rho( True, S, X, T, r, b, div, div_time, sigma, N))
print("Put's Carry Rho is ", bt_carry_rho( False, S, X, T, r, b, div, div_time, sigma, N))

# Sensitivity of the put and call to a change in dividend amount
delta = 1e-3
div_up = [0.88 + delta]
div_down = [0.88 - delta]
call_up = bt_with_div(True, S, X, T, r, b, div_up, div_time, sigma, N)    
call_down = bt_with_div(True, S, X, T, r, b, div_down, div_time, sigma, N)    
call_sens_to_div_amount = (call_up - call_down) / (2*delta)

put_up = bt_with_div(False, S, X, T, r, b, div_up, div_time, sigma, N)    
put_down = bt_with_div(False, S, X, T, r, b, div_down, div_time, sigma, N)    
put_sens_to_div_amount = (put_up - put_down) / (2*delta)
print(f"Sensitivity to dividend amount: Call: {call_sens_to_div_amount:.3f}, Put: {put_sens_to_div_amount:.3f}")

#Problem 2
print("Problem 2\n")
N = 30

from scipy.optimize import fsolve
portfolios = pd.read_csv("problem2.csv", parse_dates=["ExpirationDate"])

def find_iv(call, underlying, strike, ttm, rf, b, divAmts, divTimes, N, price, guess=0.5):
    def f(ivol):
        return bt_with_div(call, underlying, strike, ttm, rf, b, divAmts, divTimes, ivol, N) - price
    return fsolve(f, guess)[0]

current_date = datetime(2023,3,3)
S_aapl = 151.03

ivs = []

for i in range(len(portfolios.index)):
    if portfolios["Type"][i] == 'Option':    
        if portfolios["OptionType"][i] == "Call":
            call = True
        elif portfolios["OptionType"][i] =="Put":
            call = False
        X = portfolios["Strike"][i]
        T = time_to_maturity(current_date, portfolios["ExpirationDate"][i])
        N = 100
        div = [1]
        div_date = datetime(2023, 3, 15)
        div_time = [int((div_date - current_date).days / (portfolios["ExpirationDate"][i] - current_date).days * N)]
        price = portfolios["CurrentPrice"][i]
        sigma = find_iv(call, S_aapl, X, T, r, b, div, div_time, N, price)
        ivs.append(sigma)
    else:
        ivs.append(0)
# print(ivs)
portfolios['IV'] = ivs
portfolios


def cal_portfolio_value(portfolios, underlying_price, current_date):
    portfolio_values = pd.DataFrame(index=portfolios.index)
    portfolio_values['Portfolio'] = portfolios['Portfolio']

    one_values = []
    for i in range(len(portfolios.index)):
        if portfolios['Type'][i] == "Stock":
            one_p = underlying_price
        else:
            if portfolios["OptionType"][i] == "Call":
                call = True
            elif portfolios["OptionType"][i] =="Put":
                call = False
            S = underlying_price
            X = portfolios["Strike"][i]
            T = time_to_maturity(current_date, portfolios["ExpirationDate"][i])
            iv = portfolios['IV'][i]
            div = [1]
            div_date = datetime(2023, 3, 15)
            div_time = [int((div_date - current_date).days / (portfolios["ExpirationDate"][i] - current_date).days * N)]
            one_p = bt_with_div(call, S, X, T, r, b, div, div_time, iv, N)
        one_values.append(one_p)
    portfolio_values['Value'] = portfolios["Holding"] * np.array(one_values)
    return portfolio_values.groupby('Portfolio').sum()

portfolio_values_diff = cal_portfolio_value(portfolios, S_aapl, current_date)
portfolio_values_diff

# Simulation
from ml_quant_risk import VaR
import scipy

prices = pd.read_csv('DailyPrices.csv')
all_returns = VaR.return_calculate(prices, method="LOG")
AAPL_returns = all_returns['AAPL']
AAPL_ret = AAPL_returns - AAPL_returns.mean()

np.random.seed(123)
mu, std = norm.fit(AAPL_ret)
sim_returns = scipy.stats.norm(mu, std).rvs((10, 1000))
sim_prices = S_aapl * np.exp(sim_returns.sum(axis=0))

portfolio_values_sim = cal_portfolio_value(portfolios, S_aapl, current_date)
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
result['VaR'] = sim_value_changes.apply(lambda x:cal_var(x), axis=1)
result['ES'] = sim_value_changes.apply(lambda x:cal_es(x), axis=1)
print(result)



#Delta Normal

current_date = datetime(2023, 3, 3)
div_date = datetime(2023, 3, 15)
r = 0.0425
div = [1]

cal_amr_delta_num = cal_derivative_wrt_one(bt_with_div, 1, 'underlying')

deltas = []
for i in range(len(portfolios.index)):
  if portfolios["Type"][i] == "Stock":
    deltas.append(1)
  else:
    if portfolios["OptionType"][i] == "Call":
        call = True
    elif portfolios["OptionType"][i] =="Put":
        call = False
    ivol = portfolios["IV"][i]
    X = portfolios["Strike"][i]
    T = ((portfolios["ExpirationDate"][i] - current_date).days - 10) / 365
    div_time = [int((div_date - current_date).days / (portfolios["ExpirationDate"][i] - current_date).days * N)]
    delta = cal_amr_delta_num(call, S_aapl, X, T, r, b, div, div_time, ivol, N)
    deltas.append(delta)

# Store the deltas in portfolios
portfolios["deltas"] = deltas

portfolio_deltas = pd.DataFrame(index=portfolios.index)
portfolio_deltas['Portfolio'] = portfolios['Portfolio']

portfolio_deltas['Delta'] = portfolios["Holding"] * portfolios["deltas"]
portfolio_delta = portfolio_deltas.groupby('Portfolio').sum()

prices_df = pd.DataFrame(np.tile(sim_prices, (len(portfolio_delta), 1)),
                         index=portfolio_delta.index,
                         columns=[str(i) for i in sim_prices])

delta_prices = portfolio_delta['Delta'].values[:, np.newaxis] * prices_df

hedge_value = portfolio_values_sim.sub(delta_prices, fill_value=0)


current_stock = portfolio_delta * S_aapl
current_stock = current_stock.rename(columns={'Delta': 'CurrentValue'})

current_pfl = pd.DataFrame(portfolio_values_curr)

current_hedge = pd.DataFrame(current_pfl).sub(current_stock)


hedge_value_changes = (hedge_value.T - current_hedge['CurrentValue']).T
hedge_value_changes

result = pd.DataFrame(index=hedge_value_changes.index)
result['Mean'] = hedge_value_changes.mean(axis=1)
result['VaR'] = hedge_value_changes.apply(lambda x:cal_var(x), axis=1)
result['ES'] = hedge_value_changes.apply(lambda x:cal_es(x), axis=1)
print(result)

# Problem 3
print('Problem 3')
import statsmodels.api as sm

ff = pd.read_csv('F-F_Research_Data_Factors_daily.csv', parse_dates=['Date']).set_index('Date')
mom = pd.read_csv('F-F_Momentum_Factor_daily.csv', parse_dates=['Date']).set_index('Date').rename(columns={'Mom   ':  "Mom"})

factor = (ff.join(mom, how='right') / 100).loc['2013-1-31':]


all_prices = pd.read_csv('DailyPrices.csv', parse_dates=['Date'])
all_returns = pd.DataFrame(VaR.return_calculate(all_prices)).set_index('Date')

stocks = ['AAPL', 'META', 'UNH', 'MA',  
          'MSFT' ,'NVDA', 'HD', 'PFE',  
          'AMZN' ,'BRK-B', 'PG', 'XOM',  
          'TSLA' ,'JPM' ,'V', 'DIS',  
          'GOOGL', 'JNJ', 'BAC', 'CSCO']

dataset = all_returns[stocks].join(factor)

reg_set = dataset.dropna()


factors = ['Mkt-RF', 'SMB', 'HML', 'Mom']
X = reg_set[factors]
X = sm.add_constant(X)

y = reg_set[stocks].sub(reg_set['RF'], axis=0)

betas = pd.DataFrame(index=stocks, columns=factors)
alphas = pd.DataFrame(index=stocks, columns=['Alpha'])


for stock in stocks:
    ols_model = sm.OLS(y[stock], X).fit()
    betas.loc[stock] = ols_model.params[factors]
    alphas.loc[stock] = ols_model.params['const']

beta_return = pd.DataFrame(np.dot(factor[factors],betas.T), index=factor.index, columns=betas.index)

beta_rf_return = pd.merge(beta_return,factor['RF'], left_index=True, right_index=True)
alpha_beta_returns = beta_rf_return.add(beta_rf_return['RF'],axis=0).drop('RF',axis=1).add(alphas.T.loc['Alpha'], axis=1)

expected_annual_return = ((alpha_beta_returns+1).cumprod().tail(1) ** (1/alpha_beta_returns.shape[0]) - 1) * 252

expected_annual_return = expected_annual_return.reset_index()
expected_annual_return = expected_annual_return.melt(id_vars=['Date'], var_name='Stock', value_name='Annual Return')
expected_annual_return.drop('Date', axis=1, inplace=True)
print(expected_annual_return)

# Annual covariance matrix

cov_mtx = all_returns[stocks].cov()*252
print(cov_mtx)


def super_efficient_portfolio(returns, rf_rate, cov_matrix):
    annual_returns = returns['Annual Return'].values.T
    num_stocks = len(returns)

    def neg_sharpe_ratio(weights):
        pfl_return = np.sum(annual_returns * weights)
        pfl_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_ratio = (pfl_return - rf_rate) / pfl_std_dev
        return -sharpe_ratio

    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                   {'type': 'ineq', 'fun': lambda w: w}]
    
    bounds = [(0, 1) for _ in range(num_stocks)]
    
    init_weights = np.ones(num_stocks) / num_stocks  
    opt_result = scipy.optimize.minimize(neg_sharpe_ratio, init_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    
    # Return optimal weights and Sharpe ratio of resulting portfolio
    super_weights = opt_result.x
    super_sharpe_ratio = - neg_sharpe_ratio(super_weights)

    return super_weights*100, super_sharpe_ratio

weights, sharpe_ratio = super_efficient_portfolio(expected_annual_return, 0.0425, cov_mtx)

weights_df = pd.DataFrame(columns=['Stock', 'Weight'])
weights_df['Stock'] = stocks
weights_df['Weight'] = weights.round(2)
print(weights_df)
print(f"Sharpe Ratio of super efficient portfolio is {sharpe_ratio:.2f}")
