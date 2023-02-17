import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import t, norm, normaltest
from statsmodels.tsa.arima.model import ARIMA

#Problem 1

def cla_brown(P0, std, N):
    r = np.random.normal(0, std, N)
    Pt = P0 + r
    return Pt

def arith(P0, std, N):
    r = np.random.normal(0, std, N)
    Pt = P0 * (1+r)
    return Pt

def geo_brown(P0, std, N):
    r = np.random.normal(0, std, N)
    Pt = P0 * np.exp(r)
    return Pt

#Assume P0 = 100, std = 1
P0 = 100
std = 0.5
N = 10000

cla_brown_P = cla_brown(P0, std, N)
arith_P = arith(P0, std, N)
geo_brown_P = geo_brown(P0, std, N)

print("Using Classic Brownian Motion, mean = {}, std = {}".format(np.mean(cla_brown_P), np.std(cla_brown_P)))
print("Using Arithmetic Return System, mean = {}, std = {}".format(np.mean(arith_P), np.std(arith_P)))
print("Using Geometric Brownian Motion, mean = {}, std = {}".format(np.mean(geo_brown_P), np.std(geo_brown_P)))


fig, axes = plt.subplots(3, 1, figsize=(6, 10))
sns.distplot(cla_brown_P, ax=axes[0])
sns.distplot(arith_P, ax=axes[1])
sns.distplot(geo_brown_P, ax=axes[2])
axes[0].set_title("Classical Brownian Motion")
axes[1].set_title("Arithmetic Return System")
axes[2].set_title("Geometric Brownian Motion")
plt.show()

theoretically_std = np.sqrt((np.exp(std**2)-1)*np.exp(2*(np.log(P0))+std**2))

#Problem 2

def return_calculate(prices: pd.DataFrame, method="ARITHMETIC", dateColumn="Date") -> pd.DataFrame:
    vars = prices.columns.values.tolist() #list of the column names
    nVars = len(vars)
    vars.remove(dateColumn) #remove the column of "date"
    if nVars == len(vars):
        raise ValueError(f"dateColumn: {dateColumn} not in DataFrame: {vars}")
    nVars = nVars - 1
    p = np.array(prices.drop(columns=[dateColumn]))
    n = p.shape[0] #the number of rows
    m = p.shape[1] #the number of column
    p2 = np.empty((n-1, m))
    for i in range(n-1):
        for j in range(m):
            p2[i,j] = p[i+1,j] / p[i,j]
    if method.upper() == "ARITHMETIC":
        p2 = p2 - 1.0
    elif method.upper() == "LOG":
        p2 = np.log(p2)
    else:
        raise ValueError(f"method: {method} must be in (\"LOG\",\"DISCRETE\")")
    dates = prices[dateColumn][1:]
    out = pd.DataFrame({dateColumn: dates})
    for i in range(nVars):
        out[vars[i]] = p2[:,i]
    return out

prices = pd.read_csv('DailyPrices.csv')
returns = return_calculate(prices)

META_ret = returns['META']
#Remove the mean mean(META)=0
META_ret0 = META_ret - META_ret.mean()



#Calculate VaR using a normal distribution
def norml_var(returns, alpha = 0.05, N = 10000):
        mean = returns.mean()
        std = returns.std()
        Rt = np.random.normal(mean, std, N)
        Rt.sort()
        var = Rt[int(alpha * len(Rt))] * (-1)
        #print(-np.percentile(returns, alpha*100))
        return var, Rt


#Calculate VaR using a normal distribution with an Exponentially Weighted variance
def exp_w_variance(returns, w_lambda = 0.94):
    weight = np.zeros(len(returns))
    for i in range(len(returns)):
        weight[len(returns)-1-i]  = (1-w_lambda)*w_lambda**i
    weight = weight/sum(weight)    
    ret_means = returns - returns.mean()
    expo_w_var = ret_means.T @ np.diag(weight) @ ret_means
    return expo_w_var 

#META_ew_var = exp_w_variance(META_ret0)
#print(META_ew_var)

def norml_ew_var(returns, alpha = 0.05, N = 10000):
        mean = returns.mean()
        std = np.sqrt(exp_w_variance(returns))
        Rt = np.random.normal(mean, std, N)
        Rt.sort()
        var = Rt[int(alpha * len(Rt))] * (-1)
        #print(-np.percentile(returns, alpha*100))
        return var, Rt
#var_ew = norml_ew_var(META_ret0)
#print(var_ew)

#Calculate VaR using a MLE fitted T distribution
def MLE_T_var(returns, alpha = 0.05, N = 10000):
    result = t.fit(returns, method="MLE")
    df = result[0]
    loc = result[1]
    scale = result[2]

    Rt = t(df, loc, scale).rvs(N)
    Rt.sort()
    var = Rt[int(alpha * len(Rt))] * (-1)
    return var, Rt
#var_T = MLE_T_var(META_ret0)
#print(var_T)

#Calculate VaR using a fitted AR(1) model
def ar1_var(returns, alpha = 0.05, N = 10000):
    result = ARIMA(returns, order=(1, 0, 0)).fit()
    t_a = result.params[0]
    resid_std = np.std(result.resid)
    Rt = np.empty(N)
    Rt = t_a * returns[len(returns)] + np.random.normal(loc=0, scale=resid_std, size=N)
    Rt.sort()
    var = Rt[int(alpha * len(Rt))] * (-1)
    return var, Rt
#var_ar1 = ar1_var(META_ret0)
#print(var_ar1)   

#Calculate VaR using a historic simulation
def his_var(returns, alpha = 0.05):
    Rt = returns.values
    Rt.sort()
    var = Rt[int(alpha * len(Rt))] * (-1)
    return var, Rt
#var_his = his_var(META_ret0)
#print(var_his) 

var_nor, Rt_nor = norml_var(META_ret0)
var_ew, Rt_ew = norml_ew_var(META_ret0)
var_T, Rt_T = MLE_T_var(META_ret0)
var_ar1, Rt_ar1 = ar1_var(META_ret0)
var_his, Rt_his = his_var(META_ret0)


#Transfer to dollar loss
def ret_to_dollar(ret_var, return_s, price_s):
    dollar_var = (ret_var - return_s.mean()) * price_s.values[-1]
    return dollar_var

print("VaR in dollar using a normal distribution = {:.4f}".format(ret_to_dollar(var_nor, META_ret, prices['META'])))
print("VaR in dollar using a normal distribution with an Exponentially Weighted Variance = {:.4f}".format(ret_to_dollar(var_ew, META_ret, prices['META'])))
print("VaR in dollar using a MLE fitted T distribution = {:.4f}".format(ret_to_dollar(var_T, META_ret, prices['META'])))
print("VaR in dollar using a fitted AR(1) model = {:.4f}".format(ret_to_dollar(var_ar1, META_ret, prices['META'])))
print("VaR in dollar using a historic simulation = {:.4f}".format(ret_to_dollar(var_his, META_ret, prices['META'])))


#Make plots
fig, axes = plt.subplots(5, 1, figsize=(10, 30))
sns.distplot(Rt_nor, ax=axes[0])
sns.distplot(Rt_ew, ax=axes[1])
sns.distplot(Rt_T, ax=axes[2])
sns.distplot(Rt_ar1, ax=axes[3])
sns.distplot(Rt_his, ax=axes[4])


axes[0].axvline(x=-var_nor, color='red', alpha=0.3)
axes[1].axvline(x=-var_ew, color='red', alpha=0.3)
axes[2].axvline(x=-var_T, color='red', alpha=0.3)
axes[3].axvline(x=-var_ar1, color='red', alpha=0.3)
axes[4].axvline(x=-var_his, color='red', alpha=0.3)


axes[0].set_title("VaR using a normal distribution = {}".format(var_nor.round(4)))
axes[1].set_title("VaR using a normal distribution with an Exponentially Weighted Variance = {}".format(var_ew.round(4)))
axes[2].set_title("VaR using a MLE fitted T distribution = {}".format(var_T.round(4)))
axes[3].set_title("VaR using a fitted AR(1) model = {}".format(var_ar1.round(4)))
axes[4].set_title("VaR using a historic simulation = {}".format(var_his.round(4)))

plt.subplots_adjust(hspace=0.5)
plt.show()


#Problem 3
portfolio = pd.read_csv('portfolio.csv')
prices = pd.read_csv('DailyPrices.csv')

def expo_weighted_cov(ret_data,w_lambda):
    weight = np.zeros(len(ret_data))
    for i in range(len(ret_data)):
        weight[len(ret_data)-1-i]  = (1-w_lambda)*w_lambda**i
    weight = weight/sum(weight)
    ret_means = ret_data - ret_data.mean()
    #print(ret_means.T.values.shape)
    #print(np.diag(weight).shape)
    #print(ret_means.values.shape)
    expo_w_cov = ret_means.T.values @ np.diag(weight) @ ret_means.values
    return expo_w_cov

def process_portfolio_data(portfolio, prices, p_type):
    if p_type == "total":
        co_assets = portfolio.drop('Portfolio', axis = 1)
        co_assets = co_assets.groupby(["Stock"], as_index=False)["Holding"].sum()
    else:
        co_assets = portfolio[portfolio['Portfolio'] == p_type]
    dailyprices = pd.concat([prices["Date"], prices[co_assets["Stock"]]], axis=1)

    holdings = co_assets['Holding']

    portfolio_price = np.dot(prices[co_assets["Stock"]].tail(1), co_assets['Holding'])

    return portfolio_price, dailyprices, holdings


print("Portfolio current prices: A, B, C, TOTAL")
A_data_p, A_day_p, A_holding = process_portfolio_data(portfolio, prices, 'A')
B_data_p, B_day_p, B_holding = process_portfolio_data(portfolio, prices, 'B')
C_data_p, C_day_p, C_holding = process_portfolio_data(portfolio, prices, 'C')
T_data_p, T_day_p, T_holding = process_portfolio_data(portfolio, prices, 'total')
print(A_data_p)
print(B_data_p)
print(C_data_p)
print(T_data_p)

#Calculate VaR using MC simulation
# Use PCA simulation with 100% variance explianed
def pca_sim(cov_mtx, n_draws, percent_explain = 1):
    eigenvalues, eigenvectors = np.linalg.eig(cov_mtx)
    #Keep those positive eigenvalues and corresponding eigenvectors
    p_idx = eigenvalues > 1e-8
    eigenvalues = eigenvalues[p_idx]
    eigenvectors = eigenvectors[:, p_idx]
    #Sort
    s_idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[s_idx]
    eigenvectors = eigenvectors[:, s_idx]

    if percent_explain == 1.0:
        percent_explain = (np.cumsum(eigenvalues)/np.sum(eigenvalues))[-1]

    n_eigenvalues = np.where((np.cumsum(eigenvalues)/np.sum(eigenvalues))>= percent_explain)[0][0] + 1
    #print(n_eigenvalues)
    eigenvectors = eigenvectors[:,:n_eigenvalues]
    eigenvalues = eigenvalues[:n_eigenvalues]
    std_normals = np.random.normal(size=(n_eigenvalues, n_draws))

    B = eigenvectors.dot(np.diag(np.sqrt(eigenvalues)))
    return np.transpose(B.dot(std_normals))


def cal_MC_var(portfolio, prices, p_type, alpha=0.05, w_lambda=0.94, N = 10000):
    portfolio_price, dailyprices, holding = process_portfolio_data(portfolio, prices, p_type)
    returns = return_calculate(dailyprices).drop('Date',axis=1)
    returns_0 = returns - returns.mean()

    np.random.seed(0)
    cov_mtx = expo_weighted_cov(returns_0, w_lambda)
    sim_ret = pca_sim(cov_mtx, N)
    dailyprices = np.add(dailyprices.drop('Date', axis=1), returns.mean().values)
    sim_change = np.dot(sim_ret * dailyprices.tail(1).values.reshape(dailyprices.shape[1]),holding)

    var = np.percentile(sim_change, alpha*100) * (-1)
    return var, sim_change

#Calculate VaR using Delta Normal
def cal_delta_var(portfolio, prices, p_type, alpha=0.05, w_lambda=0.94, N = 10000):
    portfolio_price, dailyprices, holding = process_portfolio_data(portfolio, prices, p_type)

    returns = return_calculate(dailyprices).drop('Date',axis=1)
    dailyprices = dailyprices.drop('Date', axis=1)
    dR_dr = (dailyprices.tail(1).T.values * holding.values.reshape(-1, 1)) / portfolio_price
    cov_mtx = expo_weighted_cov(returns, w_lambda)
    R_std = np.sqrt(np.transpose(dR_dr) @ cov_mtx @ dR_dr)
    var = (-1) * portfolio_price * norm.ppf(alpha) * R_std
    return var[0][0]


#Calculate VaR using historic simulation
def cal_his_var(portfolio, prices, p_type, alpha=0.05, N = 10000):
    portfolio_price, dailyprices, holding = process_portfolio_data(portfolio, prices, p_type)
    returns = return_calculate(dailyprices).drop('Date',axis=1)
    np.random.seed(0)
    sim_ret = returns.sample(N, replace=True)
    dailyprices = dailyprices.drop('Date', axis=1)
    sim_change = np.dot(sim_ret * dailyprices.tail(1).values.reshape(dailyprices.shape[1]),holding)

    var = np.percentile(sim_change, alpha*100) * (-1)
    return var, sim_change

#Plot MC
MC_var_A, MC_dis_A = cal_MC_var(portfolio, prices, "A")
print("VaR of Portfolio A using MC = {}".format(MC_var_A.round(4)))
MC_var_B, MC_dis_B = cal_MC_var(portfolio, prices, "B")
print("VaR of Portfolio B using MC = {}".format(MC_var_B.round(4)))
MC_var_C, MC_dis_C = cal_MC_var(portfolio, prices, "C")
print("VaR of Portfolio C using MC = {}".format(MC_var_C.round(4)))
MC_var_T, MC_dis_T = cal_MC_var(portfolio, prices, "total")
print("VaR of Portfolio Total using MC = {}".format(MC_var_T.round(4)))

fig, axes = plt.subplots(4, 1, figsize=(10, 30))
sns.distplot(MC_dis_A, ax=axes[0])
sns.distplot(MC_dis_B, ax=axes[1])
sns.distplot(MC_dis_C, ax=axes[2])
sns.distplot(MC_dis_T, ax=axes[3])

axes[0].axvline(x=-MC_var_A, color='red', alpha=0.3)
axes[1].axvline(x=-MC_var_B, color='red', alpha=0.3)
axes[2].axvline(x=-MC_var_C, color='red', alpha=0.3)
axes[3].axvline(x=-MC_var_T, color='red', alpha=0.3)

axes[0].set_title("VaR of Portfolio A using MC = {}".format(MC_var_A.round(4)))
axes[1].set_title("VaR of Portfolio B using MC = {}".format(MC_var_B.round(4)))
axes[2].set_title("VaR of Portfolio C using MC = {}".format(MC_var_C.round(4)))
axes[3].set_title("VaR of Portfolio Total using MC = {}".format(MC_var_T.round(4)))
plt.subplots_adjust(hspace=0.5)
plt.show()


#print delta
delta_var_A = cal_delta_var(portfolio, prices, "A")
print("Delta Normal VaR of Portfolio A = {}".format(delta_var_A.round(4)))
delta_var_B = cal_delta_var(portfolio, prices, "B")
print("Delta Normal VaR of Portfolio B = {}".format(delta_var_B.round(4)))
delta_var_C = cal_delta_var(portfolio, prices, "C")
print("Delta Normal VaR of Portfolio C = {}".format(delta_var_C.round(4)))
delta_var_T = cal_delta_var(portfolio, prices, "total")
print("Delta Normal VaR of Portfolio TOTAL = {}".format(delta_var_T.round(4)))


#Plot Historic
his_var_A, his_dis_A = cal_his_var(portfolio, prices, "A")
print("VaR of Portfolio A using Historical Simulation = {}".format(his_var_A.round(4)))
his_var_B, his_dis_B = cal_his_var(portfolio, prices, "B")
print("VaR of Portfolio B using Historical Simulation = {}".format(his_var_B.round(4)))
his_var_C, his_dis_C = cal_his_var(portfolio, prices, "C")
print("VaR of Portfolio C using Historical Simulation = {}".format(his_var_C.round(4)))
his_var_T, his_dis_T = cal_his_var(portfolio, prices, "total")
print("VaR of Portfolio Total using Historical Simulation = {}".format(his_var_T.round(4)))

fig, axes = plt.subplots(4, 1, figsize=(10, 30))
sns.distplot(his_dis_A, ax=axes[0])
sns.distplot(his_dis_B, ax=axes[1])
sns.distplot(his_dis_C, ax=axes[2])
sns.distplot(his_dis_T, ax=axes[3])

axes[0].axvline(x=-his_var_A, color='red', alpha=0.3)
axes[1].axvline(x=-his_var_B, color='red', alpha=0.3)
axes[2].axvline(x=-his_var_C, color='red', alpha=0.3)
axes[3].axvline(x=-his_var_T, color='red', alpha=0.3)

axes[0].set_title("VaR of Portfolio A using Historical Simulation = {}".format(his_var_A.round(4)))
axes[1].set_title("VaR of Portfolio B using Historical Simulation = {}".format(his_var_B.round(4)))
axes[2].set_title("VaR of Portfolio C using Historical Simulation = {}".format(his_var_C.round(4)))
axes[3].set_title("VaR of Portfolio Total using Historical Simulation = {}".format(his_var_T.round(4)))
plt.subplots_adjust(hspace=0.5)
plt.show()



#test normal distribution
def normal_percent(portfolio, prices, p_type, alpha=0.05):
    portfolio_price, dailyprices, holding = process_portfolio_data(portfolio, prices, p_type)
    returns = return_calculate(dailyprices).drop('Date',axis=1)
    n = 0
    for i in range(returns.shape[1]):
        k2, p = normaltest(returns.iloc[:, i])
        if p < alpha:
            n += 1
    return n / returns.shape[1]

#normal distribution percent
A_NOR = normal_percent(portfolio, prices, "A", alpha=0.05)
print('The proportion of Portfolio A that does not fit a normal distribution is {}'.format(A_NOR))

B_NOR = normal_percent(portfolio, prices, "B", alpha=0.05)
print('The proportion of Portfolio B that does not fit a normal distribution is {}'.format(B_NOR))

C_NOR = normal_percent(portfolio, prices, "C", alpha=0.05)
print('The proportion of Portfolio C that does not fit a normal distribution is {}'.format(C_NOR))

T_NOR = normal_percent(portfolio, prices, "total", alpha=0.05)
print('The proportion of Portfolio TOTAL that does not fit a normal distribution is {}'.format(T_NOR))

