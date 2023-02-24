import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ml_quant_risk import MC, VaR
from scipy.stats import t, norm


#Problem 1
p1_data = pd.read_csv('problem1.csv')

#Normal Distribution
var_nor_p1, dis_nor_p1= VaR.norml_var(p1_data)
es_nor_es = VaR.calculate_es(var_nor_p1, dis_nor_p1)

print('Under Normal Distribution, VaR is ', var_nor_p1.round(4), 'ES is ',es_nor_es.round(4))

#T Distribution
var_t_p1, dis_t_p1= VaR.MLE_T_var(p1_data)
es_t_es = VaR.calculate_es(var_t_p1, dis_t_p1)

print('Under a Generalized T Distribution, VaR is ', var_t_p1.round(4), 'ES is ',es_t_es.round(4))

#Plot all into one graph
plt.hist(dis_nor_p1, bins=30, density=True, alpha=0.5, label='Normal Distribution')
sns.kdeplot(dis_nor_p1, color='blue', linewidth=1)
plt.axvline(-var_nor_p1, label='Normal VaR', color='blue', linestyle='-')
plt.axvline(-es_nor_es, label='Normal ES', color='blue', linestyle='--')

plt.hist(dis_t_p1, bins=30, density=True, alpha=0.5, label='T Distribution')
sns.kdeplot(dis_t_p1, color='red', linewidth=1)
plt.axvline(-var_t_p1, label='T VaR', color='red', linestyle='-')
plt.axvline(-es_t_es, label='T ES', color='red', linestyle='--')

plt.xlim(-0.3, 0.3)

plt.legend(loc='upper right')
plt.ylabel('Density')
plt.title('Normal Distribution vs T Distribution')

plt.show()

#Problem 2

#Test return_calculate()
df = pd.read_csv("DailyPrices.csv")
all_ret = VaR.return_calculate(df)
all_ret.drop('Date', axis=1, inplace=True)
print(all_ret)

#Test covariance estimation
ew_cov_mtx = MC.expo_weighted_cov(all_ret, 0.97)
pearson_corr_ew_var_mtx = MC.PS_corr_mtx_EW_var_vec(all_ret)
ew_corr_pearson_var_mtx = MC.EW_corr_mtx_PS_var_vec(all_ret)

print(ew_cov_mtx)
print(pearson_corr_ew_var_mtx)
print(ew_corr_pearson_var_mtx)

#Test non-PSD fixs
n = 500
sigma = np.full((n,n),0.9)
for i in range(n):
    sigma[i,i]=1.0
sigma[0,1] = 0.7357
sigma[1,0] = 0.7357
print('Is PSD? ',MC.psd(sigma))

new_mtx_near = MC.near_psd(sigma)
print('Is PSD? ',MC.psd(new_mtx_near))

weight = np.identity(len(sigma))
new_mtx_ham = MC.higham_psd(sigma, weight)
print('Is PSD? ',MC.psd(new_mtx_ham))

#Test simulation process
multi_nor_sims = MC.multiv_normal_sim(ew_cov_mtx, 100)
pca_sims = MC. pca_sim(ew_cov_mtx, 100, 1)
print(multi_nor_sims)
print(pca_sims)

#Test VaR calculations
'''
norml_var(returns, alpha = 0.05, N = 10000), 
MLE_T_var(returns, alpha = 0.05, N = 10000), 
and calculate_es(var, sim_data) have been tested in Porblem 1
'''
META = all_ret['META']
var_nor_ew =VaR.norml_ew_var(META)
print(var_nor_ew)

var_ar1 = VaR.ar1_var(META)
print(var_ar1)

var_his = VaR.his_var(META)
print(var_his)

#Test portfolio var
portfolio = pd.read_csv('portfolio.csv')
prices = pd.read_csv('DailyPrices.csv')

A_data_p, A_day_p, A_holding = VaR.process_portfolio_data(portfolio, prices, 'A')
print(A_data_p, A_day_p, A_holding)

MC_var_A, MC_dis_A = VaR.cal_MC_var(portfolio, prices, "A")
print(MC_var_A, MC_dis_A)

delta_var_A = VaR.cal_delta_var(portfolio, prices, "B")
print(delta_var_A )

his_var_C, his_dis_C = VaR.cal_his_var(portfolio, prices, "C")
print(his_var_C, his_dis_C)

#Problem 3 
def copula_var(portfolio, prices, p_type):
    p_latest_price, a_daily_prices, holdings = VaR.process_portfolio_data(portfolio, prices, p_type)
    assets_returns = VaR.return_calculate(a_daily_prices)
    assets_returns.drop('Date', axis=1, inplace=True)
    zero_mean_returns = assets_returns - assets_returns.mean()
    #print(zero_mean_returns)

    returns_transf = zero_mean_returns.copy()
    #Fit data with generalized T
    for asset in returns_transf.columns.tolist():
        result = t.fit(zero_mean_returns[asset], method="MLE")
        df = result[0]
        loc = result[1]
        scale = result[2]
        #Tansfer to U
        returns_transf[asset] = t.cdf(zero_mean_returns[asset], df=df, loc=loc, scale=scale)
    #print(returns_transf)

    #Transfer to Z
    returns_transf = pd.DataFrame(norm.ppf(returns_transf), index=returns_transf.index, columns=returns_transf.columns)
    #print(returns_transf)

    #Calculate correlation mtx
    spearman_corr_mtx = returns_transf.corr(method='spearman')
    # #transfer correlation mtx to covariance mtx
    # std_vector = np.sqrt(np.diag(spearman_corr_mtx))
    # cov_matrix = spearman_corr_mtx * np.outer(std_vector, std_vector)
    #print(cov_matrix)
    #Simulation n draws
    simulations = MC.pca_sim(spearman_corr_mtx, 10000, percent_explain = 1)
    simulations = pd.DataFrame(simulations, columns=returns_transf.columns)
    #print(simulations)

    #Convert simulations to standard normal cdf
    returns_back = pd.DataFrame(norm.cdf(simulations), index=simulations.index, columns=simulations.columns)
    #print(returns_back)
    
    #Convert cdf back to returns
    for asset in returns_transf.columns.tolist():
        result = t.fit(zero_mean_returns[asset], method="MLE")
        df = result[0]
        loc = result[1]
        scale = result[2]
        returns_back[asset] = t.ppf(returns_back[asset], df=df, loc=loc, scale=scale)
    #print(returns_back)
    
    sim_returns = np.add(returns_back, assets_returns.mean())
    a_daily_prices = a_daily_prices.drop('Date', axis=1)
    sim_change = np.dot(sim_returns * a_daily_prices.tail(1).values.reshape(a_daily_prices.shape[1]),holdings)

    var = np.percentile(sim_change, 0.05*100) * (-1)
    es = VaR.calculate_es(var, sim_change)
    return var, es, sim_change, p_latest_price



A_var, A_es, A_dis, A_p= copula_var(portfolio, prices, 'A')
print('Porfolio A')
print('Price: ', A_p)
print('VaR: ', A_var)
print('ES: ', A_es)

plt.hist(A_dis, bins=30, density=True, alpha=0.5)
sns.kdeplot(A_dis, color='blue', linewidth=1)
plt.axvline(-A_var, label='Normal VaR', color='blue', linestyle='-')
plt.axvline(-A_es, label='Normal ES', color='blue', linestyle='--')

plt.ylabel('Density')
plt.title('Simultions of Portfolio A fitted with T Distribution')

plt.show()


B_var, B_es, B_dis, B_p= copula_var(portfolio, prices, 'B')
print('Porfolio B')
print('Price: ', B_p)
print('VaR: ', B_var)
print('ES: ', B_es)

plt.hist(B_dis, bins=30, density=True, alpha=0.5)
sns.kdeplot(B_dis, color='blue', linewidth=1)
plt.axvline(-B_var, label='Normal VaR', color='blue', linestyle='-')
plt.axvline(-B_es, label='Normal ES', color='blue', linestyle='--')

plt.ylabel('Density')
plt.title('Simultions of Portfolio B fitted with T Distribution')

plt.show()


C_var, C_es, C_dis, C_p= copula_var(portfolio, prices, 'C')
print('Porfolio C')
print('Price: ', C_p)
print('VaR: ', C_var)
print('ES: ', C_es)

plt.hist(C_dis, bins=30, density=True, alpha=0.5)
sns.kdeplot(C_dis, color='blue', linewidth=1)
plt.axvline(-C_var, label='Normal VaR', color='blue', linestyle='-')
plt.axvline(-C_es, label='Normal ES', color='blue', linestyle='--')

plt.ylabel('Density')
plt.title('Simultions of Portfolio C fitted with T Distribution')

plt.show()


T_var, T_es, T_dis, T_p= copula_var(portfolio, prices, 'total')
print('Porfolio TOTAL')
print('Price: ', T_p)
print('VaR: ', T_var)
print('ES: ', T_es)

plt.hist(T_dis, bins=30, density=True, alpha=0.5)
sns.kdeplot(T_dis, color='blue', linewidth=1)
plt.axvline(-T_var, label='Normal VaR', color='blue', linestyle='-')
plt.axvline(-T_es, label='Normal ES', color='blue', linestyle='--')

plt.ylabel('Density')
plt.title('Simultions of Portfolio TOTAL fitted with T Distribution')

plt.show()