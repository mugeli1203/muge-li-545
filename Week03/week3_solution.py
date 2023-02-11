import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

#Problem 1

#Read csv data file and drop the first column
all_data = pd.read_csv("DailyReturn.csv")
all_data = all_data.drop("Unnamed: 0",axis=1)

#Calculate the exponentially weighted covariance matrix

def expo_weighted_cov(ret_data,w_lambda):
    weight = np.zeros(60)
    for i in range(len(ret_data)):
        weight[len(ret_data)-1-i]  = (1-w_lambda)*w_lambda**i
    weight = weight/sum(weight)
    ret_means = ret_data - ret_data.mean()
    expo_w_cov = ret_means.T @ np.diag(weight) @ ret_means
    return expo_w_cov.values

cov_mtx = expo_weighted_cov(all_data, 0.97)


# Implement PCA to calculate the cumulative variance explained by eigenvalues
def pca(cov_mtx, n_eigenvalues):
    eigenvalues, eigenvectors = np.linalg.eig(cov_mtx)
    #Keep those positive eigenvalues and corresponding eigenvectors
    p_idx = eigenvalues > 1e-8
    eigenvalues = eigenvalues[p_idx]
    eigenvectors = eigenvectors[:, p_idx]
    #Sort
    s_idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[s_idx]
    eigenvectors = eigenvectors[:, s_idx]
    
    cum_var = np.sum(eigenvalues[0:n_eigenvalues]) / np.sum(eigenvalues)
    return cum_var

#print(pca(cov_mtx, 40))

#Make plot
lambdas = [0.6, 0.7, 0.8, 0.9, 0.97, 0.99]
for w_lambda in lambdas:
    cov_mtx = expo_weighted_cov(all_data, w_lambda)
    cul_var = []
    for i in range(101):
        cul_var.append(pca(cov_mtx, i))
    plt.plot(cul_var, label = f"λ = {w_lambda}")
    plt.xlabel("Number of eigenvalues")
    plt.ylabel("Cumulative variance")
    plt.title("Cumulative Varience given different λ")
    plt.legend()


#Problem 2
#Cholesky Factorization for PSD matrix

def chol_psd(cov_matrix):

    cov_mtx = cov_matrix
    n = cov_mtx.shape[0]
    root = np.zeros_like(cov_mtx)
    for j in range(n):
        s = 0.0
        if j > 0:
            # calculate dot product of the preceeding row values
            s = np.dot(root[j, :j], root[j, :j])
        temp = cov_mtx[j, j] - s
        if 0 >= temp >= -1e-8:
            temp = 0.0
        root[j, j] = np.sqrt(temp)
        if root[j, j] == 0.0:
            # set the column to 0 if we have an eigenvalue of 0
            root[j + 1:, j] = 0.0
        else:
            ir = 1.0 / root[j, j]
            for i in range(j + 1, n):
                s = np.dot(root[i, :j], root[j, :j])
                root[i, j] = (cov_mtx[i, j] - s) * ir
    return root
#print(chol_psd(cov_mtx))

# Rebonato and Jackel deal with non-PSD matrix
def near_psd(mtx, epsilon=0.0):
    n = mtx.shape[0]

    invSD = None
    out = mtx.copy()

    # calculate the correlation matrix if we got a covariance
    if (np.diag(out) == 1.0).sum() != n:
        invSD = np.diag(1 / np.sqrt(np.diag(out)))
        out = invSD.dot(out).dot(invSD)

    # SVD, update the eigen value and scale
    vals, vecs = np.linalg.eigh(out)
    vals = np.maximum(vals, epsilon)
    T = np.reciprocal(np.square(vecs).dot(vals))
    T = np.diag(np.sqrt(T))
    l = np.diag(np.sqrt(vals))
    B = T.dot(vecs).dot(l)
    out = np.matmul(B, np.transpose(B))

    # Add back the variance
    if invSD is not None:
        invSD = np.diag(1 / np.diag(invSD))
        out = invSD.dot(out).dot(invSD)

    return out


#Higham deal with non-PSD matrix
#First projection
def Pu(mtx):
    new_mtx = mtx.copy()
    for i in range(len(mtx)):
        for j in range(len(mtx[0])):
            if i == j:
                new_mtx[i][j] = 1
    return new_mtx
#Second projection
def Ps(mtx, w):
    mtx = np.sqrt(w)@mtx@np.sqrt(w)
    vals, vecs = np.linalg.eigh(mtx)
    vals = np.array([max(i,0) for i in vals])
    new_mtx = np.sqrt(w)@ vecs @ np.diag(vals) @ vecs.T @ np.sqrt(w)
    return new_mtx
#Calculate Frobenius Norm

def fnorm(mtxa, mtxb):
    s = mtxa - mtxb
    norm = 0
    for i in range(len(s)):
        for j in range(len(s[0])):
            norm +=s[i][j]**2
    return norm


def higham_psd(mtx, w, max_iteration = 1000, tol = 1e-8):
    r0 = np.inf
    Y = mtx
    S = np.zeros_like(Y)
 
    invSD = None
    if np.count_nonzero(np.diag(Y) == 1.0) != mtx.shape[0]:
        invSD = np.diag(1.0 / np.sqrt(np.diag(Y)))
        Y = invSD.dot(Y).dot(invSD)
    C = Y.copy()
    
    for i in range(max_iteration):
        R = Y - S
        X = Ps(R, w)
        S = X - R
        Y = Pu(X)
        r = fnorm(Y, C)
        minval = np.linalg.eigvals(Y).min()
        if abs(r - r0) < tol and minval > -1e-8:
            break
        else:
            r0 = r
    
    if invSD is not None:
        invSD = np.diag(1 / np.diag(invSD))
        Y =invSD.dot(Y).dot(invSD)
    return Y

#Generate a non-psd correclation matrix that is 500x500

n = 500
sigma = np.full((n,n),0.9)
for i in range(n):
    sigma[i,i]=1.0
sigma[0,1] = 0.7357
sigma[1,0] = 0.7357

#Fix and Confirm matrix is PSD or not
def psd(mtx):
    eigenvalues = np.linalg.eigh(mtx)[0]
    return np.all(eigenvalues >= -1e-8)

print("Matrix generated is PSD:", psd(sigma))
print("Matrix fixed by near_psd is PSD:", psd(near_psd(sigma)))
weight = np.identity(len(sigma))
print("Matrix fixed by Highan is PSD:", psd(higham_psd(sigma, weight)))

#Compare the Frobenius Norm between near_psd and higham
range_n = [20, 50, 100, 200, 500, 1000]

fnorms_near = []
fnorms_higham = []
for n in range_n:
    sigma = np.full((n,n),0.9)
    for i in range(n):
        sigma[i,i]=1.0
    sigma[0,1] = 0.7357
    sigma[1,0] = 0.7357
    sigma_near = near_psd(sigma)
    near_norm = fnorm(sigma, sigma_near)
    fnorms_near.append(near_norm)
    
    weight = np.identity(len(sigma))
    sigma_higham = higham_psd(sigma, weight)
    higham_norm = fnorm(sigma, sigma_higham)
    fnorms_higham.append(higham_norm)
    
plt.plot(range_n,fnorms_near, label = "near_psd")
plt.plot(range_n,fnorms_higham, label = "Higham")
plt.xlabel("N")
plt.ylabel("Forbenius Norm")
plt.title("Forbenius Norm given different N")
plt.legend()

#Compare the run time between near_psd and higham
times_near = []
times_higham =[]
for n in range_n:
    sigma = np.full((n,n),0.9)
    for i in range(n):
        sigma[i,i]=1.0
    sigma[0,1] = 0.7357
    sigma[1,0] = 0.7357

    start_t = time.time()
    sigma_near = near_psd(sigma)
    stop_t = time.time()
    times_near.append(stop_t - start_t)

    start_t = time.time()
    weight = np.identity(len(sigma))
    sigma_higham = higham_psd(sigma, weight)
    stop_t = time.time()
    times_higham.append(stop_t - start_t)

plt.plot(range_n,times_near, label = "near_psd")
plt.plot(range_n,times_higham, label = "Higham")
plt.xlabel("N")
plt.ylabel("Run Time")
plt.title("Run time given different N")
plt.legend()


#Problem 3
#Multivariate normal simulation
#Direct simulation
def multiv_normal_sim(cov_mtx, n_draws):
    L = chol_psd(cov_mtx)
    std_normals = np.random.normal(size=(len(cov_mtx), n_draws))
    return np.transpose((L @ std_normals) + 0)

#PCA simulation
def pca_sim(cov_mtx, n_draws, percent_explain):
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

#Calculate 4 different covariance matrix
# Pearson corr mtx + EW var vec
def PS_corr_mtx_EW_var_vec(ret_data, w_lambda=0.97):
    ew_cov_mtx = expo_weighted_cov(ret_data, w_lambda)
    #np.diag(np.reciprocal(np.sqrt(np.diag(ew_cov_mtx)))) 
    std_dev = np.sqrt(np.diag(ew_cov_mtx))
    corr = np.corrcoef(ret_data.T)
    return np.diag(std_dev) @ corr @ np.diag(std_dev).T

#EW corr mtx + PS var vec
def EW_corr_mtx_PS_var_vec(ret_data, w_lambda=0.97):
    ew_cov_mtx = expo_weighted_cov(ret_data, w_lambda)

    invSD = np.diag(np.reciprocal(np.sqrt(np.diag(ew_cov_mtx))))
    corr = invSD.dot(ew_cov_mtx).dot(invSD)

    var = np.var(ret_data)
    std_dev = np.sqrt(var)
    return np.diag(std_dev) @ corr @ np.diag(std_dev).T

# 4 different covairance matrix
ew_cov_mtx = expo_weighted_cov(all_data, 0.97)
#print(ew_cov_mtx)
ps_cov_mtx = np.cov(all_data.T)
pearson_corr_ew_var_mtx = PS_corr_mtx_EW_var_vec(all_data)
ew_corr_pearson_var_mtx = EW_corr_mtx_PS_var_vec(all_data)
#print(ew_corr_pearson_var_mtx)
cov_mtx_list = [ew_cov_mtx, ps_cov_mtx, pearson_corr_ew_var_mtx, ew_corr_pearson_var_mtx]
index_list = ["ew_cov_mtx", "ps_cov_mtx", "pearson_corr_ew_var_mtx", "ew_corr_pearson_var_mtx"]
#for i in cov_mtx_list:
    #print(type(i))
#print(chol_psd(ew_cov_mtx))
#Direct simulation using 4 cov mtx and comparison
def direct_sim_cpr(cov_mtx):
    n_draws = 25000
    start_time = time.time()
    simulations = multiv_normal_sim(cov_mtx, n_draws)
    stop_time = time.time()
    sim_cov_mtx = np.cov(simulations.T)
    norm = fnorm(sim_cov_mtx, cov_mtx)
    return stop_time - start_time, norm
#print(direct_sim_cpr(ps_cov_mtx))

#PCA simulation using 4 cov mtx and comparison
def pca_sim_cpr(cov_mtx, percent_explain):
    n_draws = 25000
    start_time = time.time()
    simulations = pca_sim(cov_mtx, n_draws, percent_explain)
    stop_time = time.time()
    sim_cov_mtx = np.cov(simulations.T)
    norm = fnorm(sim_cov_mtx, cov_mtx)
    return stop_time - start_time, norm

#ew_cov_mtx

for c in range(len(cov_mtx_list)):

    runtime = []
    norm = []

    dir_time, dir_norm = direct_sim_cpr(cov_mtx_list[c])
    runtime.append(dir_time)
    norm.append(dir_norm)

    per_exp = [1, 0.75, 0.5]
    for i in per_exp:
        pca_time, pca_norm = pca_sim_cpr(cov_mtx_list[c], i)
        runtime.append(pca_time)
        norm.append(pca_norm)
    txt = ["direct", "1.0pca", "0.75pca", "0.5pca"]
    plt.scatter(runtime, norm, label = index_list[c])
    for j in range(len(runtime)):
        plt.annotate(txt[j], xy=(runtime[j], norm[j]))
    plt.xlabel("Run Time")
    plt.ylabel("Frobenius Norm")
    plt.title("Trade off between Run time and Frobenius Norm(ew_cov_mtx)")
    plt.legend()
#print(ew_runtime)
#print(ew_norm)

'''
#ps_cov_mtx
ps_runtime = []
ps_norm = []

dir_time, dir_norm = direct_sim_cpr(ew_corr_pearson_var_mtx)
ps_runtime.append(dir_time)
ps_norm.append(dir_norm)

per_exp = [1, 0.75, 0.5]
for i in per_exp:
    pca_time, pca_norm = pca_sim_cpr(ew_corr_pearson_var_mtx, i)
    ps_runtime.append(pca_time)
    ps_norm.append(pca_norm)
plt.plot(ps_runtime,ps_norm)
plt.xlabel("Run Time")
plt.ylabel("Frobenius Norm")
plt.title("Trade off between Run time and Frobenius Norm")
print(ps_runtime)
print(ps_norm)
'''