import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import random

def gen_data(setting, n, sig): 
    X = np.random.uniform(low=-1, high=1, size=n*20).reshape((n,20))
    
    if setting == 1: 
        mu_x = (X[:,0] * X[:,1] > 0 ) * (X[:,3]*(X[:,3]>0.5) + 0.5*(X[:,3]<=0.5)) + (X[:,0] * X[:,1] <= 0 ) * (X[:,3]*(X[:,3]<-0.5) - 0.5*(X[:,3]>-0.5))
        mu_x = mu_x * 4
        Y = mu_x + np.random.normal(size=n) * 1.5*sig
        return X, Y, mu_x
    
    if setting == 2:
        mu_x = (X[:,0] * X[:,1] + np.exp(X[:,3] - 1)) * 5
        Y = mu_x + np.random.normal(size=n) * 1.5 * sig 
        return X, Y, mu_x
    if setting == 3:
        mu_x = (X[:,0] * X[:,1] + np.exp(X[:,3] - 1)) * 5
        Y = mu_x + np.random.normal(size=n) * (5.5 - abs(mu_x))/2 * sig 
        return X, Y, mu_x
    
    if setting == 4:
        mu_x = (X[:,0] * X[:,1] + np.exp(X[:,3] - 1)) * 5
        sig_x = 0.25 * mu_x**2 * (np.abs(mu_x) < 2) + 0.5 * np.abs(mu_x) * (np.abs(mu_x) >= 1)
        Y = mu_x + np.random.normal(size=n) * sig_x * sig
        return X, Y, mu_x
    
    if setting == 5:
        mu_x = (X[:,0] * X[:,1] > 0 ) * (X[:,3]>0.5) * (0.25+X[:,3]) + (X[:,0] * X[:,1] <= 0 ) * (X[:,3]<-0.5) * (X[:,3]-0.25)
        mu_x = mu_x  
        Y = mu_x + np.random.normal(size=n) * sig
        return X, Y, mu_x
    
    if setting == 6:
        mu_x = (X[:,0] * X[:,1] + X[:,2]**2 + np.exp(X[:,3] - 1) - 1) * 2
        Y = mu_x + np.random.normal(size=n) * 1.5 * sig 
        return X, Y, mu_x
    
    if setting == 7:
        mu_x = (X[:,0] * X[:,1] + X[:,2]**2 + np.exp(X[:,3] - 1) - 1) * 2
        Y = mu_x + np.random.normal(size=n) * (5.5 - abs(mu_x))/2 * sig 
        return X, Y, mu_x
    
    if setting == 8:
        mu_x = (X[:,0] * X[:,1] + X[:,2]**2 + np.exp(X[:,3] - 1) - 1) * 2
        sig_x = 0.25 * mu_x**2 * (np.abs(mu_x) < 2) + 0.5 * np.abs(mu_x) * (np.abs(mu_x) >= 1)
        Y = mu_x + np.random.normal(size=n) * sig_x * sig
        return X, Y, mu_x
    
def gen_data_multi(n, sig): 
    X = np.random.uniform(low=-1, high=1, size=n*20).reshape((n,20))
    mu_x = (X[:,0] * X[:,1] > 0 ) * (X[:,3]*(X[:,3]>0.5) + 0.5*(X[:,3]<=0.5)) + (X[:,0] * X[:,1] <= 0 ) * (X[:,3]*(X[:,3]<-0.5) - 0.5*(X[:,3]>-0.5))
    mu_x = mu_x * 4
    Y_1 = mu_x + np.random.normal(size=n) * 1.5*sig
    mu_x = (X[:,0] * X[:,1] + np.exp(X[:,3] - 1)) * 5
    Y_2 = mu_x + np.random.normal(size=n) * 1.5 * sig 
    return X, Y_1, Y_2, mu_x


def get_regressor(reg_name):
    if reg_name == 'gbr':
        return GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=0)
    elif reg_name == 'rf':
        return RandomForestRegressor(max_depth=5, random_state=0)
    elif reg_name == 'svm':
        return SVR(kernel="rbf", gamma=0.1)
    raise ValueError(f"unknown regressor：{reg_name}")

def calc_pval(calib_scores, test_scores, calib_size):
    pval = np.zeros(len(test_scores))
    for j in range(len(test_scores)):
        lt = np.sum(calib_scores < test_scores[j])
        eq = np.sum(calib_scores == test_scores[j])
        pval[j] = (lt + random.uniform(0,1) * (eq + 1)) / (calib_size + 1)
    return pval.flatten()

def get_metrics(selected, y_test):
    if len(selected) == 0:
        return 0, 0, 0
    fdp = np.sum(y_test[selected] < 0) / len(selected)
    total_pos = np.sum(y_test >= 0)
    power = 0 if total_pos == 0 else np.sum(y_test[selected] >= 0) / total_pos
    return fdp, power, len(selected)

