import numpy as np

def load_dataset(url, intercept=False):
    data = np.loadtxt(url,delimiter=',',skiprows=1)
    X = data[:, :-1]
    y = data[:, -1]
    if intercept:
        X = add_intercept(X)
    
    return X, y

def add_intercept(X):
    intercept = np.ones(X.shape[0])
    X = np.c_[intercept, X]
    
    return X