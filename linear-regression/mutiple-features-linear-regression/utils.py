import numpy as np 
import copy

def load_house_data():
    data = np.loadtxt("./data/houses.txt", delimiter=',', skiprows=1)
    X = data[:, :4]
    y = data[:, 4]
    return X, y

def compute_cost(X, y, w, b):
    """
    Computes cost
    Args:
        X (ndarray(n, d))   : matrix of input variables
        w (ndarray(d,))     : model parameters
        b (scalar)          : model parameters
    Returns:
        cost (scalar)       : cost
    """
    n = X.shape[0]
    cost = 0.0

    for i in range(n):
        f = np.dot(w, X[i]) + b
        cost += (f - y[i]) ** 2
    
    cost /= 2 * n
    
    return cost

def compute_gradient(X, y, w, b):
    """ 
    Computes the gradient of cost J wrt parameters w, b
    Args:
        X (ndarray(n, d))   : matrix of input variables
        y (ndarray(n,))     : target values
        w (ndarray(d,))     : model parameters
        b (scalar)          : model parameters
    Returns:
        dJ_dw (ndarray(d,)) : gradient of J wrt w
        dJ_db (scalar)      : gradient of J wrt b
    """
    n, d = X.shape
    dJ_dw = np.zeros((d,))
    dJ_db = 0.0

    for i in range(n):
        err = (np.dot(w, X[i]) + b) - y[i]
        for j in range(d):
            dJ_dw[j] += err * X[i, j]
        dJ_db += err

    dJ_dw /= n
    dJ_db /= n
        
    return dJ_dw, dJ_db

def gradient_descent_houses(X, y, w_in, b_in, alpha, num_iters):
    """ 
    Performs batch gradient descent to learn parameters. Updates parameters
    by taking num_iters steps with learning rate alpha
    Args:
        X (ndarray(n, d))   : matrix of input variables
        y (ndarray(n,))     : target values
        w_in (ndarray(d,))  : initial values of w
        b_in (scalar)       : initial value of b
    Returns:
        w (ndarray(d,))     : updated values of w after running gradient descent
        b (scalar)          : updated value of w after running gradient descent
    """
    w = copy.deepcopy(w_in)
    b = b_in
    J_hist = []

    print(f"Iteration Cost          w0       w1       w2       w3       b       djdw0    djdw1    djdw2    djdw3    djdb  ")
    print(f"---------------------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|")

    for i in range(num_iters):
        dJ_dw, dJ_db = compute_gradient(X, y, w, b)

        w -= alpha * dJ_dw
        b -= alpha * dJ_db

        J_hist.append(compute_cost(X, y, w, b))

        if i % (num_iters/10) == 0:
            print(f"{i:9d} {J_hist[-1]:0.5e} {w[0]: 0.1e} {w[1]: 0.1e} {w[2]: 0.1e} {w[3]: 0.1e} {b: 0.1e} {dJ_dw[0]: 0.1e} {dJ_dw[1]: 0.1e} {dJ_dw[2]: 0.1e} {dJ_dw[3]: 0.1e} {dJ_db: 0.1e}")
    
    return w, b, J_hist