import numpy as np

def load_house_data():
    data = np.loadtxt("./data/houses.txt", delimiter=",", skiprows=1)
    X = data[:,:4]
    y = data[:,4]
    return X, y

def add_intercept(X):
    """
    adds intercept to X

    Args:
        X (ndarray(n, d)): feature matrix, n examples, d features

    Returns:
        X_itc (ndarray(n, d+1)): new feature matrix with 1's in the 0th column
    """

    X_itc = np.zeros((X.shape[0], X.shape[1] + 1))
    X_itc[:, 0] = 1
    X_itc[:, 1:] = X

    return X_itc