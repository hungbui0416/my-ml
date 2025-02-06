class LinearModel(object):
    """
    Base class for linear models
    """
    def __init__(self, alpha=0.1, max_iter=1000, epsilon=1e-5, theta_0=None, hist=True):
        """
        Args:
            alpha       : step size for iterative solvers
            max_iter    : maximum number of iteratons
            epsilon     : threshold for determining convergence
            theta_0     : intial value for theta. If none, use zero vector
            hist        : print cost values during training
        """
        self.theta = theta_0
        self.alpha = alpha
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.hist = hist

        def fit(self, X, y):
            """
            Run solver to fit linear model
            Args:
                X (ndarray(n, d+1))     : training example inputs
                y (ndarray(n,))         : training example labels
            """

            raise NotImplementedError("Subclass of LinearModel must implement fit method")
        
        def predict(self, X):
            """
            Make predictions given inputs X
            Args:
                X (ndarray(n, d+1))     : training example inputs
            Returns:
                y_hat (ndarray(n,))     : predictions
            """
            raise NotImplementedError("Subclass of LinearModel must implement predict method")

        