import numpy as np

class LinearRegression:
    """
    A custom, vectorized implementation of linear regression model trained using gradient
    descent, designed to mimic scikit-learn API.
    """
    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegression':
        """
        Fit the linear regression model to the training data.

        Parameters:
        X (np.ndarray): Training data of shape (n_samples, n_features).
        y (np.ndarray): Target values of shape (n_samples,).

        Returns:
        self: Returns an instance of self.
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iterations):
            y_pred = np.dot(X, self.weights) + self.bias
            error = y_pred - y
            dw = (1 / n_samples) * np.dot(X.T, error)
            db = (1 / n_samples) * np.sum(error)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict target values for the given input data.

        Parameters:
        X (np.ndarray): Input data of shape (n_samples, n_features).

        Returns:
        np.ndarray: Predicted target values of shape (n_samples,).
        """
        return np.dot(X, self.weights) + self.bias
    
    