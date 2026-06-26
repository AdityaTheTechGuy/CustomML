import numpy as np
from .base import BaseEstimator

class LinearRegression(BaseEstimator):
    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000):
        super().__init__()
        self.lr = learning_rate
        self.n_iterations = n_iterations
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegression':
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(self.n_iterations):
            y_pred = np.dot(X, self.weights) + self.bias
            error = y_pred - y
            dw = (1 / n_samples) * np.dot(X.T, error)
            db = (1 / n_samples) * np.sum(error)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.dot(X, self.weights) + self.bias


class LogisticRegression(BaseEstimator):
    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000):
        super().__init__()
        self.lr = learning_rate
        self.n_iterations = n_iterations

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        # Stable sigmoid preventing exponential overflow
        return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LogisticRegression':
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        
        for _ in range(self.n_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)

            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        linear_model = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_model)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return np.where(self.predict_proba(X) >= threshold, 1, 0)