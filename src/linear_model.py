import numpy as np
from .base import BaseEstimator


def _validate_supervised_inputs(X: np.ndarray, y: np.ndarray) -> None:
    if X.ndim != 2:
        raise ValueError("X must be a 2D array with shape (n_samples, n_features).")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array with shape (n_samples,).")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must contain the same number of samples.")

class LinearRegression(BaseEstimator):
    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000, method: str = 'gd'):
        super().__init__()
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.method = method

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegression':
        _validate_supervised_inputs(X, y)
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        
        if self.method == 'normal_equation':
            # Normal Equation: w = (X^T * X)^(-1) * X^T * y
            X_b = np.c_[np.ones((n_samples, 1)), X]  # Add bias term
            theta_best = np.linalg.pinv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
            self.bias = theta_best[0]
            self.weights = theta_best[1:]
        elif self.method != 'gd':
            raise ValueError("method must be either 'gd' or 'normal_equation'.")
        else:
            for _ in range(self.n_iterations):
                y_pred = np.dot(X, self.weights) + self.bias
                error = y_pred - y
                dw = (1 / n_samples) * np.dot(X.T, error)
                db = (1 / n_samples) * np.sum(error)

                self.weights -= self.lr * dw
                self.bias -= self.lr * db
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if X.ndim != 2:
            raise ValueError("X must be a 2D array with shape (n_samples, n_features).")
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
        _validate_supervised_inputs(X, y)
        if not np.all(np.isin(np.unique(y), [0, 1])):
            raise ValueError("y must contain binary class labels 0 and 1.")
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
        if X.ndim != 2:
            raise ValueError("X must be a 2D array with shape (n_samples, n_features).")
        linear_model = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_model)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return np.where(self.predict_proba(X) >= threshold, 1, 0)