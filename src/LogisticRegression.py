import numpy as np

class LogisticRegression:
    """
    A custom, vectorized Logistic Regression classifier using Gradient Descent.
    """
    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Helper method to compute the sigmoid activation."""
        return 1 / (1 + np.exp(-z))

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LogisticRegression':
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.n_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)

            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Update weights and bias
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
    
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return the predicted probabilities (between 0 and 1)."""
        linear_model = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_model)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return the final class labels (0 or 1)."""
        return (self.predict_proba(X) >= 0.5).astype(int)