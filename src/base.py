import numpy as np
from abc import ABC, abstractmethod

class BaseEstimator(ABC):
    """Abstract Base Class for all machine learning models."""
    
    def __init__(self):
        self.weights = None
        self.bias = None

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the model to the training data."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels or values for given inputs."""
        pass