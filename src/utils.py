import numpy as np

def train_test_split(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_state: int = None):
    """
    Splits datasets into random train and test subsets.
    """
    if random_state is not None:
        np.random.seed(random_state)
        
    n_samples = X.shape[0]
    
    # 1. Create a shuffled array of indices
    shuffled_indices = np.random.permutation(n_samples)
    
    # 2. Determine the split threshold
    test_set_size = int(n_samples * test_size)
    
    # 3. Slice the indices
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    
    # 4. Split the data using array indexing
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculates the Mean Squared Error."""
    return float(np.mean((y_true.reshape(-1, 1) - y_pred.reshape(-1, 1)) ** 2))

def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculates the R-squared (Coefficient of Determination) score."""
    y_true = y_true.reshape(-1, 1)
    y_pred = y_pred.reshape(-1, 1)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - (ss_res / (ss_tot + 1e-8)))

def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculates the accuracy score."""
    return float(np.mean(y_true.reshape(-1, 1) == y_pred.reshape(-1, 1)))