import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.linear_model import LinearRegression as SklearnLinearRegression

from src import LinearRegression, LogisticRegression, train_test_split, mean_squared_error, r2_score

def benchmark_logistic_regression():
    print("="*50)
    print("🧪 BENCHMARKING: LOGISTIC REGRESSION")
    print("="*50)
    
    # Generate classification dataset
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Custom Model
    print("Training Custom Logistic Regression...")
    custom_model = LogisticRegression(learning_rate=0.01, n_iterations=1000)
    custom_model.fit(X_train, y_train)
    custom_preds = custom_model.predict(X_test)
    custom_acc = np.mean(custom_preds == y_test)
    print(f"Custom Accuracy: {custom_acc:.4f}")

    # Train Sklearn Model
    print("Training Sklearn Logistic Regression...")
    sklearn_model = SklearnLogisticRegression(max_iter=1000)
    sklearn_model.fit(X_train, y_train)
    sklearn_preds = sklearn_model.predict(X_test)
    sklearn_acc = np.mean(sklearn_preds == y_test)
    print(f"Sklearn Accuracy: {sklearn_acc:.4f}\n")


def benchmark_linear_regression():
    print("="*50)
    print("🧪 BENCHMARKING: LINEAR REGRESSION")
    print("="*50)
    
    # Generate regression dataset
    X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Custom Model
    print("Training Custom Linear Regression...")
    custom_model = LinearRegression(learning_rate=0.01, n_iterations=1000)
    custom_model.fit(X_train, y_train)
    custom_preds = custom_model.predict(X_test)
    print(f"Custom R2 Score: {r2_score(y_test, custom_preds):.4f}")
    print(f"Custom MSE: {mean_squared_error(y_test, custom_preds):.4f}")

    # Train Sklearn Model
    print("Training Sklearn Linear Regression...")
    sklearn_model = SklearnLinearRegression()
    sklearn_model.fit(X_train, y_train)
    sklearn_preds = sklearn_model.predict(X_test)
    print(f"Sklearn R2 Score: {r2_score(y_test, sklearn_preds):.4f}")
    print(f"Sklearn MSE: {mean_squared_error(y_test, sklearn_preds):.4f}\n")


if __name__ == "__main__":
    benchmark_linear_regression()
    benchmark_logistic_regression()