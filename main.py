import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from src.LogisticRegression import LogisticRegression
from utils import train_test_split, accuracy_score

# Generate a synthetic binary classification dataset
X,y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=42)

#Split the dataset using the custom train_test_split function
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training Custom Logistic Regression...")
# Train the custom Logistic Regression model
custom_model = LogisticRegression(learning_rate=0.01, n_iterations=1000)
custom_model.fit(X_train, y_train)
custom_predictions = custom_model.predict(X_test)

print(f"Custom Logistic Regression Accuracy: {accuracy_score(y_test, custom_predictions):.4f}\n")

print("Training Sklearn Logistic Regression...")
# Train the sklearn Logistic Regression model
sklearn_model = SklearnLogisticRegression(max_iter=1000)
sklearn_model.fit(X_train, y_train)
sklearn_predictions = sklearn_model.predict(X_test)

print(f"Sklearn Logistic Regression Accuracy: {accuracy_score(y_test, sklearn_predictions):.4f}")