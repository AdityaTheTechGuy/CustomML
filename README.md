# CustomML

CustomML is a small, growing machine learning library built from scratch in Python with NumPy. It is designed as a study project for implementing core algorithms from first principles, comparing them against scikit-learn, and expanding over time as more models are added.

## What Is Included

- Vectorized NumPy implementations of linear regression and logistic regression
- A shared abstract base class for estimator-style models
- Utility functions for train/test splitting and common evaluation metrics
- Benchmark scripts that compare custom models with scikit-learn equivalents

## Project Structure

```text
CustomML/
├── benchmarks/
│   └── run_benchmarks.py
└── src/
    ├── __init__.py
    ├── base.py
    ├── linear_model.py
    └── utils.py
```

The repository currently exposes library code through `src/` and uses the benchmark script as the main runnable entry point.

## Requirements

- Python 3.10+
- NumPy
- scikit-learn

## Installation

Clone the repository and install the dependencies:

```bash
pip install numpy scikit-learn
```

If you prefer, you can also install them into a virtual environment first.

## Usage

### Train a model

```python
import numpy as np
from src.linear_model import LinearRegression, LogisticRegression

X = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
y = np.array([5.0, 8.0, 11.0])

model = LinearRegression(learning_rate=0.01, n_iterations=1000)
model.fit(X, y)
predictions = model.predict(X)
```

### Run benchmarks

```bash
python benchmarks/run_benchmarks.py
```

The benchmark script trains the custom linear and logistic regression models and compares them with scikit-learn on synthetic datasets.

## Roadmap

This repository is intentionally open-ended. As I continue studying machine learning, I plan to add more models and supporting utilities here, including additional estimators, evaluation helpers, and benchmark scripts.

Planned additions may include:

- More supervised learning models
- Regularization variants
- Classification metrics and diagnostics
- Data preprocessing helpers
- More benchmark comparisons

## Notes

- The codebase follows a simple, educational structure so each model stays easy to inspect and extend.
- Contributions and experiments are welcome as the library grows.
