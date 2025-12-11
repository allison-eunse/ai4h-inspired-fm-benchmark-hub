import numpy as np

class DummyClassifier:
    def __init__(self, random_seed=42):
        self.rng = np.random.default_rng(random_seed)
        
    def predict(self, X):
        # Random binary predictions
        return self.rng.integers(0, 2, size=X.shape[0])
        
    def predict_proba(self, X):
        # Random probabilities
        p = self.rng.random(size=X.shape[0])
        return np.vstack([1-p, p]).T

class DummyRegressor:
    def __init__(self, random_seed=42):
        self.rng = np.random.default_rng(random_seed)
        
    def predict(self, X):
        # Return random noise as prediction, matched to X's length
        # Assuming single target for simplicity, or matching input length if X is (N, D)
        return self.rng.standard_normal(size=X.shape[0])




