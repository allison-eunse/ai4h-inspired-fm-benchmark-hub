"""
AI4H-Aligned Local Benchmark Runner

This module allows researchers to run benchmarks on their OWN models locally,
without sharing model weights or code. Only the evaluation results are submitted.

## AI4H Alignment (DEL3 Section 5.2)

The ITU/WHO FG-AI4H framework recommends:
1. **Local Evaluation**: Models stay on researcher's infrastructure
2. **Standardized Interface**: Simple predict/encode methods
3. **Privacy-Preserving**: Only metrics are shared, not model weights
4. **Reproducible**: Seed-controlled evaluations with standard datasets

## Usage

1. Researcher wraps their model:

    ```python
    class MyModel:
        def predict(self, X):
            return self.model(X)
        def predict_proba(self, X):
            return self.model.predict_proba(X)
    ```

2. Create config YAML:

    ```yaml
    model_id: my_awesome_model
    type: python_class
    import_path: "my_package.model:MyModel"
    ```

3. Run locally:

    ```bash
    python -m fmbench run --suite SUITE-TOY-CLASS --model my_config.yaml --out results/
    ```

4. Submit results/eval.yaml via GitHub Issue
"""

import os
import sys
import yaml
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


@runtime_checkable
class BenchmarkableModel(Protocol):
    """
    Protocol defining the interface a model must implement to be benchmarked.
    
    Researchers only need to implement these methods on their model class.
    The actual model architecture and weights remain private.
    """
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions for input data.
        
        Args:
            X: Input data array of shape (n_samples, ...)
            
        Returns:
            Predictions array of shape (n_samples,)
        """
        ...
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Generate probability predictions (optional, for classification).
        
        Args:
            X: Input data array of shape (n_samples, ...)
            
        Returns:
            Probability array of shape (n_samples, n_classes)
        """
        ...


class SimpleModelWrapper:
    """
    Simple wrapper to make any callable benchmarkable.
    
    Example:
        >>> model = SimpleModelWrapper(lambda x: (x.mean(axis=1) > 0).astype(int))
        >>> predictions = model.predict(X)
    """
    
    def __init__(self, predict_fn, predict_proba_fn=None):
        self._predict = predict_fn
        self._predict_proba = predict_proba_fn
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._predict(X)
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._predict_proba:
            return self._predict_proba(X)
        # Fallback: convert predictions to pseudo-probabilities
        preds = self.predict(X)
        probs = np.zeros((len(X), 2))
        probs[np.arange(len(X)), preds.astype(int)] = 1.0
        return probs


def validate_model_interface(model: Any) -> List[str]:
    """
    Validate that a model implements the required interface.
    
    Returns list of missing methods (empty if valid).
    """
    issues = []
    
    if not hasattr(model, 'predict'):
        issues.append("Missing required method: predict(X)")
    elif not callable(getattr(model, 'predict')):
        issues.append("predict must be callable")
        
    if not hasattr(model, 'predict_proba'):
        issues.append("Missing optional method: predict_proba(X) - will use fallback")
        
    return issues


def generate_submission_yaml(
    model_id: str,
    benchmark_id: str,
    metrics: Dict[str, Any],
    dataset_id: str,
    hardware_info: Optional[str] = None,
    notes: Optional[str] = None,
) -> str:
    """
    Generate a submission-ready evaluation YAML.
    
    This is what researchers submit - contains NO model weights or code,
    only metrics and metadata per AI4H DEL3 requirements.
    """
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    
    submission = {
        "eval_id": f"{benchmark_id}-{model_id}-{timestamp}",
        "benchmark_id": benchmark_id,
        "model_ids": {
            "candidate": model_id
        },
        "dataset_id": dataset_id,
        "run_metadata": {
            "date": datetime.utcnow().strftime("%Y-%m-%d"),
            "runner": "fmbench-local",
            "runner_version": "1.0.0",
            "hardware": hardware_info or "not specified",
        },
        "metrics": metrics,
        "notes": notes,
        "status": "Completed",
        
        # AI4H compliance markers
        "ai4h_compliance": {
            "local_evaluation": True,
            "model_weights_shared": False,
            "reproducible_seed": True,
        }
    }
    
    return yaml.dump(submission, default_flow_style=False, sort_keys=False)


def print_submission_instructions(eval_path: str):
    """Print instructions for submitting results."""
    print("\n" + "=" * 60)
    print("📤 SUBMISSION INSTRUCTIONS (AI4H-Aligned)")
    print("=" * 60)
    print(f"""
Your evaluation results have been saved to:
  {eval_path}

To add your model to the leaderboard:

1. Go to: https://github.com/allison-eunse/ai4h-inspired-fm-benchmark-hub/issues/new

2. Select template: "📊 Benchmark Submission"

3. Paste the contents of your eval.yaml file

4. The maintainer will review and add your results

✅ What IS submitted:
   - Model name/ID
   - Benchmark metrics (AUROC, Accuracy, etc.)
   - Run metadata (date, hardware)

❌ What is NOT submitted:
   - Model weights
   - Model code
   - Training data
   - Private information

This workflow is aligned with ITU/WHO FG-AI4H DEL3 Section 5.2:
"Local evaluation with standardized result reporting"
""")


# Example model wrapper for researchers
EXAMPLE_WRAPPER_CODE = '''
# Example: How to wrap your model for fmbench

import numpy as np

class MyModelWrapper:
    """Wrap your model to work with fmbench benchmarks."""
    
    def __init__(self):
        # Load your model here
        # self.model = load_my_model("path/to/checkpoint")
        pass
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions.
        
        Args:
            X: Input array of shape (n_samples, n_features)
            
        Returns:
            Predictions of shape (n_samples,)
        """
        # Your inference code here
        # return self.model.predict(X)
        return np.zeros(len(X))  # placeholder
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Generate probability predictions.
        
        Args:
            X: Input array of shape (n_samples, n_features)
            
        Returns:
            Probabilities of shape (n_samples, n_classes)
        """
        # Your inference code here
        # return self.model.predict_proba(X)
        return np.ones((len(X), 2)) * 0.5  # placeholder


# Create config file (my_model_config.yaml):
"""
model_id: my_awesome_model
type: python_class
import_path: "my_wrapper:MyModelWrapper"
init_kwargs: {}
"""

# Run benchmark:
# python -m fmbench run --suite SUITE-TOY-CLASS --model my_model_config.yaml --out results/
'''


def get_example_wrapper() -> str:
    """Return example wrapper code for researchers."""
    return EXAMPLE_WRAPPER_CODE

