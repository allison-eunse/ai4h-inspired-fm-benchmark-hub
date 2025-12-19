# Custom Model Adapters

This directory contains templates and examples for creating your own model adapters.

## Quick Start

1. Copy `template_adapter.py` to a new file (e.g., `my_model_adapter.py`)
2. Implement the `load()`, `encode()`, and optionally `predict_proba()` methods
3. Register your adapter in `fmbench/model_adapters.py`
4. Create a config YAML in `configs/`

## Adapter Interface

Your adapter must implement:

```python
class MyModelAdapter(BaseModelAdapter):
    def load(self):
        """Load model weights."""
        pass
    
    def encode(self, X: np.ndarray) -> np.ndarray:
        """
        Generate embeddings for input data.
        
        Args:
            X: Input array, shape (n_samples, n_features)
               For scRNA-seq: (n_cells, n_genes)
        
        Returns:
            Embeddings array, shape (n_samples, embedding_dim)
        """
        pass
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        (Optional) Generate class probabilities.
        
        If not implemented, base class will use encode() + simple classifier.
        """
        pass
```

## See Also

- `template_adapter.py` - Minimal template
- `example_scrnaseq_adapter.py` - Example for scRNA-seq models
- `../model_adapters.py` - Built-in adapters (Geneformer, BrainLM, etc.)
