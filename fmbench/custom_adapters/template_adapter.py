"""
Template for creating a custom model adapter.

Copy this file and implement the methods for your model.
"""

import numpy as np
from typing import Optional
from ..model_adapters import BaseModelAdapter


class MyCustomAdapter(BaseModelAdapter):
    """
    Adapter for MyCustomModel.
    
    Replace this with your model's description, paper link, repo, etc.
    """
    
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: str = "cpu",
        # Add your model-specific parameters here
        embedding_dim: int = 512,
        **kwargs
    ):
        super().__init__(checkpoint_path=checkpoint_path, device=device, **kwargs)
        self.embedding_dim = embedding_dim
        
    def load(self):
        """
        Load your model weights.
        
        This is called once before inference.
        """
        try:
            # Example: Load a PyTorch model
            # import torch
            # self.model = torch.load(self.checkpoint_path)
            # self.model.to(self.device)
            # self.model.eval()
            
            # Example: Load a HuggingFace model
            # from transformers import AutoModel
            # self.model = AutoModel.from_pretrained(self.checkpoint_path)
            
            print(f"[MyCustomModel] ✅ Model loaded from {self.checkpoint_path}")
            
        except Exception as e:
            print(f"[MyCustomModel] ⚠️ Could not load: {e}")
            # Fallback to random embeddings for testing
            self.model = None
            
        self.is_loaded = True
        
    def encode(self, X: np.ndarray) -> np.ndarray:
        """
        Generate embeddings for input samples.
        
        Args:
            X: Input array, shape (n_samples, n_features)
               For scRNA-seq: (n_cells, n_genes), values are normalized counts
               
        Returns:
            Embeddings, shape (n_samples, embedding_dim)
        """
        self._ensure_loaded()
        
        n_samples = len(X)
        
        if self.model is not None:
            # TODO: Replace with your actual model inference
            # Example for PyTorch:
            # import torch
            # with torch.no_grad():
            #     X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            #     embeddings = self.model.encode(X_tensor).cpu().numpy()
            # return embeddings
            pass
        
        # Fallback: return truncated/padded input as "embeddings"
        # (useful for testing the pipeline before real model is ready)
        if X.shape[1] >= self.embedding_dim:
            return X[:, :self.embedding_dim]
        else:
            # Pad with zeros
            embeddings = np.zeros((n_samples, self.embedding_dim), dtype=np.float32)
            embeddings[:, :X.shape[1]] = X
            return embeddings
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        (Optional) Generate class probabilities.
        
        If your model has a classification head, implement this.
        Otherwise, the base class will use encode() + simple logistic.
        
        Returns:
            Probabilities, shape (n_samples, n_classes)
        """
        # Option 1: Use the base class implementation (encode → logistic)
        return super().predict_proba(X)
        
        # Option 2: Use your model's classification head
        # self._ensure_loaded()
        # with torch.no_grad():
        #     logits = self.model.classify(torch.tensor(X))
        #     probs = torch.softmax(logits, dim=-1).numpy()
        # return probs


# To register this adapter, add to fmbench/model_adapters.py:
#
# from .custom_adapters.template_adapter import MyCustomAdapter
#
# ADAPTER_REGISTRY = {
#     ...
#     "my_custom_model": MyCustomAdapter,
# }
