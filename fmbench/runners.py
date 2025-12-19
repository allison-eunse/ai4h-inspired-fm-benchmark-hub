import os
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, mean_squared_error, r2_score

# Default stratification columns to look for in metadata
DEFAULT_STRATIFY_COLS = ["site", "sex", "scanner", "disease_stage", "ethnicity"]
# Age bins for numeric stratification
AGE_BINS = [(20, 40), (40, 60), (60, 80), (80, 100)]


def _safe_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute AUROC safely, returning 0.0 if not computable."""
    if len(np.unique(y_true)) < 2:
        return 0.0  # Can't compute AUROC with single class
    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return 0.0


def _compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """Compute classification metrics for a subset."""
    if len(y_true) == 0:
        return {"AUROC": 0.0, "Accuracy": 0.0, "F1-Score": 0.0, "N": 0}
    
    # AUROC
    if y_prob is not None:
        if y_prob.ndim == 2 and y_prob.shape[1] == 2:
            auroc = _safe_auroc(y_true, y_prob[:, 1])
        elif y_prob.ndim == 2:
            # Multi-class
            try:
                auroc = roc_auc_score(y_true, y_prob, multi_class="ovr")
            except ValueError:
                auroc = 0.0
        else:
            auroc = _safe_auroc(y_true, y_prob)
    else:
        auroc = 0.0
    
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    
    return {
        "AUROC": round(auroc, 4),
        "Accuracy": round(acc, 4),
        "F1-Score": round(f1, 4),
        "N": int(len(y_true))
    }


def _compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """Compute regression metrics for a subset."""
    if len(y_true) == 0:
        return {"MSE": 0.0, "R2": 0.0, "N": 0}
    
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        "MSE": round(float(mse), 4),
        "R2": round(float(r2), 4),
        "N": int(len(y_true))
    }


def _bin_age(age: int) -> str:
    """Convert age to a bin label."""
    for low, high in AGE_BINS:
        if low <= age < high:
            return f"age_{low}-{high}"
    return "age_unknown"


class BaseRunner:
    def __init__(
        self,
        model: Any,
        data_dir: str,
        stratify_cols: Optional[List[str]] = None
    ):
        self.model = model
        self.data_dir = data_dir
        self.stratify_cols = stratify_cols or DEFAULT_STRATIFY_COLS
        self.X = None
        self.y = None
        self.metadata = None
        
    def load_data(self):
        self.X = np.load(os.path.join(self.data_dir, "X.npy"))
        self.y = np.load(os.path.join(self.data_dir, "y.npy"))
        meta_path = os.path.join(self.data_dir, "metadata.csv")
        if os.path.exists(meta_path):
            self.metadata = pd.read_csv(meta_path)
            
    def run(self) -> Dict[str, Any]:
        raise NotImplementedError


class ClassificationRunner(BaseRunner):
    def run(self) -> Dict[str, Any]:
        if self.X is None:
            self.load_data()
        
        # Get predictions
        if hasattr(self.model, "predict_proba"):
            y_prob = self.model.predict_proba(self.X)
            y_pred = np.argmax(y_prob, axis=1)
        else:
            y_pred = self.model.predict(self.X)
            y_prob = None
        
        # Overall metrics
        overall = _compute_classification_metrics(self.y, y_pred, y_prob)
        
        # Build result
        result = {
            "AUROC": overall["AUROC"],
            "Accuracy": overall["Accuracy"],
            "F1-Score": overall["F1-Score"],
        }
        
        # Stratified metrics
        if self.metadata is not None:
            stratified = self._compute_stratified_metrics(y_pred, y_prob)
            if stratified:
                result["stratified"] = stratified
        
        return result
    
    def _compute_stratified_metrics(
        self,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray]
    ) -> Dict[str, Dict[str, Any]]:
        """Compute metrics for each stratum."""
        stratified = {}
        
        # Categorical stratification
        for col in self.stratify_cols:
            if col in self.metadata.columns:
                stratified[col] = {}
                for group in self.metadata[col].unique():
                    mask = self.metadata[col] == group
                    if mask.sum() < 5:  # Skip tiny groups
                        continue
                    y_true_sub = self.y[mask]
                    y_pred_sub = y_pred[mask]
                    y_prob_sub = y_prob[mask] if y_prob is not None else None
                    
                    metrics = _compute_classification_metrics(
                        y_true_sub, y_pred_sub, y_prob_sub
                    )
                    stratified[col][str(group)] = metrics
        
        # Age binning (if 'age' column exists)
        if "age" in self.metadata.columns:
            stratified["age_group"] = {}
            age_bins = self.metadata["age"].apply(_bin_age)
            for bin_label in age_bins.unique():
                mask = age_bins == bin_label
                if mask.sum() < 5:
                    continue
                y_true_sub = self.y[mask]
                y_pred_sub = y_pred[mask]
                y_prob_sub = y_prob[mask] if y_prob is not None else None
                
                metrics = _compute_classification_metrics(
                    y_true_sub, y_pred_sub, y_prob_sub
                )
                stratified["age_group"][bin_label] = metrics
        
        return stratified


class RegressionRunner(BaseRunner):
    def run(self) -> Dict[str, Any]:
        if self.X is None:
            self.load_data()
            
        y_pred = self.model.predict(self.X)
        
        # Ensure shapes match
        if y_pred.shape != self.y.shape:
            if y_pred.size == self.y.size:
                y_pred = y_pred.reshape(self.y.shape)
        
        # Overall metrics
        overall = _compute_regression_metrics(self.y, y_pred)
        
        result = {
            "MSE": overall["MSE"],
            "R2": overall["R2"],
        }
        
        # Stratified metrics
        if self.metadata is not None:
            stratified = self._compute_stratified_metrics(y_pred)
            if stratified:
                result["stratified"] = stratified
        
        return result
    
    def _compute_stratified_metrics(
        self,
        y_pred: np.ndarray
    ) -> Dict[str, Dict[str, Any]]:
        """Compute metrics for each stratum."""
        stratified = {}
        
        # Categorical stratification
        for col in self.stratify_cols:
            if col in self.metadata.columns:
                stratified[col] = {}
                for group in self.metadata[col].unique():
                    mask = self.metadata[col] == group
                    if mask.sum() < 5:
                        continue
                    y_true_sub = self.y[mask]
                    y_pred_sub = y_pred[mask]
                    
                    metrics = _compute_regression_metrics(y_true_sub, y_pred_sub)
                    stratified[col][str(group)] = metrics
        
        # Age binning
        if "age" in self.metadata.columns:
            stratified["age_group"] = {}
            age_bins = self.metadata["age"].apply(_bin_age)
            for bin_label in age_bins.unique():
                mask = age_bins == bin_label
                if mask.sum() < 5:
                    continue
                y_true_sub = self.y[mask]
                y_pred_sub = y_pred[mask]
                
                metrics = _compute_regression_metrics(y_true_sub, y_pred_sub)
                stratified["age_group"][bin_label] = metrics
        
        return stratified


def get_runner(
    runner_type: str,
    model: Any,
    data_dir: str,
    stratify_cols: Optional[List[str]] = None,
    **kwargs,
) -> BaseRunner:
    if runner_type == "classification":
        return ClassificationRunner(model, data_dir, stratify_cols)
    elif runner_type == "regression":
        return RegressionRunner(model, data_dir, stratify_cols)
    elif runner_type == "robustness":
        # Import here to avoid circular imports and make brainaug-lab optional
        from .robustness import RobustnessRunner
        return RobustnessRunner(model, data_dir, **kwargs)
    elif runner_type == "dna" or runner_type == "dna_classification":
        # DNA sequence classification runner
        from .dna_runner import DNASequenceRunner
        return DNASequenceRunner(model=model, data_dir=data_dir, **kwargs)
    else:
        raise ValueError(f"Unknown runner type: {runner_type}")
