"""
DNA Sequence Classification Runner

Reads DNA TSV files (sequence, label) and runs classification benchmarks
with support for multiple encoding methods.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Callable
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from collections import Counter

# =============================================================================
# DNA ENCODING UTILITIES
# =============================================================================

# Standard nucleotide mappings
NUCLEOTIDE_MAP = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
NUCLEOTIDE_COMPLEMENT = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}

def one_hot_encode(sequence: str, max_len: Optional[int] = None) -> np.ndarray:
    """
    One-hot encode a DNA sequence.
    
    Args:
        sequence: DNA sequence string (A, C, G, T, N)
        max_len: Maximum length (pad/truncate if specified)
    
    Returns:
        One-hot encoded array of shape (seq_len, 4) or (max_len, 4)
    """
    seq = sequence.upper()
    if max_len:
        if len(seq) > max_len:
            seq = seq[:max_len]
        elif len(seq) < max_len:
            seq = seq + 'N' * (max_len - len(seq))
    
    encoding = np.zeros((len(seq), 4), dtype=np.float32)
    for i, nuc in enumerate(seq):
        if nuc in 'ACGT':
            encoding[i, NUCLEOTIDE_MAP[nuc]] = 1.0
        # N is left as all zeros
    return encoding

def kmer_encode(sequence: str, k: int = 6) -> np.ndarray:
    """
    Encode DNA sequence as k-mer frequency vector.
    
    Args:
        sequence: DNA sequence string
        k: k-mer length (default 6)
    
    Returns:
        K-mer frequency vector of shape (4^k,)
    """
    seq = sequence.upper().replace('N', '')
    n_kmers = 4 ** k
    kmer_counts = np.zeros(n_kmers, dtype=np.float32)
    
    # Build k-mer to index mapping
    bases = 'ACGT'
    
    for i in range(len(seq) - k + 1):
        kmer = seq[i:i+k]
        if all(b in bases for b in kmer):
            # Convert k-mer to index
            idx = 0
            for j, base in enumerate(kmer):
                idx += bases.index(base) * (4 ** (k - 1 - j))
            kmer_counts[idx] += 1
    
    # Normalize to frequencies
    total = kmer_counts.sum()
    if total > 0:
        kmer_counts /= total
    
    return kmer_counts

def character_encode(sequence: str, max_len: Optional[int] = None) -> np.ndarray:
    """
    Encode DNA as integer character IDs.
    
    Args:
        sequence: DNA sequence string
        max_len: Maximum length (pad/truncate if specified)
    
    Returns:
        Integer array of shape (seq_len,) or (max_len,)
    """
    seq = sequence.upper()
    if max_len:
        if len(seq) > max_len:
            seq = seq[:max_len]
        elif len(seq) < max_len:
            seq = seq + 'N' * (max_len - len(seq))
    
    encoding = np.array([NUCLEOTIDE_MAP.get(c, 4) for c in seq], dtype=np.int32)
    return encoding

def get_encoder(encoding_type: str, **kwargs) -> Callable:
    """
    Get an encoding function by name.
    
    Args:
        encoding_type: One of 'one_hot', 'kmer', 'character'
        **kwargs: Arguments passed to encoder (e.g., k=6 for k-mer)
    
    Returns:
        Encoder function
    """
    if encoding_type == 'one_hot':
        max_len = kwargs.get('max_len', None)
        return lambda seq: one_hot_encode(seq, max_len).flatten()
    elif encoding_type == 'kmer' or encoding_type == 'k-mer':
        k = kwargs.get('k', 6)
        return lambda seq: kmer_encode(seq, k)
    elif encoding_type == 'character':
        max_len = kwargs.get('max_len', None)
        return lambda seq: character_encode(seq, max_len)
    else:
        raise ValueError(f"Unknown encoding: {encoding_type}. Use 'one_hot', 'kmer', or 'character'")

# =============================================================================
# DNA DATA LOADING
# =============================================================================

def load_dna_tsv(
    path: str,
    sequence_col: str = 'sequence',
    label_col: str = 'label',
    task_col: Optional[str] = None,
    task_filter: Optional[str] = None,
) -> Tuple[List[str], np.ndarray]:
    """
    Load DNA sequences and labels from TSV file.
    
    Args:
        path: Path to TSV file
        sequence_col: Name of sequence column (or 'seq')
        label_col: Name of label column
        task_col: Optional task column for multi-task datasets
        task_filter: If task_col specified, filter to this task
    
    Returns:
        Tuple of (sequences list, labels array)
    """
    df = pd.read_csv(path, sep='\t')
    
    # Handle different column naming conventions
    if sequence_col not in df.columns:
        if 'seq' in df.columns:
            sequence_col = 'seq'
        elif 'Sequence' in df.columns:
            sequence_col = 'Sequence'
        else:
            raise ValueError(f"Cannot find sequence column. Available: {list(df.columns)}")
    
    # Filter by task if specified
    if task_col and task_filter and task_col in df.columns:
        df = df[df[task_col] == task_filter]
    
    sequences = df[sequence_col].tolist()
    labels = df[label_col].values.astype(int)
    
    return sequences, labels

# =============================================================================
# METRICS
# =============================================================================

def _safe_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute AUROC safely, returning 0.0 if not computable."""
    if len(np.unique(y_true)) < 2:
        return 0.0
    try:
        if y_score.ndim == 2 and y_score.shape[1] == 2:
            return float(roc_auc_score(y_true, y_score[:, 1]))
        elif y_score.ndim == 2:
            return float(roc_auc_score(y_true, y_score, multi_class="ovr"))
        else:
            return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return 0.0

def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """Compute classification metrics."""
    if len(y_true) == 0:
        return {"AUROC": 0.0, "Accuracy": 0.0, "F1-Score": 0.0, "N": 0}
    
    auroc = _safe_auroc(y_true, y_prob) if y_prob is not None else 0.0
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    
    return {
        "AUROC": round(auroc, 4),
        "Accuracy": round(acc, 4),
        "F1-Score": round(f1, 4),
        "N": int(len(y_true))
    }

# =============================================================================
# DNA CLASSIFICATION RUNNER
# =============================================================================

class DNASequenceRunner:
    """
    Runner for DNA sequence classification benchmarks.
    
    Supports:
    - Loading DNA TSV data (sequence, label format)
    - Multiple encoding methods (k-mer, one-hot, character)
    - Using external model adapters or simple baseline
    - Computing AUROC, Accuracy, F1-Score
    
    Example:
        runner = DNASequenceRunner(
            model=my_dna_model,
            train_path="toy_data/genomics/dna_sequences/enhancers_cohn/train.tsv",
            test_path="toy_data/genomics/dna_sequences/enhancers_cohn/test.tsv",
            encoding="kmer",
            k=6
        )
        results = runner.run()
    """
    
    def __init__(
        self,
        model: Any = None,
        train_path: Optional[str] = None,
        test_path: Optional[str] = None,
        data_dir: Optional[str] = None,
        encoding: str = "kmer",
        sequence_col: str = "sequence",
        label_col: str = "label",
        task_col: Optional[str] = None,
        task_filter: Optional[str] = None,
        use_baseline: bool = True,
        test_size: float = 0.2,
        random_state: int = 42,
        **encoding_kwargs
    ):
        """
        Initialize DNA sequence runner.
        
        Args:
            model: Model adapter with encode()/predict_proba() methods
            train_path: Path to training TSV (or None to use data_dir)
            test_path: Path to test TSV (or None for train/test split)
            data_dir: Directory containing train.tsv and test.tsv
            encoding: Encoding method ('kmer', 'one_hot', 'character')
            sequence_col: Name of sequence column
            label_col: Name of label column
            task_col: Optional task column for multi-task data
            task_filter: Filter to specific task
            use_baseline: If True and model is None, use logistic regression
            test_size: If no test set, split ratio
            random_state: Random seed for reproducibility
            **encoding_kwargs: Additional args for encoder (e.g., k=6)
        """
        self.model = model
        self.train_path = train_path
        self.test_path = test_path
        self.data_dir = data_dir
        self.encoding = encoding
        self.sequence_col = sequence_col
        self.label_col = label_col
        self.task_col = task_col
        self.task_filter = task_filter
        self.use_baseline = use_baseline
        self.test_size = test_size
        self.random_state = random_state
        self.encoding_kwargs = encoding_kwargs
        
        # Data storage
        self.train_seqs = None
        self.train_labels = None
        self.test_seqs = None
        self.test_labels = None
        self.X_train = None
        self.X_test = None
        
    def _resolve_paths(self):
        """Resolve train/test paths from data_dir if needed."""
        if self.data_dir:
            if not self.train_path:
                self.train_path = os.path.join(self.data_dir, "train.tsv")
            if not self.test_path:
                test_candidate = os.path.join(self.data_dir, "test.tsv")
                if os.path.exists(test_candidate):
                    self.test_path = test_candidate
    
    def load_data(self):
        """Load and encode DNA sequences."""
        self._resolve_paths()
        
        # Load training data
        print(f"[DNA Runner] Loading training data from {self.train_path}")
        self.train_seqs, self.train_labels = load_dna_tsv(
            self.train_path,
            sequence_col=self.sequence_col,
            label_col=self.label_col,
            task_col=self.task_col,
            task_filter=self.task_filter
        )
        print(f"  → {len(self.train_seqs)} training sequences")
        
        # Load or split test data
        if self.test_path and os.path.exists(self.test_path):
            print(f"[DNA Runner] Loading test data from {self.test_path}")
            self.test_seqs, self.test_labels = load_dna_tsv(
                self.test_path,
                sequence_col=self.sequence_col,
                label_col=self.label_col,
                task_col=self.task_col,
                task_filter=self.task_filter
            )
            print(f"  → {len(self.test_seqs)} test sequences")
        else:
            print(f"[DNA Runner] No test set, splitting train ({1-self.test_size:.0%}/{self.test_size:.0%})")
            self.train_seqs, self.test_seqs, self.train_labels, self.test_labels = \
                train_test_split(
                    self.train_seqs, self.train_labels,
                    test_size=self.test_size,
                    random_state=self.random_state,
                    stratify=self.train_labels
                )
        
        # Encode sequences
        print(f"[DNA Runner] Encoding with method: {self.encoding}")
        encoder = get_encoder(self.encoding, **self.encoding_kwargs)
        
        self.X_train = np.array([encoder(seq) for seq in self.train_seqs])
        self.X_test = np.array([encoder(seq) for seq in self.test_seqs])
        
        print(f"  → X_train shape: {self.X_train.shape}")
        print(f"  → X_test shape: {self.X_test.shape}")
        
        # Show label distribution
        train_dist = Counter(self.train_labels)
        test_dist = Counter(self.test_labels)
        print(f"  → Train labels: {dict(train_dist)}")
        print(f"  → Test labels: {dict(test_dist)}")
        
    def run(self) -> Dict[str, Any]:
        """
        Run DNA sequence classification benchmark.
        
        Returns:
            Dictionary with metrics (AUROC, Accuracy, F1-Score)
        """
        if self.X_train is None:
            self.load_data()

        # Get predictions
        if self.model is not None:
            print("[DNA Runner] Using provided model")

            # Sequence-aware models (preferred for real DNA foundation models)
            if hasattr(self.model, "encode_sequences") and callable(getattr(self.model, "encode_sequences")):
                print("[DNA Runner] Detected sequence encoder (encode_sequences).")
                Xtr = self.model.encode_sequences(self.train_seqs)
                Xte = self.model.encode_sequences(self.test_seqs)

                clf = LogisticRegression(
                    max_iter=1000,
                    random_state=self.random_state,
                    class_weight='balanced'
                )
                clf.fit(Xtr, self.train_labels)
                y_prob = clf.predict_proba(Xte)
                y_pred = clf.predict(Xte)

            else:
                # Feature-based models (legacy)
                if hasattr(self.model, 'fit'):
                    self.model.fit(self.X_train, self.train_labels)

                if hasattr(self.model, 'predict_proba'):
                    y_prob = self.model.predict_proba(self.X_test)
                    y_pred = np.argmax(y_prob, axis=1) if y_prob.ndim == 2 else (y_prob > 0.5).astype(int)
                else:
                    y_pred = self.model.predict(self.X_test)
                    y_prob = None

        elif self.use_baseline:
            # Use logistic regression baseline
            print("[DNA Runner] Using logistic regression baseline")
            clf = LogisticRegression(
                max_iter=1000,
                random_state=self.random_state,
                class_weight='balanced'
            )
            clf.fit(self.X_train, self.train_labels)
            y_prob = clf.predict_proba(self.X_test)
            y_pred = clf.predict(self.X_test)
        else:
            raise ValueError("No model provided and use_baseline=False")
        
        # Compute metrics
        metrics = compute_classification_metrics(
            self.test_labels, y_pred, y_prob
        )
        
        print(f"[DNA Runner] Results:")
        print(f"  → AUROC: {metrics['AUROC']}")
        print(f"  → Accuracy: {metrics['Accuracy']}")
        print(f"  → F1-Score: {metrics['F1-Score']}")
        
        return metrics

# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def run_dna_benchmark(
    data_dir: str,
    model: Any = None,
    encoding: str = "kmer",
    **kwargs
) -> Dict[str, Any]:
    """
    Run a DNA classification benchmark.
    
    Args:
        data_dir: Directory with train.tsv and test.tsv
        model: Optional model (uses baseline if None)
        encoding: Encoding method ('kmer', 'one_hot', 'character')
        **kwargs: Additional arguments
    
    Returns:
        Metrics dictionary
    """
    runner = DNASequenceRunner(
        model=model,
        data_dir=data_dir,
        encoding=encoding,
        **kwargs
    )
    return runner.run()

def benchmark_all_dna_datasets(
    toy_data_root: str = "toy_data/genomics/dna_sequences",
    encoding: str = "kmer",
    model: Any = None,
    **kwargs
) -> Dict[str, Dict[str, Any]]:
    """
    Run benchmarks on all DNA datasets in toy_data.
    
    Args:
        toy_data_root: Root directory of DNA datasets
        encoding: Encoding method
        model: Optional model
    
    Returns:
        Dictionary mapping dataset name to metrics
    """
    results = {}
    
    # Find all directories with train.tsv
    import glob
    datasets = glob.glob(os.path.join(toy_data_root, "*", "train.tsv"))
    
    for train_path in datasets:
        data_dir = os.path.dirname(train_path)
        dataset_name = os.path.basename(data_dir)
        
        print(f"\n{'='*60}")
        print(f"Benchmarking: {dataset_name}")
        print(f"{'='*60}")
        
        try:
            metrics = run_dna_benchmark(
                data_dir=data_dir,
                model=model,
                encoding=encoding,
                **kwargs
            )
            results[dataset_name] = metrics
        except Exception as e:
            print(f"  ⚠️ Error: {e}")
            results[dataset_name] = {"error": str(e)}
    
    return results

# =============================================================================
# MAIN (for testing)
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run DNA sequence benchmarks")
    parser.add_argument("--data-dir", type=str, help="Directory with train.tsv/test.tsv")
    parser.add_argument("--encoding", type=str, default="kmer", 
                        choices=["kmer", "one_hot", "character"])
    parser.add_argument("--k", type=int, default=6, help="K-mer size")
    parser.add_argument("--all", action="store_true", help="Run on all toy datasets")
    
    args = parser.parse_args()
    
    if args.all:
        results = benchmark_all_dna_datasets(encoding=args.encoding, k=args.k)
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        for name, metrics in results.items():
            if "error" not in metrics:
                print(f"{name}: AUROC={metrics['AUROC']}, Acc={metrics['Accuracy']}, F1={metrics['F1-Score']}")
            else:
                print(f"{name}: ERROR - {metrics['error']}")
    elif args.data_dir:
        metrics = run_dna_benchmark(
            data_dir=args.data_dir,
            encoding=args.encoding,
            k=args.k
        )
        print(f"\nFinal: {metrics}")
    else:
        print("Usage: python dna_runner.py --data-dir <path> or --all")
