#!/usr/bin/env python3
"""
Run DNA Foundation Model Benchmarks

This script runs DNA FMs on the real benchmark datasets:
- Enhancers (Cohn et al.) - ChIP-seq derived
- Promoters (EPD) - Real curated promoters
- Regulatory (Ensembl) - ENCODE + Roadmap data
"""

import os
import sys
import json
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from collections import Counter

# Add fmbench to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# ============================================================================
# DNA Sequence Encoding
# ============================================================================

def one_hot_encode_dna(sequence: str, max_len: int = 500) -> np.ndarray:
    """One-hot encode DNA sequence (A, C, G, T)."""
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
    seq = sequence.upper()[:max_len]
    
    # Pad or truncate
    if len(seq) < max_len:
        seq = seq + 'N' * (max_len - len(seq))
    
    encoded = np.zeros((max_len, 4), dtype=np.float32)
    for i, base in enumerate(seq):
        if base in mapping and mapping[base] < 4:
            encoded[i, mapping[base]] = 1.0
    
    return encoded.flatten()


def kmer_encode_dna(sequence: str, k: int = 6) -> np.ndarray:
    """K-mer frequency encoding for DNA sequences."""
    from itertools import product
    
    # All possible k-mers
    bases = 'ACGT'
    all_kmers = [''.join(p) for p in product(bases, repeat=k)]
    kmer_to_idx = {kmer: i for i, kmer in enumerate(all_kmers)}
    
    seq = sequence.upper()
    counts = np.zeros(len(all_kmers), dtype=np.float32)
    
    for i in range(len(seq) - k + 1):
        kmer = seq[i:i+k]
        if kmer in kmer_to_idx:
            counts[kmer_to_idx[kmer]] += 1
    
    # Normalize
    if counts.sum() > 0:
        counts = counts / counts.sum()
    
    return counts


# ============================================================================
# Model Wrappers
# ============================================================================

class DNAModelWrapper:
    """Base class for DNA model wrappers with fallback encoding."""
    
    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.use_fallback = False
        
    def encode(self, sequences: list) -> np.ndarray:
        """Encode sequences. Override in subclasses."""
        # Default: k-mer encoding
        return np.array([kmer_encode_dna(seq, k=6) for seq in sequences])
    
    def fit_classifier(self, X: np.ndarray, y: np.ndarray):
        """Train a classifier on embeddings."""
        self.classifier = LogisticRegression(max_iter=1000)
        self.classifier.fit(X, y)
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get probability predictions."""
        return self.classifier.predict_proba(X)


class HyenaDNAWrapper(DNAModelWrapper):
    """HyenaDNA wrapper with HuggingFace fallback."""
    
    def __init__(self):
        super().__init__("hyenadna")
        self.tokenizer = None
        self.model_hf = None
        
    def load(self):
        try:
            from transformers import AutoTokenizer, AutoModel
            print("[HyenaDNA] Loading from HuggingFace...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                "LongSafari/hyenadna-small-32k-seqlen",
                trust_remote_code=True
            )
            self.model_hf = AutoModel.from_pretrained(
                "LongSafari/hyenadna-small-32k-seqlen",
                trust_remote_code=True
            )
            self.model_hf.eval()
            print("[HyenaDNA] ✅ Loaded successfully")
            return True
        except Exception as e:
            print(f"[HyenaDNA] ⚠️ Using fallback encoding: {e}")
            self.use_fallback = True
            return False
    
    def encode(self, sequences: list, batch_size: int = 32) -> np.ndarray:
        if self.use_fallback or self.model_hf is None:
            return super().encode(sequences)
        
        import torch
        embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(sequences), batch_size):
                batch = sequences[i:i+batch_size]
                inputs = self.tokenizer(
                    batch, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True,
                    max_length=512
                )
                outputs = self.model_hf(**inputs)
                # Mean pooling
                emb = outputs.last_hidden_state.mean(dim=1).numpy()
                embeddings.append(emb)
        
        return np.vstack(embeddings)


class DNABERT2Wrapper(DNAModelWrapper):
    """DNABERT-2 wrapper."""
    
    def __init__(self):
        super().__init__("dnabert2")
        
    def load(self):
        try:
            from transformers import AutoTokenizer, AutoModel
            print("[DNABERT-2] Loading from HuggingFace...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                "zhihan1996/DNABERT-2-117M",
                trust_remote_code=True
            )
            self.model_hf = AutoModel.from_pretrained(
                "zhihan1996/DNABERT-2-117M",
                trust_remote_code=True
            )
            self.model_hf.eval()
            print("[DNABERT-2] ✅ Loaded successfully")
            return True
        except Exception as e:
            print(f"[DNABERT-2] ⚠️ Using fallback encoding: {e}")
            self.use_fallback = True
            return False
    
    def encode(self, sequences: list, batch_size: int = 32) -> np.ndarray:
        if self.use_fallback or not hasattr(self, 'model_hf'):
            return super().encode(sequences)
        
        import torch
        embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(sequences), batch_size):
                batch = sequences[i:i+batch_size]
                inputs = self.tokenizer(
                    batch, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True,
                    max_length=512
                )
                outputs = self.model_hf(**inputs)
                emb = outputs.last_hidden_state.mean(dim=1).numpy()
                embeddings.append(emb)
        
        return np.vstack(embeddings)


class NucleotideTransformerWrapper(DNAModelWrapper):
    """Nucleotide Transformer wrapper."""
    
    def __init__(self):
        super().__init__("nucleotide_transformer")
        
    def load(self):
        try:
            from transformers import AutoTokenizer, AutoModel
            print("[NT] Loading Nucleotide Transformer from HuggingFace...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                "InstaDeepAI/nucleotide-transformer-500m-human-ref",
                trust_remote_code=True
            )
            self.model_hf = AutoModel.from_pretrained(
                "InstaDeepAI/nucleotide-transformer-500m-human-ref",
                trust_remote_code=True
            )
            self.model_hf.eval()
            print("[NT] ✅ Loaded successfully")
            return True
        except Exception as e:
            print(f"[NT] ⚠️ Using fallback encoding: {e}")
            self.use_fallback = True
            return False
    
    def encode(self, sequences: list, batch_size: int = 16) -> np.ndarray:
        if self.use_fallback or not hasattr(self, 'model_hf'):
            return super().encode(sequences)
        
        import torch
        embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(sequences), batch_size):
                batch = sequences[i:i+batch_size]
                inputs = self.tokenizer(
                    batch, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True,
                    max_length=1000
                )
                outputs = self.model_hf(**inputs)
                emb = outputs.last_hidden_state.mean(dim=1).numpy()
                embeddings.append(emb)
        
        return np.vstack(embeddings)


class KmerBaselineWrapper(DNAModelWrapper):
    """Simple k-mer baseline for comparison."""
    
    def __init__(self, k: int = 6):
        super().__init__(f"kmer_k{k}")
        self.k = k
        
    def load(self):
        print(f"[K-mer Baseline] k={self.k} (no model to load)")
        return True
    
    def encode(self, sequences: list) -> np.ndarray:
        return np.array([kmer_encode_dna(seq, k=self.k) for seq in sequences])


# ============================================================================
# Benchmark Runner
# ============================================================================

def load_dataset(data_dir: Path, max_samples: int = None):
    """Load DNA benchmark dataset."""
    train_file = data_dir / "train.tsv"
    test_file = data_dir / "test.tsv"
    
    train_df = pd.read_csv(train_file, sep='\t')
    test_df = pd.read_csv(test_file, sep='\t')
    
    # Handle different column names
    seq_col = 'sequence' if 'sequence' in train_df.columns else 'seq'
    label_col = 'label'
    
    if max_samples and len(train_df) > max_samples:
        # Stratified sampling to keep class balance
        from sklearn.model_selection import train_test_split
        train_df, _ = train_test_split(
            train_df, 
            train_size=max_samples, 
            stratify=train_df[label_col],
            random_state=42
        )
        test_df, _ = train_test_split(
            test_df, 
            train_size=min(max_samples // 4, len(test_df)),
            stratify=test_df[label_col],
            random_state=42
        )
    
    return {
        'train_seqs': train_df[seq_col].tolist(),
        'train_labels': train_df[label_col].values,
        'test_seqs': test_df[seq_col].tolist(),
        'test_labels': test_df[label_col].values,
    }


def run_benchmark(model, dataset_name: str, data: dict, output_dir: Path):
    """Run benchmark on a single dataset."""
    print(f"\n{'='*60}")
    print(f"Running {model.name} on {dataset_name}")
    print(f"{'='*60}")
    
    start_time = datetime.now()
    
    # Encode sequences
    print(f"Encoding {len(data['train_seqs'])} train sequences...")
    X_train = model.encode(data['train_seqs'])
    print(f"Encoding {len(data['test_seqs'])} test sequences...")
    X_test = model.encode(data['test_seqs'])
    
    y_train = data['train_labels']
    y_test = data['test_labels']
    
    print(f"Embedding shape: {X_train.shape}")
    print(f"Label distribution: {Counter(y_train)}")
    
    # Train classifier
    print("Training classifier...")
    model.fit_classifier(X_train, y_train)
    
    # Evaluate
    y_pred_proba = model.predict_proba(X_test)
    y_pred = y_pred_proba.argmax(axis=1)
    
    # Metrics
    if len(np.unique(y_test)) == 2:
        auroc = roc_auc_score(y_test, y_pred_proba[:, 1])
    else:
        auroc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    runtime = (datetime.now() - start_time).total_seconds()
    
    results = {
        'AUROC': round(auroc, 4),
        'Accuracy': round(accuracy, 4),
        'F1-Score': round(f1, 4),
    }
    
    print(f"\nResults:")
    for metric, value in results.items():
        print(f"  {metric}: {value}")
    print(f"  Runtime: {runtime:.1f}s")
    
    # Generate eval.yaml
    eval_id = f"DNA-{dataset_name.upper()}-{model.name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    eval_yaml = {
        'eval_id': eval_id,
        'benchmark_id': f'BM-DNA-{dataset_name.upper()}',
        'model_ids': {
            'candidate': model.name,
        },
        'dataset_id': f'DS-DNA-{dataset_name.upper()}',
        'run_metadata': {
            'date': datetime.now().isoformat(),
            'runner': 'fmbench',
            'suite_id': 'SUITE-DNA-CLASS',
            'hardware': 'CPU',
            'runtime_seconds': round(runtime, 1),
            'n_train': len(data['train_seqs']),
            'n_test': len(data['test_seqs']),
            'encoding': 'k-mer' if model.use_fallback else 'model_embedding',
        },
        'metrics': results,
        'status': 'Completed',
    }
    
    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_file = output_dir / f"eval_{model.name}_{dataset_name}.yaml"
    with open(eval_file, 'w') as f:
        yaml.dump(eval_yaml, f, default_flow_style=False, sort_keys=False)
    
    print(f"Saved: {eval_file}")
    
    return eval_yaml


def main():
    """Run all DNA benchmarks."""
    print("=" * 70)
    print("DNA FOUNDATION MODEL BENCHMARK RUNNER")
    print("=" * 70)
    print()
    
    # Paths
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "toy_data" / "genomics" / "dna_sequences"
    output_dir = base_dir / "results" / "dna_benchmarks"
    evals_dir = base_dir / "evals"
    
    # Datasets to benchmark
    datasets = {
        'enhancers': data_dir / "enhancers_cohn",
        'promoters': data_dir / "promoters_nontata",
    }
    
    # Models to run
    models = [
        KmerBaselineWrapper(k=6),
        HyenaDNAWrapper(),
        DNABERT2Wrapper(),
        NucleotideTransformerWrapper(),
    ]
    
    # Limit samples for faster testing (remove for full benchmark)
    MAX_SAMPLES = 5000  # Use None for full dataset
    
    all_results = []
    
    for model in models:
        print(f"\n{'#'*70}")
        print(f"# MODEL: {model.name}")
        print(f"{'#'*70}")
        
        model.load()
        
        for ds_name, ds_path in datasets.items():
            try:
                data = load_dataset(ds_path, max_samples=MAX_SAMPLES)
                result = run_benchmark(model, ds_name, data, output_dir)
                all_results.append(result)
                
                # Also save to evals/ for leaderboard
                evals_dir.mkdir(parents=True, exist_ok=True)
                eval_file = evals_dir / f"eval_dna_{model.name}_{ds_name}.yaml"
                with open(eval_file, 'w') as f:
                    yaml.dump(result, f, default_flow_style=False, sort_keys=False)
                    
            except Exception as e:
                print(f"ERROR running {model.name} on {ds_name}: {e}")
                import traceback
                traceback.print_exc()
    
    # Summary
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    
    summary_df = pd.DataFrame([
        {
            'Model': r['model_ids']['candidate'],
            'Dataset': r['dataset_id'].split('-')[-1],
            'AUROC': r['metrics']['AUROC'],
            'Accuracy': r['metrics']['Accuracy'],
            'F1-Score': r['metrics']['F1-Score'],
        }
        for r in all_results
    ])
    
    print(summary_df.to_string(index=False))
    print()
    
    # Save summary
    summary_file = output_dir / "summary.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"Saved summary: {summary_file}")
    
    print("\n✅ Done! Run 'python -m fmbench build-leaderboard' to update leaderboard.")


if __name__ == "__main__":
    main()
