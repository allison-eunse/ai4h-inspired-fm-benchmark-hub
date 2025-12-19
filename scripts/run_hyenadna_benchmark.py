#!/usr/bin/env python3
"""
Run HyenaDNA on DNA benchmarks with real embeddings.
Uses the character-level tokenizer which doesn't require triton.
"""

import os
import sys
import yaml
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.model_selection import train_test_split

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def load_dataset(data_dir: Path, max_samples: int = None):
    """Load DNA benchmark dataset with stratified sampling."""
    train_file = data_dir / "train.tsv"
    test_file = data_dir / "test.tsv"
    
    train_df = pd.read_csv(train_file, sep='\t')
    test_df = pd.read_csv(test_file, sep='\t')
    
    seq_col = 'sequence' if 'sequence' in train_df.columns else 'seq'
    label_col = 'label'
    
    if max_samples and len(train_df) > max_samples:
        train_df, _ = train_test_split(
            train_df, train_size=max_samples, 
            stratify=train_df[label_col], random_state=42
        )
        test_df, _ = train_test_split(
            test_df, train_size=min(max_samples // 4, len(test_df)),
            stratify=test_df[label_col], random_state=42
        )
    
    return {
        'train_seqs': train_df[seq_col].tolist(),
        'train_labels': train_df[label_col].values,
        'test_seqs': test_df[seq_col].tolist(),
        'test_labels': test_df[label_col].values,
    }


class HyenaDNAModel:
    """HyenaDNA model wrapper - uses character-level DNA encoding."""
    
    def __init__(self):
        self.name = "hyenadna_real"
        self.model_name = "LongSafari/hyenadna-tiny-1k-seqlen-hf"  # Tiny model for CPU
        
    def load(self):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
        print(f"[HyenaDNA] Loading {self.model_name}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        self.model.eval()
        print("[HyenaDNA] ✅ Model loaded!")
        
    def encode(self, sequences: list, batch_size: int = 32) -> np.ndarray:
        """Encode sequences using HyenaDNA embeddings."""
        embeddings = []
        max_len = 512  # Tiny model supports up to 1k
        
        with torch.no_grad():
            for i in range(0, len(sequences), batch_size):
                if i % 100 == 0:
                    print(f"  Encoding {i}/{len(sequences)}...")
                    
                batch = sequences[i:i+batch_size]
                batch = [s[:max_len] for s in batch]
                
                inputs = self.tokenizer(
                    batch, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True,
                    max_length=max_len
                )
                
                outputs = self.model(**inputs)
                # Mean pooling over sequence dimension
                if hasattr(outputs, 'last_hidden_state'):
                    emb = outputs.last_hidden_state.mean(dim=1).numpy()
                else:
                    emb = outputs[0].mean(dim=1).numpy()
                embeddings.append(emb)
        
        return np.vstack(embeddings)


def run_benchmark(model, dataset_name: str, data: dict, output_dir: Path):
    """Run benchmark and save results."""
    print(f"\n{'='*60}")
    print(f"Running {model.name} on {dataset_name}")
    print(f"Train: {len(data['train_seqs'])}, Test: {len(data['test_seqs'])}")
    print(f"{'='*60}")
    
    start_time = datetime.now()
    
    # Encode
    print("Encoding train sequences...")
    X_train = model.encode(data['train_seqs'])
    print(f"Train embeddings: {X_train.shape}")
    
    print("Encoding test sequences...")
    X_test = model.encode(data['test_seqs'])
    print(f"Test embeddings: {X_test.shape}")
    
    y_train = data['train_labels']
    y_test = data['test_labels']
    
    # Train classifier
    print("Training classifier...")
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred_proba = clf.predict_proba(X_test)
    y_pred = y_pred_proba.argmax(axis=1)
    
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
    
    print(f"\n🎯 Results:")
    for metric, value in results.items():
        print(f"  {metric}: {value}")
    print(f"  Runtime: {runtime:.1f}s")
    
    eval_yaml = {
        'eval_id': f"DNA-{dataset_name.upper()}-{model.name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        'benchmark_id': f'BM-DNA-{dataset_name.upper()}',
        'model_ids': {'candidate': 'HyenaDNA'},
        'dataset_id': f'DS-DNA-{dataset_name.upper()}',
        'run_metadata': {
            'date': datetime.now().isoformat(),
            'runner': 'fmbench',
            'suite_id': 'SUITE-DNA-CLASS',
            'hardware': 'CPU',
            'runtime_seconds': round(runtime, 1),
            'n_train': len(data['train_seqs']),
            'n_test': len(data['test_seqs']),
            'encoding': 'hyenadna_mamba_embeddings',
            'embedding_dim': X_train.shape[1],
        },
        'metrics': results,
        'status': 'Completed',
    }
    
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_file = output_dir / f"eval_{model.name}_{dataset_name}.yaml"
    with open(eval_file, 'w') as f:
        yaml.dump(eval_yaml, f, default_flow_style=False, sort_keys=False)
    print(f"Saved: {eval_file}")
    
    return eval_yaml


def main():
    print("=" * 70)
    print("HYENADNA BENCHMARK")
    print("Using REAL Mamba-based embeddings")
    print("=" * 70)
    
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "toy_data" / "genomics" / "dna_sequences"
    output_dir = base_dir / "results" / "dna_benchmarks"
    evals_dir = base_dir / "evals"
    
    datasets = {
        'enhancer': data_dir / "enhancers_cohn",
        'promoter': data_dir / "promoters_nontata",
    }
    
    MAX_SAMPLES = 500
    
    model = HyenaDNAModel()
    model.load()
    
    all_results = []
    
    for ds_name, ds_path in datasets.items():
        try:
            data = load_dataset(ds_path, max_samples=MAX_SAMPLES)
            result = run_benchmark(model, ds_name, data, output_dir)
            all_results.append(result)
            
            evals_dir.mkdir(parents=True, exist_ok=True)
            eval_file = evals_dir / f"eval_dna_hyenadna_real_{ds_name}.yaml"
            with open(eval_file, 'w') as f:
                yaml.dump(result, f, default_flow_style=False, sort_keys=False)
                
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for r in all_results:
        print(f"{r['dataset_id']}: AUROC={r['metrics']['AUROC']}, Acc={r['metrics']['Accuracy']}")
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
