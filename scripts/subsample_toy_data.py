#!/usr/bin/env python3
"""
Subsample large genomics datasets to create smaller toy versions.

This script creates stratified subsamples of the training data, preserving:
- Real DNA sequences (not synthetic)
- Class distribution (proportional sampling)
- Reproducibility (fixed random seed)

The resulting files are suitable for quick pipeline testing while maintaining
data validity for model benchmarking.
"""

import os
import random
from collections import defaultdict
from pathlib import Path

# Fixed seed for reproducibility
RANDOM_SEED = 42

# Target samples per class (adjust based on desired file size)
# Using 500 samples per class gives ~1500-2000 total rows, keeping files under 5MB
SAMPLES_PER_CLASS = 500


def subsample_tsv(input_path: str, output_path: str, 
                  label_col: int, samples_per_class: int = SAMPLES_PER_CLASS,
                  seed: int = RANDOM_SEED) -> dict:
    """
    Create a stratified subsample of a TSV file.
    
    Args:
        input_path: Path to the original TSV file
        output_path: Path for the subsampled output
        label_col: 0-indexed column number containing the label
        samples_per_class: Number of samples to keep per class
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary with sampling statistics
    """
    random.seed(seed)
    
    # Read and group by label
    rows_by_label = defaultdict(list)
    header = None
    
    print(f"Reading {input_path}...")
    with open(input_path, 'r') as f:
        for i, line in enumerate(f):
            if i == 0:
                header = line
                continue
            parts = line.strip().split('\t')
            if len(parts) > label_col:
                label = parts[label_col]
                rows_by_label[label].append(line)
    
    # Report original distribution
    print(f"  Original distribution:")
    for label, rows in sorted(rows_by_label.items()):
        print(f"    Label {label}: {len(rows)} samples")
    
    # Sample from each class
    sampled_rows = []
    stats = {'original': {}, 'sampled': {}}
    
    for label, rows in sorted(rows_by_label.items()):
        stats['original'][label] = len(rows)
        n_samples = min(len(rows), samples_per_class)
        sampled = random.sample(rows, n_samples)
        sampled_rows.extend(sampled)
        stats['sampled'][label] = n_samples
        print(f"    Sampled {n_samples} from label {label}")
    
    # Shuffle to avoid ordering by class
    random.shuffle(sampled_rows)
    
    # Write output
    print(f"  Writing {len(sampled_rows)} samples to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(header)
        for row in sampled_rows:
            f.write(row)
    
    # Report file size
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Output file size: {size_mb:.2f} MB")
    stats['file_size_mb'] = size_mb
    stats['total_samples'] = len(sampled_rows)
    
    return stats


def main():
    # Get project root (parent of scripts/)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    toy_data_dir = project_root / "toy_data" / "genomics" / "dna_sequences"
    
    print("=" * 60)
    print("Creating stratified subsamples of genomics toy data")
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Target samples per class: {SAMPLES_PER_CLASS}")
    print("=" * 60)
    
    # Define files to subsample
    files_to_process = [
        {
            "name": "nucleotide_transformer",
            "input": toy_data_dir / "nucleotide_transformer" / "train.tsv",
            "label_col": 2,  # 0-indexed: sequence, name, label, task
        },
        {
            "name": "regulatory_ensembl", 
            "input": toy_data_dir / "regulatory_ensembl" / "train.tsv",
            "label_col": 1,  # 0-indexed: seq, label
        },
        {
            "name": "open_chromatin",
            "input": toy_data_dir / "open_chromatin" / "train.tsv",
            "label_col": 1,  # 0-indexed: seq, label
        },
    ]
    
    all_stats = {}
    
    for file_info in files_to_process:
        input_path = file_info["input"]
        
        if not input_path.exists():
            print(f"\nSkipping {file_info['name']}: file not found at {input_path}")
            continue
        
        print(f"\n{'=' * 60}")
        print(f"Processing: {file_info['name']}")
        print(f"{'=' * 60}")
        
        # Backup original to _full.tsv if it doesn't exist
        full_backup = input_path.parent / "train_full.tsv"
        if not full_backup.exists():
            print(f"  Backing up original to {full_backup.name}...")
            os.rename(input_path, full_backup)
            input_for_sampling = full_backup
        else:
            # Already backed up, use the backup as source
            input_for_sampling = full_backup
            print(f"  Using existing backup: {full_backup.name}")
        
        # Create subsampled version as train.tsv
        output_path = input_path  # Replace original train.tsv
        
        stats = subsample_tsv(
            str(input_for_sampling),
            str(output_path),
            file_info["label_col"],
        )
        all_stats[file_info["name"]] = stats
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, stats in all_stats.items():
        print(f"\n{name}:")
        print(f"  Original total: {sum(stats['original'].values())} samples")
        print(f"  Subsampled total: {stats['total_samples']} samples")
        print(f"  File size: {stats['file_size_mb']:.2f} MB")
    
    print("\n✓ Subsampling complete!")
    print("  Original files backed up as train_full.tsv")
    print("  New train.tsv files contain reproducible subsamples")
    print(f"  Random seed used: {RANDOM_SEED}")


if __name__ == "__main__":
    main()

