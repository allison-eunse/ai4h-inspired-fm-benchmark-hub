# DNA Classification Report: DNA-promoters_nontata-kmer_baseline-20251219-023228

## Overview

- **Dataset**: promoters_nontata
- **Model**: kmer_baseline
- **Encoding**: kmer (k=6)
- **Date**: 2025-12-19T02:32:28

## Performance Metrics

| Metric | Score |
|--------|-------|
| **AUROC** | 0.8426 |
| **Accuracy** | 0.7730 |
| **F1-Score** | 0.7728 |

## Dataset Info

- **Training samples**: 27097
- **Test samples**: 9034

## Method

DNA sequences were encoded using **kmer** encoding with k=6.
A logistic regression classifier was trained on the encoded features.

## References

- Genomic Benchmarks: https://huggingface.co/datasets/katielink/genomic-benchmarks
- Nucleotide Transformer: https://huggingface.co/InstaDeepAI
