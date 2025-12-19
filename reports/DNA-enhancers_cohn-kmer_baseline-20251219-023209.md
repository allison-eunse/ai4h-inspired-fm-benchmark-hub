# DNA Classification Report: DNA-enhancers_cohn-kmer_baseline-20251219-023209

## Overview

- **Dataset**: enhancers_cohn
- **Model**: kmer_baseline
- **Encoding**: kmer (k=6)
- **Date**: 2025-12-19T02:32:09

## Performance Metrics

| Metric | Score |
|--------|-------|
| **AUROC** | 0.7433 |
| **Accuracy** | 0.6740 |
| **F1-Score** | 0.6738 |

## Dataset Info

- **Training samples**: 20843
- **Test samples**: 6948

## Method

DNA sequences were encoded using **kmer** encoding with k=6.
A logistic regression classifier was trained on the encoded features.

## References

- Genomic Benchmarks: https://huggingface.co/datasets/katielink/genomic-benchmarks
- Nucleotide Transformer: https://huggingface.co/InstaDeepAI
