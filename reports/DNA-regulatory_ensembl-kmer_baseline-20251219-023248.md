# DNA Classification Report: DNA-regulatory_ensembl-kmer_baseline-20251219-023248

## Overview

- **Dataset**: regulatory_ensembl
- **Model**: kmer_baseline
- **Encoding**: kmer (k=6)
- **Date**: 2025-12-19T02:32:48

## Performance Metrics

| Metric | Score |
|--------|-------|
| **AUROC** | 0.6956 |
| **Accuracy** | 0.5044 |
| **F1-Score** | 0.5007 |

## Dataset Info

- **Training samples**: 1500
- **Test samples**: 57713

## Method

DNA sequences were encoded using **kmer** encoding with k=6.
A logistic regression classifier was trained on the encoded features.

## References

- Genomic Benchmarks: https://huggingface.co/datasets/katielink/genomic-benchmarks
- Nucleotide Transformer: https://huggingface.co/InstaDeepAI
