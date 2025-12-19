---
name: 📊 Benchmark Submission
about: Submit your model's benchmark results for review
title: "[SUBMISSION] Model Name - Benchmark Name"
labels: submission, pending-review
assignees: allison-eunse
---

## Model Information

**Model Name**: 
**Model Type**: (e.g., fMRI Foundation Model, Genomics FM, etc.)
**Paper/Preprint**: (link if available)
**Code Repository**: (link if public)
**Model ID**: (a short stable ID, e.g. `my_model_v1`)

## Benchmark Results

**Benchmark ID**: (e.g., BM-001, BM-FMRI-GRANULAR, robustness_testing)
**Dataset Used**: 

### Overall Metrics

| Metric | Value |
|--------|-------|
| AUROC | |
| Accuracy | |
| F1-Score | |
| (other) | |

### Stratified Results (if applicable)

<details>
<summary>By Scanner</summary>

| Scanner | AUROC | Accuracy | N |
|---------|-------|----------|---|
| Siemens | | | |
| GE | | | |
| Philips | | | |

</details>

<details>
<summary>By Site/Dataset</summary>

| Site | AUROC | N |
|------|-------|---|
| | | |

</details>

## Evaluation Details

**Hardware Used**: 
**Runtime**: 
**Reproducibility Seed**: 
**fmbench Version**: 

## eval.yaml (required for automatic processing)

Paste your `eval.yaml` content here. This will be automatically processed and added to the leaderboard:

```yaml
# Paste your eval.yaml content here
eval_id: 
benchmark_id: 
model_ids:
  candidate: 
dataset_id: 
run_metadata:
  date: 
  runner: fmbench
  suite_id: 
  hardware: 
  runtime_seconds: 
metrics:
  AUROC: 
  Accuracy: 
  F1-Score: 
status: Completed
```

## Attachments (optional)

- [ ] (Optional) I also attached `report.md` for human review

## Checklist

- [ ] I ran the official fmbench evaluation suite
- [ ] Results are reproducible with the provided seed
- [ ] I have included all required metrics
- [ ] Model is publicly available or described in a paper

## Additional Notes

(Any other relevant information about the evaluation)

