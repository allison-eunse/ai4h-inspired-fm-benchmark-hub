# External Research Repositories

This directory contains **vendored upstream code** for **neuro** (brain imaging / brain signals) and **genomics** (DNA + scRNA-seq) foundation models.

We intentionally **do not vendor non-neuro/non-genomics foundation models** here (e.g., general vision-language models, pathology-only models, report-generation LLMs).

## Included upstream projects (neuro + genomics only)

| Directory | Upstream project |
| --- | --- |
| `brainlm/` | https://github.com/vandijklab/BrainLM |
| `brainmt/` | https://github.com/arunkumar-kannan/brainmt-fmri |
| `brainjepa/` | https://github.com/Eric-LRL/Brain-JEPA |
| `brainharmony/` | https://github.com/hzlab/Brain-Harmony |
| `swift/` | https://github.com/Transconnectome/SwiFT |
| `neuroclips/` | https://github.com/gongzix/NeuroClips |
| `geneformer/` | https://huggingface.co/ctheodoris/Geneformer |
| `caduceus/` | https://github.com/kuleshov-group/caduceus |
| `dnabert2/` | https://github.com/Zhihan1996/DNABERT2 |
| `evo2/` | https://github.com/ArcInstitute/evo2 |
| `hyena/` | https://github.com/togethercomputer/stripedhyena |

## Weights

Model weights are **not stored in this repository**. Use `python -m fmbench download-weights ...` to fetch weights into the local cache (outside git).
