# External Research Repositories

This directory centralizes the upstream model code referenced throughout the KB. Two repos (BAGEL and MoT) are vendored snapshots so we can cite concrete file paths in walkthroughs. Every other entry starts life as a lightweight placeholder; run the fetch script whenever you need the full upstream sources.

## Tracked snapshots

| Directory | Upstream project |
| --- | --- |
| `bagel/` | https://github.com/bytedance-seed/BAGEL |
| `MoT/` | https://github.com/facebookresearch/Mixture-of-Transformers |
| `flamingo/` | https://github.com/mlfoundations/open_flamingo |
| `hyena/` | https://github.com/togethercomputer/stripedhyena |

These directories are versioned with the KB for reproducibility. Pull upstream changes manually if you want a fresher snapshot.

## Fetch-on-demand placeholders

```bash
./scripts/fetch_external_repos.sh
```

The script replaces each placeholder with the upstream repo inside `external_repos/<name>`. Remove the directory first if you want to reclone from scratch.

| Directory | Upstream project |
| --- | --- |
| `brainlm` | https://github.com/vandijklab/BrainLM |
| `brainmt` | https://github.com/arunkumar-kannan/brainmt-fmri |
| `brainjepa` | https://github.com/janklees/brainjepa |
| `brainharmony` | https://github.com/hzlab/Brain-Harmony |
| `caduceus` | https://github.com/kuleshov-group/caduceus |
| `dnabert2` | https://github.com/Zhihan1996/DNABERT2 |
| `evo2` | https://github.com/ArcInstitute/evo2 |
| `generator` | https://github.com/GenerTeam/GENERator |
| `swift` | https://github.com/Transconnectome/SwiFT |
| `titan` | https://github.com/mahmoodlab/TITAN |
| `me-lamma` | https://github.com/BIDS-Xu-Lab/Me-LLaMA |
| `M3FM` | https://github.com/ai-in-health/M3FM |
| `fms-medical` | https://github.com/YutingHe-list/Awesome-Foundation-Models-for-Advancing-Healthcare |

Feel free to edit `scripts/fetch_external_repos.sh` if you need to add another upstream.
