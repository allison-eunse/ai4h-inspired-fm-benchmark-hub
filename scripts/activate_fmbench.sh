#!/bin/bash
# Activate the fmbench conda environment with all required settings
#
# Usage:
#   source scripts/activate_fmbench.sh
#
# After activation, you can run:
#   python -m fmbench run --suite SUITE-GEN-CLASS-001 --model configs/model_geneformer.yaml --dataset DS-PBMC --out results/my_run

# Initialize conda
source "$HOME/miniforge3/etc/profile.d/conda.sh"

# Activate the fmbench environment
conda activate fmbench

# Fix numba caching issue
export NUMBA_CACHE_DIR=/tmp/numba_cache
mkdir -p "$NUMBA_CACHE_DIR"

# Fix matplotlib directory issue (optional)
export MPLCONFIGDIR=/tmp/matplotlib_cache
mkdir -p "$MPLCONFIGDIR"

# Add external models to Python path
export PYTHONPATH="$PWD/external/geneformer:$PYTHONPATH"

echo "✅ fmbench environment activated!"
echo ""
echo "Python: $(python --version)"
echo ""
echo "Available commands:"
echo "  python -m fmbench --help              # Show all commands"
echo "  python -m fmbench list-suites         # List available benchmark suites"
echo "  python -m fmbench run --suite ...     # Run a benchmark"
echo "  python -m fmbench build-leaderboard   # Rebuild leaderboard"
echo ""
