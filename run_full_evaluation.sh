#!/bin/bash
# Complete evaluation workflow - runs all evaluation methods

set -e

cd "$(dirname "$0")"
source venv/bin/activate

echo "=" | head -c 70 && echo
echo "Complete FiQA Evaluation Workflow"
echo "=" | head -c 70 && echo
echo ""

# Check if we want to compare with base model
COMPARE_BASE=${1:-"no"}
MAX_SAMPLES=${2:-"648"}

echo "Configuration:"
echo "  Compare with base: $COMPARE_BASE"
echo "  Max samples: $MAX_SAMPLES"
echo ""

# Method 1: Direct Evaluation (Recommended)
echo "=" | head -c 70 && echo
echo "Method 1: Direct Evaluation (Recommended)"
echo "=" | head -c 70 && echo
echo ""

if [ "$COMPARE_BASE" = "compare" ]; then
    echo "Running direct evaluation with base comparison..."
    python evaluate_fiqa.py --compare-base --max-samples $MAX_SAMPLES --output ./results/direct_eval_results.json
else
    echo "Running direct evaluation (PEFT only)..."
    python evaluate_fiqa.py --max-samples $MAX_SAMPLES --output ./results/direct_eval_results.json
fi

echo ""
echo "✅ Direct evaluation complete!"
echo "   Results: ./results/direct_eval_results.json"
echo ""

echo ""
echo "=" | head -c 70 && echo
echo "✅ Evaluation workflow complete!"
echo "=" | head -c 70 && echo
echo ""
echo "Results summary:"
echo "  - Direct evaluation: ./results/direct_eval_results.json"
echo ""

