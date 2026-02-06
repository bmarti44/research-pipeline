#!/bin/bash
# Run baseline_1000_continued control experiment
# This loads baseline_1000 checkpoint and trains for 1000 more steps
# WITHOUT COCONUT, to isolate "continued training" from "latent mechanism"

set -e

BASE_DIR="/Users/briamart/github/tool-calling/papers/efficient_architecture_proof"
CODE_DIR="$BASE_DIR/code"
DATA_DIR="$CODE_DIR/data"
RESULTS_DIR="$BASE_DIR/results/v4.0_scaleup"
OUTPUT_DIR="$RESULTS_DIR/baseline_1000_continued"

# Activate Python virtual environment
source /Users/briamart/github/tool-calling/.venv/bin/activate

SEEDS="42 123 456 789 1001"

echo "============================================"
echo "Running baseline_1000_continued control (n=5)"
echo "============================================"

cd "$CODE_DIR"

for SEED in $SEEDS; do
    CHECKPOINT="$RESULTS_DIR/baseline_1000/seed_$SEED/baseline_model.pt"

    if [ ! -f "$CHECKPOINT" ]; then
        echo "ERROR: Checkpoint not found: $CHECKPOINT"
        continue
    fi

    echo ""
    echo "=== Seed $SEED ==="
    echo "Loading checkpoint: $CHECKPOINT"

    python3 training/train_abc.py \
        --condition baseline \
        --train_data "$DATA_DIR/multistep_train.json" \
        --val_data "$DATA_DIR/multistep_val.json" \
        --output "$OUTPUT_DIR/seed_$SEED" \
        --size medium \
        --max_steps 1000 \
        --seed $SEED \
        --tokenizer bpe \
        --checkpoint "$CHECKPOINT" \
        --warmstart_from baseline

    echo "Seed $SEED complete"
done

echo ""
echo "============================================"
echo "All seeds complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "============================================"
