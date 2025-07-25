#!/bin/bash

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"

export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

cd "$PROJECT_ROOT"


python train_craft.py \
    --images_dir path/to/your/images \
    --labels_dir path/to/your/labels \
    --model_name florence2 \
    --num_gpus 2 \
    --sample_nums 100 \
    --alpha 0.01568627450980392 \
    --epsilon 0.06274509803921569 \
    --iters 300 \
    --output_base_dir path/to/your/output/directory

echo "All attack scenes completed at $(date)" | tee -a $master_log