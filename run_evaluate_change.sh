#!/bin/bash
IMAGE_DIR="path/to/images/directory"
LABELS_DIR="path/to/labels/directory"


dir="path/to/output/directory"

echo "Processing directory: $dir"
# 运行评估脚本
python evaluate/evaluate_f2.py \
    --output_dir "$dir" \
    --IMAGE_DIR "$IMAGE_DIR" \
    --LABELS_DIR "$LABELS_DIR" \

echo "Completed processing $dir"
echo "----------------------------------------"

echo "All evaluations completed!"