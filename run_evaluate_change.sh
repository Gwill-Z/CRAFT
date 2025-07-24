#!/bin/bash

# 设置基础输出目录
BASE_OUTPUT_DIR="./outputs/craft_ablation-eps-iters"

IMAGE_DIR="datasets/transforming-100-cat/images"
LABELS_DIR="datasets/transforming-100-cat/labels"


dir="outputs/craft/craft_florence2_mask_True_neg_1_0723_212453"

echo "Processing directory: $dir"
# 运行评估脚本
python evaluate/evaluate_f2.py \
    --output_dir "$dir" \
    --IMAGE_DIR "$IMAGE_DIR" \
    --LABELS_DIR "$LABELS_DIR" \

echo "Completed processing $dir"
echo "----------------------------------------"

echo "All evaluations completed!"