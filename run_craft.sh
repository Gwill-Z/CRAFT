#!/bin/bash

# 获取项目根目录的绝对路径
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"

# 添加项目根目录到PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# 切换到项目根目录
cd "$PROJECT_ROOT"


python train_craft.py \
    --images_dir datasets/transforming-100-cat/images \
    --labels_dir datasets/transforming-100-cat/labels \
    --model_name florence2 \
    --num_gpus 2 \
    --sample_nums 100 \
    --alpha 0.01568627450980392 \
    --epsilon 0.06274509803921569 \
    --iters 300 \
    --output_base_dir outputs/craft

echo "All attack scenes completed at $(date)" | tee -a $master_log