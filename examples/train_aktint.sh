#!/bin/bash

# List of dataset names
# datasets=("assist2012" "assist2017" "bridge2algebra2006" "nips_task34" "ednet")
datasets=("assist2015" "assist2017" "assist2012" "bridge2algebra2006" "forget_se")

rm -rf ../data/*/*.pkl

for dataset in "${datasets[@]}"; do
    echo "===== Running dataset: $dataset ====="
    python wandb_aktint_train.py --dataset_name="$dataset"
    echo "===== Finished dataset: $dataset ====="
done