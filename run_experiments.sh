#!/bin/bash
# bash run_experiments.sh

echo "Starting all experiments sequentially..."

# 1. Hybrid Loss Experiments
python src/run_parallel_folds.py \
    --root-dir "data/processed/seg_300s/data_300s_order5.csv" \
    --model "Resnet34_hybrid" \
    --log-dir "result/hybrid_NoRmNegativeRRI" \
    --batch-size 16 \
    --max-epochs 150 \
    --n-splits 5 \
    --precision 16-mixed --num-workers 4

sleep 10

python src/run_parallel_folds.py \
    --root-dir "data/processed/seg_300s/data_300s_order5_rmNegativeRRI.csv" \
    --model "Resnet34_hybrid" \
    --log-dir "result/hybrid" \
    --batch-size 16 \
    --max-epochs 150 \
    --n-splits 5 \
    --precision 16-mixed --num-workers 4

sleep 10

python src/run_parallel_folds.py \
    --root-dir "data/processed/seg_300s/data_300s_order5_rmNegativeRRI.csv" \
    --model "Resnet34_hybrid" \
    --log-dir "result/hybrid_bmi_sex" \
    --batch-size 16 \
    --max-epochs 150 \
    --n-splits 5 \
    --use-bmi --use-sex \
    --precision 16-mixed --num-workers 4

sleep 10

python src/run_parallel_folds.py \
    --root-dir "data/processed/seg_300s/data_300s_order5_rmNegativeRRI.csv" \
    --model "Resnet34_hybrid" \
    --log-dir "result/hybrid_bmi" \
    --batch-size 16 \
    --max-epochs 150 \
    --n-splits 5 \
    --use-bmi \
    --precision 16-mixed --num-workers 4

sleep 10

python src/run_parallel_folds.py \
    --root-dir "data/processed/seg_300s/data_300s_order5_rmNegativeRRI.csv" \
    --model "Resnet34_hybrid" \
    --log-dir "result/hybrid_sex" \
    --batch-size 16 \
    --max-epochs 150 \
    --n-splits 5 \
    --use-sex \
    --precision 16-mixed --num-workers 4

sleep 10

# 2. FocalCos Experiments
python src/run_parallel_folds.py \
    --root-dir "data/processed/seg_300s/data_300s_order5_rmNegativeRRI.csv" \
    --model "Resnet34_FocalCos" \
    --log-dir "result/FocalCos" \
    --batch-size 16 \
    --max-epochs 150 \
    --n-splits 5 \
    --use-uncertainty-weighting \
    --precision 16-mixed --num-workers 4

sleep 10

python src/run_parallel_folds.py \
    --root-dir "data/processed/seg_300s/data_300s_order5_rmNegativeRRI.csv" \
    --model "Resnet34_FocalCos" \
    --log-dir "result/FocalCos_bmi_sex" \
    --batch-size 16 \
    --max-epochs 150 \
    --n-splits 5 \
    --use-uncertainty-weighting --use-sex --use-bmi \
    --precision 16-mixed --num-workers 4

sleep 10

python src/run_parallel_folds.py \
    --root-dir "data/processed/seg_300s/data_300s_order5_rmNegativeRRI.csv" \
    --model "Resnet34_FocalCos" \
    --log-dir "result/FocalCos_bmi" \
    --batch-size 16 \
    --max-epochs 150 \
    --n-splits 5 \
    --use-uncertainty-weighting --use-bmi \
    --precision 16-mixed --num-workers 4

sleep 10

# 3. Coral Loss Experiments
python src/run_parallel_folds.py \
    --root-dir "data/processed/seg_300s/data_300s_order5_rmNegativeRRI.csv" \
    --model "Resnet34_coral" \
    --log-dir "result/Resnet34_coral" \
    --batch-size 16 \
    --max-epochs 150 \
    --n-splits 5 \
    --precision 16-mixed --num-workers 4

sleep 10

python src/run_parallel_folds.py \
    --root-dir "data/processed/seg_300s/data_300s_order5_rmNegativeRRI.csv" \
    --model "Resnet34_coral" \
    --log-dir "result/Resnet34_coral_bmi_sex" \
    --batch-size 16 \
    --max-epochs 150 \
    --n-splits 5 \
    --use-sex --use-bmi \
    --precision 16-mixed --num-workers 4

sleep 10

python src/run_parallel_folds.py \
    --root-dir "data/processed/seg_300s/data_300s_order5_rmNegativeRRI.csv" \
    --model "Resnet34_coral" \
    --log-dir "result/Resnet34_coral_bmi" \
    --batch-size 16 \
    --max-epochs 150 \
    --n-splits 5 \
    --use-bmi \
    --precision 16-mixed --num-workers 4

sleep 10

python src/run_parallel_folds.py \
    --root-dir "data/processed/seg_300s/data_300s_order5_rmNegativeRRI.csv" \
    --model "Resnet34_coral" \
    --log-dir "result/Resnet34_coral_sex" \
    --batch-size 16 \
    --max-epochs 150 \
    --n-splits 5 \
    --use-sex \
    --precision 16-mixed --num-workers 4

echo "All experiments completed!"
