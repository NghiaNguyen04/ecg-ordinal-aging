# ------------------------------
# 1. Experiment: CoralLoss
# ------------------------------

python src/run.py \
    --root-dir "data/processed/seg_300s/data_300s_order5_rmNegativeRRI.csv" \
    --dataset-name "data_300s_order5" \
    --model "Resnet34_coral" \
    --log-dir "result/Resnet34_coral" \
    --batch-size 16 \
    --max-epochs 150 \
    --n-splits 5


# ------------------------------
# 2. Experiment: hybrid loss
# ------------------------------

python src/run.py \
    --root-dir "data/processed/seg_300s/data_300s_order5_rmNegativeRRI.csv" \
    --dataset-name "data_300s_order5" \
    --model "Resnet34_hybrid" \
    --log-dir "result/hybrid" \
    --batch-size 16 \
    --max-epochs 150 \
    --n-splits 5 

# ------------------------------
# 3. Experiment: FocalCosLoss
# ------------------------------
python src/run.py \
    --root-dir "data/processed/seg_300s/data_300s_order5_rmNegativeRRI.csv" \
    --dataset-name "data_300s_order5" \
    --model "Resnet34_FocalCos" \
    --log-dir "result/FocalCos" \
    --batch-size 16 \
    --max-epochs 150 \
    --n-splits 5 \
    --use-uncertainty-weighting


