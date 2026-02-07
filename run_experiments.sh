
# ------------------------------
# 1. Experiment: hybrid loss
# ------------------------------

python src/run.py \
    --root-dir "data/processed/seg_300s/data_300s_order5_rmNegativeRRI.csv" \
    --dataset-name "data_300s_order5" \
    --model "Resnet34_hybrid" \
    --log-dir "result/hybrid" \
    --batch-size 16 \
    --max-epochs 150 \
    --n-splits 5 


python src/run.py \
    --root-dir "data/processed/seg_300s/data_300s_order5_rmNegativeRRI.csv" \
    --dataset-name "data_300s_order5" \
    --model "Resnet34_hybrid" \
    --log-dir "result/hybrid_bmi_sex" \
    --batch-size 16 \
    --max-epochs 150 \
    --n-splits 5 \
    --use-bmi \
    --use-sex


python src/run.py \
    --root-dir "data/processed/seg_300s/data_300s_order5_rmNegativeRRI.csv" \
    --dataset-name "data_300s_order5" \
    --model "Resnet34_hybrid" \
    --log-dir "result/hybrid_bmi" \
    --batch-size 16 \
    --max-epochs 150 \
    --n-splits 5 \
    --use-bmi 

python src/run.py \
    --root-dir "data/processed/seg_300s/data_300s_order5_rmNegativeRRI.csv" \
    --dataset-name "data_300s_order5" \
    --model "Resnet34_hybrid" \
    --log-dir "result/hybrid_sex" \
    --batch-size 16 \
    --max-epochs 150 \
    --n-splits 5 \
    --use-sex


# ------------------------------
# 2. Experiment: FocalCosLoss
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


python src/run.py \
    --root-dir "data/processed/seg_300s/data_300s_order5_rmNegativeRRI.csv" \
    --dataset-name "data_300s_order5" \
    --model "Resnet34_FocalCos" \
    --log-dir "result/FocalCos_bmi_sex" \
    --batch-size 16 \
    --max-epochs 150 \
    --n-splits 5 \
    --use-uncertainty-weighting \
    --use-sex \
    --use-bmi \


python src/run.py \
    --root-dir "data/processed/seg_300s/data_300s_order5_rmNegativeRRI.csv" \
    --dataset-name "data_300s_order5" \
    --model "Resnet34_FocalCos" \
    --log-dir "result/FocalCos_bmi" \
    --batch-size 16 \
    --max-epochs 150 \
    --n-splits 5 \
    --use-uncertainty-weighting \
    --use-bmi


python src/run.py \
    --root-dir "data/processed/seg_300s/data_300s_order5_rmNegativeRRI.csv" \
    --dataset-name "data_300s_order5" \
    --model "Resnet34_FocalCos" \
    --log-dir "result/FocalCos_sex" \
    --batch-size 16 \
    --max-epochs 150 \
    --n-splits 5 \
    --use-uncertainty-weighting \
    --use-sex



# ------------------------------
# 3. Experiment: CoralLoss
# ------------------------------

python src/run.py \
    --root-dir "data/processed/seg_300s/data_300s_order5_rmNegativeRRI.csv" \
    --dataset-name "data_300s_order5" \
    --model "Resnet34_coral" \
    --log-dir "result/Resnet34_coral" \
    --batch-size 16 \
    --max-epochs 150 \
    --n-splits 5

python src/run.py \
    --root-dir "data/processed/seg_300s/data_300s_order5_rmNegativeRRI.csv" \
    --dataset-name "data_300s_order5" \
    --model "Resnet34_coral" \
    --log-dir "result/Resnet34_coral_bmi_sex" \
    --batch-size 16 \
    --max-epochs 150 \
    --n-splits 5 \
    --use-sex \
    --use-bmi 

python src/run.py \
    --root-dir "data/processed/seg_300s/data_300s_order5_rmNegativeRRI.csv" \
    --dataset-name "data_300s_order5" \
    --model "Resnet34_coral" \
    --log-dir "result/Resnet34_coral_bmi" \
    --batch-size 16 \
    --max-epochs 150 \
    --n-splits 5 \
    --use-bmi 

python src/run.py \
    --root-dir "data/processed/seg_300s/data_300s_order5_rmNegativeRRI.csv" \
    --dataset-name "data_300s_order5" \
    --model "Resnet34_coral" \
    --log-dir "result/Resnet34_coral_sex" \
    --batch-size 16 \
    --max-epochs 150 \
    --n-splits 5 \
    --use-sex 
    
