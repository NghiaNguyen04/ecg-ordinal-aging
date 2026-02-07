python src/run.py \
    --root-dir "data/processed/seg_300s/data_300s_order5_rmNegativeRRI.csv" \
    --dataset-name "data_300s_order5" \
    --model "Resnet34_coral" \
    --log-dir "result/Coral_bmi_sex" \
    --batch-size 16 \
    --max-epochs 150 \
    --n-splits 5 \
    --use-sex \
    --use-bmi 

python src/run.py \
    --root-dir "data/processed/seg_300s/data_300s_order5_rmNegativeRRI.csv" \
    --dataset-name "data_300s_order5" \
    --model "Resnet34_coral" \
    --log-dir "result/Coral_bmi" \
    --batch-size 16 \
    --max-epochs 150 \
    --n-splits 5 \
    --use-bmi 

python src/run.py \
    --root-dir "data/processed/seg_300s/data_300s_order5_rmNegativeRRI.csv" \
    --dataset-name "data_300s_order5" \
    --model "Resnet34_coral" \
    --log-dir "result/Coral_sex" \
    --batch-size 16 \
    --max-epochs 150 \
    --n-splits 5 \
    --use-sex 
    
