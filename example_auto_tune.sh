#!/bin/bash
# 自动调参示例脚本

# 方案1：批量调参（推荐用于快速测试）
python auto_tune.py \
    --base_cmd train.py \
    --base_args \
        --task ICL \
        --model ResNet1D_MoE \
        --dataset mp20 \
        --use_processed_npy \
        --processed_data_dir /opt/data/private/xrd2c_data \
        --dataset_name ccdc_sg \
        --rule_matrix_path /opt/data/private/ICL/rule_matrix.csv \
        --device 0 \
        --loss_scheme scheme2 \
        --use_contrast_scheme2 \
        --use_adamw \
        --adjust_lr_strategy \
        --epochs 50 \
    --lambda_dol 0.5 \
    --lambda_col 0.3 \
    --lambda_hier 1.5 \
    --lambda_scl 0.2 \
    --max_trials 3 \
    --min_epochs 5 \
    --patience 3

# 方案2：实时调参（需要训练代码支持配置文件读取）
# python auto_tune_realtime.py \
#     --base_cmd train.py \
#     --base_args \
#         --task ICL \
#         --model ResNet1D_MoE \
#         --dataset mp20 \
#         --use_processed_npy \
#         --processed_data_dir /opt/data/private/xrd2c_data \
#         --dataset_name ccdc_sg \
#         --rule_matrix_path /opt/data/private/ICL/rule_matrix.csv \
#         --device 0 \
#         --loss_scheme scheme2 \
#         --use_contrast_scheme2 \
#         --use_adamw \
#         --adjust_lr_strategy \
#         --epochs 200 \
#     --lambda_dol 0.5 \
#     --lambda_col 0.3 \
#     --lambda_hier 1.5 \
#     --lambda_scl 0.2 \
#     --min_epochs 5 \
#     --patience 3 \
#     --check_interval 30.0
