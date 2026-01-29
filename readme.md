# 训练参数示例

## Scheme2 + 对比学习 + Col Loss 训练命令

### 禁用 Col Loss 的方法

如果不想使用协作学习损失（L_col），可以使用 `--disable_col` 参数：

```bash
python train.py \
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
    --disable_col \
    --lambda_dol 0.5 \
    --lambda_hier 1.5 \
    --lambda_scl 0.2 \
    --epochs 200 \
    --seed 42 \
    --save_log True
```

**注意**: 使用 `--disable_col` 后，`--lambda_col` 参数会被忽略，L_col 不会参与损失计算。

### 基础训练命令（CCDC数据集）

```bash
python train.py \
    --task ICL \
    --model ResNet1D_MoE \
    --dataset mp20 \
    --use_processed_npy \
    --processed_data_dir /opt/data/private/xrd2c_data \
    --dataset_name ccdc_sg \
    --rule_matrix_path /opt/data/private/ICL/rule_matrix.csv \
    --device 1 \
    --loss_scheme scheme2 \
    --use_contrast_scheme2 \
    --epochs 200 \
    --seed 42 \
    --save_log True
```

### 完整训练命令（包含所有损失权重参数）

```bash
python train.py \
    --task ICL \
    --model ResNet1D_MoE \
    --dataset mp20 \
    --use_processed_npy \
    --processed_data_dir /opt/data/private/xrd2c_data \
    --dataset_name ccdc_sg \
    --rule_matrix_path /opt/data/private/ICL/rule_matrix.csv \
    --device 1 \
    --loss_scheme scheme2 \
    --use_contrast_scheme2 \
    --lambda_dol 0.5 \
    --lambda_col 0.3 \
    --lambda_hier 1.5 \
    --lambda_scl 0.2 \
    --epochs 200 \
    --seed 42 \
    --save_log True
```

### 使用ViT前端的训练命令

```bash
python train.py \
    --task ICL \
    --model ResNet1D_MoE \
    --dataset mp20 \
    --use_processed_npy \
    --processed_data_dir /opt/data/private/xrd2c_data \
    --dataset_name ccdc_sg \
    --sg_count_path /opt/data/private/xrd2c_data/ccdc_sg_count.csv \
    --rule_matrix_path /opt/data/private/ICL/rule_matrix.csv \
    --device 0 \
    --loss_scheme scheme2 \
    --use_contrast_scheme2 \
    --frontend_type vit \
    --vit_patch_size 50 \
    --vit_embed_dim 256 \
    --vit_depth 6 \
    --vit_num_heads 8 \
    --vit_mlp_ratio 4.0 \
    --lambda_dol 0.5 \
    --lambda_col 0.3 \
    --lambda_hier 1.5 \
    --lambda_scl 0.2 \
    --epochs 200 \
    --seed 42 \
    --save_log True
```

python train.py \
    --use_processed_npy \
    --processed_data_dir /opt/data/private/xrd2c_data \
    --dataset_name inor \
    --sg_count_path /opt/data/private/xrd2c_data/inor_sg_count.csv \
    --rule_matrix_path /opt/data/private/ICL/rule_matrix.csv \
    --device 0 \
    --loss_scheme scheme2 \
    --use_contrast_scheme2 \
    --epochs 100 \
    --frontend_type vit \
    --vit_patch_size 50 \
    --vit_embed_dim 256 \
    --vit_depth 6 \
    --vit_num_heads 8 \
    --vit_mlp_ratio 4.0 \
    --vit_use_cls_token

### 使用AdamW优化器的训练命令

```bash
python train.py \
    --task ICL \
    --model ResNet1D_MoE \
    --dataset mp20 \
    --use_processed_npy \
    --processed_data_dir /opt/data/private/xrd2c_data \
    --dataset_name ccdc_sg \
    --rule_matrix_path /opt/data/private/ICL/rule_matrix.csv \
    --device 1 \
    --loss_scheme scheme2 \
    --use_contrast_scheme2 \
    --use_adamw \
    --lambda_dol 0.5 \
    --lambda_col 0.3 \
    --lambda_hier 1.5 \
    --lambda_scl 0.2 \
    --epochs 200 \
    --seed 42 \
    --save_log True
```

### 使用其他数据集（inor或coremof19）

```bash
# 使用inor数据集
python train.py \
    --task ICL \
    --model ResNet1D_MoE \
    --dataset mp20 \
    --use_processed_npy \
    --processed_data_dir /opt/data/private/xrd2c_data \
    --dataset_name inor \
    --rule_matrix_path /opt/data/private/ICL/rule_matrix.csv \
    --device 1 \
    --loss_scheme scheme2 \
    --use_contrast_scheme2 \
    --lambda_dol 0.5 \
    --lambda_col 0.3 \
    --lambda_hier 1.5 \
    --lambda_scl 0.2 \
    --epochs 200 \
    --seed 42 \
    --save_log True

# 使用coremof19数据集
python train.py \
    --task ICL \
    --model ResNet1D_MoE \
    --dataset mp20 \
    --use_processed_npy \
    --processed_data_dir /opt/data/private/xrd2c_data \
    --dataset_name coremof19 \
    --rule_matrix_path /opt/data/private/ICL/rule_matrix.csv \
    --device 1 \
    --loss_scheme scheme2 \
    --use_contrast_scheme2 \
    --lambda_dol 0.5 \
    --lambda_col 0.3 \
    --lambda_hier 1.5 \
    --lambda_scl 0.2 \
    --epochs 200 \
    --seed 42 \
    --save_log True
```

## 损失权重参数说明

### Scheme2损失组件

1. **L_dol** (独立优化损失)
   - 参数: `--lambda_dol`
   - 默认值: `0.5`
   - 作用: 让每个专家独立优化自己的预测
   - 建议范围: 0.3 - 0.8

2. **L_col** (协作学习损失) ⭐ **新增**
   - 参数: `--lambda_col`
   - 默认值: `0.3`
   - 作用: 让专家之间协作，通过KL散度让专家向集成输出看齐
   - 建议范围: 0.1 - 0.5

3. **L_hier** (层级化损失 = L_sys + L_sg)
   - 参数: `--lambda_hier`
   - 默认值: `1.5`
   - 作用: 晶系分类损失 + 空间群分类损失
   - 建议范围: 1.0 - 2.0

4. **L_scl** (对比学习损失)
   - 参数: `--lambda_scl`
   - 默认值: `0.2`
   - 作用: 物理约束的对比学习损失（仅在`--use_contrast_scheme2`时启用）
   - 建议范围: 0.1 - 0.5

### 总损失公式（Scheme2 + 对比学习）

```
Total Loss = lambda_dol * L_dol + 
             lambda_col * L_col + 
             lambda_hier * (L_sys + L_sg) + 
             lambda_scl * L_scl
```

## 训练优化参数

### 学习率策略

```bash
# 使用调整后的学习率策略（延迟早期衰减，提高目标LR）
python train.py \
    ... \
    --adjust_lr_strategy \
    ...
```

### 梯度裁剪

```bash
# 自定义梯度裁剪阈值（默认1.0）
python train.py \
    ... \
    --max_grad_norm 1.0 \
    --grad_warn_threshold 50.0 \
    ...
```

### 学习率重启

```bash
# 启用学习率重启（默认已启用）
python train.py \
    ... \
    --lr_restart_enabled \
    --lr_restart_threshold 20 \
    --lr_restart_factor 0.5 \
    ...
```

## 完整训练示例（推荐配置）

```bash
python train.py \
    --task ICL \
    --model ResNet1D_MoE \
    --dataset mp20 \
    --use_processed_npy \
    --processed_data_dir /opt/data/private/xrd2c_data \
    --dataset_name ccdc_sg \
    --rule_matrix_path /opt/data/private/ICL/rule_matrix.csv \
    --device 1 \
    --loss_scheme scheme2 \
    --use_contrast_scheme2 \
    --frontend_type resnet \
    --use_adamw \
    --adjust_lr_strategy \
    --lambda_dol 0.5 \
    --lambda_col 0.3 \
    --lambda_hier 1.5 \
    --lambda_scl 0.2 \
    --max_grad_norm 1.0 \
    --grad_warn_threshold 50.0 \
    --epochs 200 \
    --seed 42 \
    --save_log True
```

## 损失权重调优建议

### 初始配置
- `lambda_dol = 0.5`: 平衡独立优化
- `lambda_col = 0.3`: 适度的协作学习
- `lambda_hier = 1.5`: 强调层级化学习
- `lambda_scl = 0.2`: 适度的对比学习

### 如果训练不稳定
- 降低 `lambda_dol` 到 0.3-0.4
- 降低 `lambda_col` 到 0.1-0.2
- 增加 `lambda_hier` 到 2.0

### 如果尾部类性能差
- 增加 `lambda_col` 到 0.4-0.5（加强专家协作）
- 增加 `lambda_scl` 到 0.3-0.4（加强对比学习）

### 如果头部类性能差
- 增加 `lambda_dol` 到 0.6-0.8（加强独立优化）
- 降低 `lambda_col` 到 0.1-0.2

## 注意事项

1. **Col Loss已启用**: Scheme2现在包含协作学习损失（L_col），默认权重为0.3
2. **损失权重平衡**: 确保各损失组件的权重平衡，避免某个损失主导训练
3. **监控损失**: 训练时注意观察各个损失组件的数值，确保都在合理范围内
4. **梯度监控**: 如果出现梯度爆炸警告，可以降低损失权重或增加梯度裁剪阈值
