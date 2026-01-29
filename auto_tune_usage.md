# 自动调参系统使用说明

本系统提供了两种自动调参方案：

## 方案1：批量调参（auto_tune.py）

每次训练完成后，根据结果调整参数并重新训练。

### 使用方法

```bash
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
    --max_trials 5 \
    --min_epochs 10 \
    --patience 5
```

### 参数说明

- `--base_cmd`: 训练脚本路径
- `--base_args`: 基础训练参数（不包含超参数）
- `--lambda_*`: 初始损失权重
- `--max_trials`: 最大调参次数（默认5次）
- `--min_epochs`: 最小训练轮数，之后才开始调参（默认10）
- `--patience`: 耐心值，多少轮没有改进后调整参数（默认5）

### 工作原理

1. 启动训练并监控终端输出
2. 解析每个epoch的指标（loss, accuracy等）
3. 如果检测到性能停滞（patience个epoch没有改进），分析问题类型：
   - **尾部类性能差**: 增加协作学习和对比学习权重
   - **头部类性能差**: 增加独立优化权重
   - **性能停滞**: 随机调整权重组合
4. 训练完成后，根据结果调整参数并重新训练
5. 重复上述过程，直到达到最大调参次数

## 方案2：实时调参（auto_tune_realtime.py）

实时监控训练输出，动态调整参数（通过配置文件与训练进程通信）。

### 使用方法

```bash
python auto_tune_realtime.py \
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
        --epochs 200 \
    --lambda_dol 0.5 \
    --lambda_col 0.3 \
    --lambda_hier 1.5 \
    --lambda_scl 0.2 \
    --min_epochs 5 \
    --patience 3 \
    --check_interval 30.0 \
    --config_file /tmp/auto_tune_config.json
```

### 参数说明

- `--config_file`: 配置文件路径，用于与训练进程通信（默认：/tmp/auto_tune_config.json）
- `--check_interval`: 检查间隔（秒），默认30秒
- 其他参数同方案1

### 工作原理

1. 启动训练进程并实时监控输出
2. 解析每个epoch的指标
3. 如果检测到性能问题，立即更新配置文件
4. 训练进程可以在每个epoch读取配置文件并应用新参数
5. 持续监控直到训练完成

### 注意

实时调参需要训练代码支持从配置文件读取参数。如果训练代码不支持，可以使用方案1（批量调参）。

## 调参策略说明

### 问题类型识别

系统会自动识别以下问题类型：

1. **尾部类性能差** (`poor_tail`)
   - 条件: `tail_acc < 0.3` 且 `head_acc > 0.6`
   - 策略: 增加 `lambda_col` 和 `lambda_scl`

2. **头部类性能差** (`poor_head`)
   - 条件: `head_acc < 0.5` 且 `tail_acc > 0.4`
   - 策略: 增加 `lambda_dol`，降低 `lambda_col`

3. **损失过高** (`high_loss`)
   - 条件: `loss > 2.0`
   - 策略: 增加 `max_grad_norm`，降低 `lambda_dol`

4. **性能停滞** (`stagnation`)
   - 条件: 其他情况
   - 策略: 微调权重组合

### 参数搜索空间

默认搜索空间：
- `lambda_dol`: [0.3, 0.4, 0.5, 0.6, 0.7]
- `lambda_col`: [0.1, 0.2, 0.3, 0.4, 0.5]
- `lambda_hier`: [1.0, 1.2, 1.5, 1.8, 2.0]
- `lambda_scl`: [0.1, 0.15, 0.2, 0.25, 0.3]
- `max_grad_norm`: [0.5, 1.0, 1.5, 2.0]

## 示例输出

```
开始第 1/5 次训练
训练命令: python train.py --task ICL ...
Epoch 0[123.45s]: {'epoch': 0, 'ce': 2.3456, 'acc1': 0.1234, ...}
Epoch 1[125.67s]: {'epoch': 1, 'ce': 2.1234, 'acc1': 0.2345, ...}
...
Epoch 10[150.23s]: {'epoch': 10, 'ce': 1.5678, 'acc1': 0.4567, ...}
检测到性能停滞，调整损失权重...
新配置: lambda_dol=0.6, lambda_col=0.4, lambda_hier=1.8, lambda_scl=0.25
训练完成，最终best_metric: 0.5678

开始第 2/5 次训练
...
```

## 注意事项

1. **训练时间**: 批量调参会多次重新训练，总时间较长
2. **资源占用**: 确保有足够的GPU/CPU资源
3. **日志保存**: 每次训练的日志会保存在各自的目录中
4. **最佳配置**: 系统会输出最佳配置，可以手动使用该配置进行最终训练

## 进阶使用

### 自定义搜索空间

可以修改 `AutoTuner._default_search_space()` 方法来自定义搜索空间。

### 自定义调参策略

可以修改 `AutoTuner.tune()` 和 `AdaptiveTuner.tune()` 方法来实现自定义的调参策略。

### 集成到训练代码

如果要使用实时调参，需要在训练代码中添加读取配置文件的逻辑：

```python
import json
import os

config_file = '/tmp/auto_tune_config.json'
if os.path.exists(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)
    args.model.lambda_dol = config.get('lambda_dol', args.model.lambda_dol)
    args.model.lambda_col = config.get('lambda_col', args.model.lambda_col)
    # ... 其他参数
```
