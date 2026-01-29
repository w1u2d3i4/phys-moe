# Loss 模块介绍

## 概述

Loss 模块包含了用于训练 ResNet1D_MoE 模型的所有损失函数。该模块设计用于解决极端长尾分布问题，通过物理规则约束和混合专家（MoE）机制来提升模型性能。

## 模块结构

```
loss/
├── __init__.py                    # 模块初始化文件
├── moe_loss.py                    # MoE相关损失函数
├── physics_contrast_loss.py       # 物理规则约束的损失函数
└── HNM.py                         # 硬负样本挖掘模块
```

## 1. moe_loss.py

### 1.1 soft_entropy

**功能**: 支持软标签的交叉熵损失函数

**参数**:
- `input`: 模型预测的logits `[batch_size, num_classes]`
- `target`: 目标标签，可以是硬标签或软标签（概率分布）`[batch_size, num_classes]`
- `reduction`: 损失归约方式，'mean' 或 'sum'

**公式**:
```
loss = -mean(sum(target * log_softmax(input), dim=1))
```

**特点**:
- 支持软标签（概率分布）作为目标
- 适用于知识蒸馏等场景

### 1.2 mix_outputs

**功能**: 混合多个专家输出的损失计算

**参数**:
- `outputs`: 专家输出列表 `List[[batch_size, num_classes]]`
- `labels`: 真实标签 `[batch_size]`
- `balance`: 是否使用类别平衡（默认False）
- `label_dis`: 类别分布列表（用于平衡）

**返回**:
- `loss`: 总损失
- `avg_output`: 平均输出
- `loss_distillation`: 蒸馏损失
- `loss_parallel`: 并行损失

**损失组成**:

1. **基础交叉熵损失**:
   ```python
   loss = sum(soft_entropy(outputs[i], labels)) for i in range(num_experts)
   ```

2. **蒸馏损失** (Distillation Loss):
   ```python
   loss_distillation = sum(KL_div(softmax(outputs[i]), softmax(outputs[j]))) 
                       for i != j
   ```
   - 鼓励专家之间学习相似的分布

3. **并行损失** (Parallel Loss):
   ```python
   # 基于每个专家CE损失的倒数加权
   reciprocal_sum = sum(1 / (ce_i + eps))
   fused_output = 1 / reciprocal_sum
   loss_parallel = fused_output * num_experts
   ```
   - 自适应平衡不同专家的性能

4. **类别平衡模式** (balance=True):
   ```python
   loss = sum(soft_entropy(outputs[i] + log(label_dis), labels))
   ```
   - 通过添加类别分布的对数来平衡长尾分布

### 1.3 feature_diversity_loss

**功能**: 特征多样性损失，鼓励不同专家学习不同的特征表示

**参数**:
- `features`: 专家特征列表 `List[[batch_size, feature_dim]]`
- `lambda_reg`: 正则化权重（默认0.01）

**计算过程**:
1. 将所有专家特征堆叠: `[batch_size, num_experts, feature_dim]`
2. 计算所有特征对之间的成对距离（使用 `torch.cdist`）
3. 排除自身距离（对角线）
4. 计算平均距离作为损失

**公式**:
```
loss = lambda_reg * mean(pairwise_distances(expert_features))
```

**作用**:
- 防止专家坍塌（所有专家学习相同特征）
- 鼓励专家专业化，学习互补的特征

## 2. physics_contrast_loss.py

### 2.1 PhysicsContrastiveLoss

**功能**: 物理规则加权的对比学习损失

**设计理念**: 
- 将"正样本"从"同类样本"扩展为"物理语义相似的样本"
- 通过规则相似度矩阵引入领域知识

**参数**:
- `temperature`: 温度参数（默认0.07）
- `rule_threshold`: 规则相似度阈值（默认0.8）

**输入**:
- `features`: 归一化的特征 `[batch_size, feature_dim]`
- `labels`: 类别标签 `[batch_size]`
- `rule_matrix`: 规则相似度矩阵 `[num_classes, num_classes]`

**计算流程**:

1. **特征相似度矩阵**:
   ```python
   sim_matrix = features @ features.T / temperature
   sim_matrix = sim_matrix - max(sim_matrix, dim=1)  # 数值稳定性
   ```

2. **规则相似度映射**:
   ```python
   batch_rule_sim = rule_matrix[labels][:, labels]  # [batch, batch]
   ```

3. **正样本定义**:
   ```python
   is_same_class = labels[i] == labels[j]
   is_rule_similar = batch_rule_sim[i,j] > threshold
   mask_pos = (is_same_class | is_rule_similar) & (i != j)
   ```

4. **规则加权**:
   ```python
   neg_weights = 1.0 - batch_rule_sim  # 相似度越高，权重越小
   weighted_exp = exp(sim_matrix) * neg_weights
   ```

5. **损失计算**:
   ```python
   log_prob = sim_matrix - log(sum(weighted_exp))
   loss = -mean(sum(log_prob * mask_pos) / sum(mask_pos))
   ```

**特点**:
- 软正样本对：规则相似度 > 0.8 即视为正样本
- 规则加权：物理上相似的负样本不会被过度排斥
- 保持特征空间的物理连续性

### 2.2 ICLMoELoss

**功能**: 独立协作学习（ICL）的MoE损失

**组成**: `L_moe = L_dol + L_col`

#### 2.2.1 独立优化损失 (L_dol: Division of Labor Loss)

**设计**: 不同专家使用不同的损失函数

- **首部专家** (Head Experts): 标准交叉熵损失
  ```python
  loss = CrossEntropy(logits, labels)
  ```

- **尾部专家** (Tail Experts): Focal Loss
  ```python
  ce_loss = CrossEntropy(logits, labels, reduction='none')
  pt = exp(-ce_loss)
  focal_loss = ((1 - pt) ** gamma) * ce_loss
  ```
  - `gamma=2.0`: Focal Loss 的聚焦参数
  - 更关注难分类样本

#### 2.2.2 协作学习损失 (L_col: Collaboration Loss)

**设计**: 通过KL散度让专家向集成输出看齐

```python
ensemble_logits = weighted_sum(expert_logits, gating_weights)
ensemble_probs = softmax(ensemble_logits)

for each expert:
    expert_probs = log_softmax(expert_logits)
    loss_col += KL_div(expert_probs, ensemble_probs)
```

**作用**:
- 知识蒸馏：让每个专家学习集成输出的分布
- 防止专家偏离：保持专家输出的一致性

### 2.3 HierarchicalLoss

**功能**: 层级化约束损失，利用晶体学的层级结构

**组成**: `L_hier = L_sys + L_sg`

#### 2.3.1 晶系分类损失 (L_sys)

```python
sys_labels = sg_to_sys_map[labels]  # 230类 -> 7类
loss_sys = CrossEntropy(sys_logits, sys_labels)
```

#### 2.3.2 空间群分类损失 (L_sg) - 带掩码

**掩码机制**:
```python
# 只允许属于真实晶系的空间群参与计算
valid_mask = (sg_to_sys_map == target_sys)
masked_logits = sg_logits.clone()
masked_logits[~valid_mask] = -1e9  # 掩码无效类别
loss_sg = CrossEntropy(masked_logits, labels)
```

**优势**:
- 将230类任务分解为7个子任务
- 降低尾部类的干扰
- 利用层级先验知识

### 2.4 CenterRegularizationLoss

**功能**: 特征中心正则化，防止尾部类特征中心漂移

**组成**: `L_reg = L_center + 0.5 * L_prototype`

#### 2.4.1 中心损失 (L_center)

```python
normalized_centers = normalize(centers, dim=1)
batch_centers = normalized_centers[labels]
cosine_sim = (features * batch_centers).sum(dim=1)
loss_center = (1 - cosine_sim).mean()
```

- 使用余弦距离（特征已归一化）
- 拉近样本到对应类别的特征中心

#### 2.4.2 原型约束损失 (L_prototype)

```python
# 基于规则相似度计算理论中心
sim_weights = rule_matrix.clone()
sim_weights.fill_diagonal_(0)
norm_weights = sim_weights / sum(sim_weights)
expected_centers = norm_weights @ normalized_centers

# 约束实际中心接近理论中心
cosine_sim_reg = (normalized_centers * expected_centers).sum(dim=1)
loss_reg = (1 - cosine_sim_reg).mean()
```

**设计理念**:
- 尾部类的中心应向规则相似的首部类中心看齐
- 利用首部类的丰富数据"锚定"尾部类特征

**初始化**:
```python
centers = Parameter(randn(num_classes, feature_dim) * 0.1)
```
- 使用较小的初始化值，避免初始损失过大

### 2.5 PhysMoELossWrapper

**功能**: 完整的物理规则约束损失包装器（包含对比学习）

**总损失公式**:
```
L_total = λ_scl * L_scl + L_dol + L_col + λ_hier * (L_sys + L_sg) + λ_reg * L_reg
```

**权重配置**:
- `lambda_scl = 0.1`: 对比学习损失权重
- `lambda_hier = 1.0`: 层级化损失权重
- `lambda_reg = 0.001`: 中心正则化权重（已降低，避免损失过大）

**输入**:
```python
outputs = {
    'features': [B, D],           # 特征
    'sys_logits': [B, 7],         # 晶系logits
    'expert_logits': List[[B, 230]],  # 专家logits列表
    'gating_weights': [B, K]      # 门控权重
}
targets = {
    'labels': [B]                 # 空间群标签 (0-229)
}
```

**输出**:
- `total_loss`: 总损失
- `loss_dict`: 各组件损失的字典

### 2.6 PhysMoELossWrapperNoContrast

**功能**: 无对比学习版本的损失包装器

**总损失公式**:
```
L_total = L_dol + L_col + λ_hier * (L_sys + L_sg) + λ_reg * L_reg
```

**与完整版本的区别**:
- 不包含 `PhysicsContrastiveLoss`
- 其他组件完全相同
- 适用于不需要对比学习的场景

## 3. HNM.py

### 3.1 HardNegativeMining_Proto

**功能**: 基于原型的硬负样本挖掘

**核心组件**:

1. **原型 (Prototypes)**:
   ```python
   prototypes = Parameter(zeros(num_classes, feature_dim))
   ```
   - 每个类别维护一个特征原型
   - 使用动量更新：`prototype = momentum * old + (1-momentum) * new_mean`

2. **混淆矩阵 (Confusion Matrix)**:
   ```python
   confusion_matrix = zeros(num_classes, num_classes)
   ```
   - 记录类别间的混淆情况
   - 用于识别容易混淆的类别（硬负样本）

**主要方法**:

#### 3.1.1 update_prototypes

**功能**: 更新类别原型

```python
for each unique label:
    feature_mean = mean(features[label == label])
    prototypes[label] = momentum * old + (1-momentum) * feature_mean
```

#### 3.1.2 update_confusion_matrix

**功能**: 更新混淆矩阵

```python
confusion_matrix[true_label, predicted_label] += 1
```

#### 3.1.3 get_hard_negative_classes

**功能**: 获取硬负样本类别

```python
for each label:
    hard_classes = topk(confusion_matrix[label], k=k)
```
- 返回每个类别最容易被混淆的k个类别

#### 3.1.4 compute_contrastive_loss

**功能**: 计算对比学习损失（使用硬负样本）

**计算流程**:
1. 归一化特征和原型
2. 计算正样本相似度：`pos_sim = cosine(features, prototypes[labels])`
3. 获取硬负样本原型：`hard_negatives = get_hard_negative_classes(labels)`
4. 计算负样本相似度：`neg_sim = cosine(features, prototypes[hard_negatives])`
5. 计算损失：
   ```python
   loss = -log(exp(pos_sim/t) / (exp(pos_sim/t) + mean(exp(neg_sim/t))))
   ```

**优势**:
- 关注难样本：只使用容易混淆的负样本
- 提高训练效率：减少计算量
- 动态更新：混淆矩阵随训练更新

#### 3.1.5 apply_epoch_momentum

**功能**: 在每个epoch结束时应用动量衰减

```python
confusion_matrix *= (1 - momentum)
```

- 防止混淆矩阵过度累积
- 保持对最近混淆模式的敏感性

## 损失函数总览

### 完整版本 (PhysMoELossWrapper)

```
L_total = 0.1 * L_contrastive      # 对比学习
        + L_dol + L_col            # MoE核心损失
        + 1.0 * (L_sys + L_sg)     # 层级化损失
        + 0.001 * L_reg             # 中心正则化
```

### 无对比学习版本 (PhysMoELossWrapperNoContrast)

```
L_total = L_dol + L_col            # MoE核心损失
        + 1.0 * (L_sys + L_sg)     # 层级化损失
        + 0.001 * L_reg             # 中心正则化
```

## 设计特点

### 1. 多层次损失设计
- **层级化**: 晶系 → 空间群的两级分类
- **专家级**: 每个专家独立优化 + 协作学习
- **特征级**: 对比学习 + 中心正则化

### 2. 物理规则约束
- **规则矩阵**: 引入领域知识（晶系、点群、消光规则）
- **软正样本**: 扩展正样本定义
- **规则加权**: 保护物理相似性

### 3. 长尾分布处理
- **Focal Loss**: 尾部专家使用Focal Loss
- **层级掩码**: 减少尾部类干扰
- **中心正则化**: 利用首部类数据锚定尾部类

### 4. 数值稳定性
- **归一化**: 特征和中心都归一化
- **余弦距离**: 使用余弦相似度而非MSE
- **掩码值优化**: 避免使用-1e9等极端值
- **梯度裁剪**: 在训练器中添加梯度裁剪

## 使用示例

### 完整版本（带对比学习）

```python
from loss.physics_contrast_loss import PhysMoELossWrapper

cfg = SimpleNamespace(num_experts=8, tail_expert_ids=[4,5,6,7], feat_dim=1024)
loss_wrapper = PhysMoELossWrapper(cfg, rule_matrix, sg_to_sys_map)

outputs = {
    'features': features,           # [B, 1024]
    'sys_logits': sys_logits,      # [B, 7]
    'expert_logits': expert_logits_list,  # List of [B, 230]
    'gating_weights': gating_weights  # [B, 8]
}
targets = {'labels': labels}      # [B]

total_loss, loss_dict = loss_wrapper(outputs, targets)
```

### 无对比学习版本

```python
from loss.physics_contrast_loss import PhysMoELossWrapperNoContrast

cfg = SimpleNamespace(num_experts=8, tail_expert_ids=[4,5,6,7], feat_dim=1024)
loss_wrapper = PhysMoELossWrapperNoContrast(cfg, rule_matrix, sg_to_sys_map)

total_loss, loss_dict = loss_wrapper(outputs, targets)
```

## 关键参数说明

### 温度参数 (Temperature)
- **对比学习**: `temperature=0.07` - 控制相似度分布的尖锐程度
- **HNM**: `temperature=0.07` - 控制硬负样本的权重

### 规则阈值 (Rule Threshold)
- `rule_threshold=0.8` - 规则相似度超过此值视为正样本

### Focal Loss 参数
- `gamma=2.0` - 控制难样本的权重

### 动量参数
- **原型更新**: `momentum=0.9` - 原型更新的平滑系数
- **混淆矩阵**: `momentum=0.9` - 混淆矩阵的衰减系数

### 损失权重
- `lambda_scl = 0.1`: 对比学习权重
- `lambda_hier = 1.0`: 层级化损失权重
- `lambda_reg = 0.001`: 中心正则化权重（已优化）

## 注意事项

1. **特征归一化**: 所有特征在使用前都应归一化
2. **规则矩阵**: 需要预计算并加载规则相似度矩阵
3. **类别映射**: 需要提供空间群到晶系的映射表
4. **数值稳定性**: 已优化掩码值和损失计算，避免数值溢出
5. **梯度裁剪**: 建议在训练器中添加梯度裁剪（max_norm=1.0）

## 总结

Loss 模块提供了一个完整的、针对长尾分布和领域知识的损失函数体系。通过物理规则约束、层级化设计、专家协作和特征正则化，有效解决了晶体空间群分类中的极端长尾分布问题。
